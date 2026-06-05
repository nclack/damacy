#include "store/store_fs.h"

#include "damacy_limits.h"
#include "log/log.h"
#include "store/store.h"
#include "store/store_internal.h"
#include "util/hash.h"
#include "util/lru.h"
#include "util/pool.h"
#include "util/prelude.h"
#include "util/strbuf.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define FS_FD_CACHE_MAX_PROBE 16u

struct fs_cache_entry
{
  char* key;
  platform_file* file;
};

struct fs_submit_state
{
  atomic_uint refs;
  atomic_int status;
};

static struct fs_submit_state*
fs_submit_state_new(void)
{
  struct fs_submit_state* st = (struct fs_submit_state*)calloc(1, sizeof(*st));
  if (!st)
    return NULL;
  atomic_init(&st->refs, 1);
  atomic_init(&st->status, DAMACY_OK);
  return st;
}

static void
fs_submit_state_ref(struct fs_submit_state* st)
{
  atomic_fetch_add_explicit(&st->refs, 1, memory_order_relaxed);
}

static void
fs_submit_state_drop(struct fs_submit_state* st)
{
  if (!st)
    return;
  if (atomic_fetch_sub_explicit(&st->refs, 1, memory_order_acq_rel) == 1)
    free(st);
}

static void
fs_submit_state_fail(struct fs_submit_state* st)
{
  if (st)
    atomic_store_explicit(&st->status, DAMACY_IO, memory_order_release);
}

static enum damacy_status
fs_submit_state_finish(struct store_event ev)
{
  struct fs_submit_state* st = (struct fs_submit_state*)ev.impl;
  if (!st)
    return DAMACY_OK;
  enum damacy_status status =
    (enum damacy_status)atomic_load_explicit(&st->status, memory_order_acquire);
  fs_submit_state_drop(st);
  return status;
}

static int
fs_cache_eq(const void* value, const void* probe_key, void* user)
{
  (void)user;
  const struct fs_cache_entry* e = (const struct fs_cache_entry*)value;
  return strcmp(e->key, (const char*)probe_key) == 0;
}

static void
fs_cache_destroy(void* value, void* user)
{
  (void)user;
  struct fs_cache_entry* e = (struct fs_cache_entry*)value;
  if (!e)
    return;
  platform_file_close(e->file);
  free(e->key);
  free(e);
}

platform_file*
store_fs_acquire(struct store_fs* fs,
                 const char* key,
                 struct lru_entry** pin_out)
{
  if (pin_out)
    *pin_out = NULL;
  if (!fs || !key || !pin_out)
    return NULL;

  uint64_t hash = hash_fnv1a_str(key);

  platform_mutex_lock(fs->cache_lock);
  {
    struct lru_entry* hit = lru_get(fs->fd_cache, hash, key);
    if (hit) {
      lru_entry_acquire_locked(hit);
      platform_file* f = ((struct fs_cache_entry*)lru_entry_value(hit))->file;
      *pin_out = hit;
      platform_mutex_unlock(fs->cache_lock);
      return f;
    }
  }
  platform_mutex_unlock(fs->cache_lock);

  platform_file* opened = NULL;
  {
    struct strbuf path = { 0 };
    CHECK_SILENT(OpenDone, strbuf_join_path(&path, fs->root, key) == 0);
    opened = platform_file_open_read(strbuf_cstr(&path), 0);
  OpenDone:
    strbuf_free(&path);
  }
  if (!opened)
    return NULL;

  struct fs_cache_entry* entry =
    (struct fs_cache_entry*)calloc(1, sizeof(*entry));
  if (!entry) {
    platform_file_close(opened);
    return NULL;
  }
  entry->file = opened;
  entry->key = strdup(key);
  if (!entry->key) {
    fs_cache_destroy(entry, NULL);
    return NULL;
  }

  platform_mutex_lock(fs->cache_lock);
  {
    struct lru_entry* existing = lru_peek(fs->fd_cache, hash, key);
    if (existing) {
      lru_entry_acquire_locked(existing);
      platform_file* f =
        ((struct fs_cache_entry*)lru_entry_value(existing))->file;
      *pin_out = existing;
      platform_mutex_unlock(fs->cache_lock);
      fs_cache_destroy(entry, NULL);
      return f;
    }
    struct lru_entry* inserted = lru_put(fs->fd_cache, hash, key, entry);
    if (!inserted) {
      // lru_put already destroyed entry via fs_cache_destroy on failure.
      platform_mutex_unlock(fs->cache_lock);
      return NULL;
    }
    lru_entry_acquire_locked(inserted);
    platform_file* f =
      ((struct fs_cache_entry*)lru_entry_value(inserted))->file;
    *pin_out = inserted;
    platform_mutex_unlock(fs->cache_lock);
    return f;
  }
}

void
store_fs_release(struct store_fs* fs, struct lru_entry* pin)
{
  if (!fs || !pin)
    return;
  lru_entry_release(pin);
}

static void
atomic_max_u64(_Atomic uint64_t* dst, uint64_t val)
{
  uint64_t cur = atomic_load_explicit(dst, memory_order_relaxed);
  while (cur < val &&
         !atomic_compare_exchange_weak_explicit(
           dst, &cur, val, memory_order_relaxed, memory_order_relaxed)) {
  }
}

void
store_fs_stats_get(struct store_fs* fs, struct lru_stats* out)
{
  if (!out)
    return;
  platform_mutex_lock(fs->cache_lock);
  lru_stats_get(fs ? fs->fd_cache : NULL, out);
  platform_mutex_unlock(fs->cache_lock);
}

void
store_fs_io_stats_get(struct store_fs* fs, struct store_fs_io_stats* out)
{
  if (!out)
    return;
  if (!fs) {
    *out = (struct store_fs_io_stats){ 0 };
    return;
  }
  *out = (struct store_fs_io_stats){
    .read_jobs = atomic_load_explicit(&fs->read_jobs, memory_order_relaxed),
    .read_active = atomic_load_explicit(&fs->read_active, memory_order_relaxed),
    .read_max_active =
      atomic_load_explicit(&fs->read_max_active, memory_order_relaxed),
  };
}

void
store_fs_io_stats_reset(struct store_fs* fs)
{
  if (!fs)
    return;
  uint64_t active =
    atomic_load_explicit(&fs->read_active, memory_order_relaxed);
  atomic_store_explicit(&fs->read_jobs, 0, memory_order_relaxed);
  atomic_store_explicit(&fs->read_max_active, active, memory_order_relaxed);
}

struct fs_read_job
{
  struct store_fs* fs;
  struct fs_submit_state* state;
  const char* key; // borrowed: planner-interned, outlives the IO
  struct lru_entry* pin;
  void* dst;
  uint64_t offset;
  size_t len;
};

static void
fs_read_job_fn(void* vctx)
{
  struct fs_read_job* j = (struct fs_read_job*)vctx;
  platform_file* f =
    (platform_file*)((struct fs_cache_entry*)lru_entry_value(j->pin))->file;
  atomic_fetch_add_explicit(&j->fs->read_jobs, 1, memory_order_relaxed);
  uint64_t active =
    atomic_fetch_add_explicit(&j->fs->read_active, 1, memory_order_acq_rel) + 1;
  atomic_max_u64(&j->fs->read_max_active, active);
  int64_t n = platform_file_pread(f, j->dst, j->len, j->offset);
  atomic_fetch_sub_explicit(&j->fs->read_active, 1, memory_order_acq_rel);
  if (n < 0) {
    log_warn("store_fs: read failed for key=%s off=%llu len=%zu got=%lld",
             j->key ? j->key : "(null)",
             (unsigned long long)j->offset,
             j->len,
             (long long)n);
    fs_submit_state_fail(j->state);
  }
}

static void
fs_read_job_free(void* vctx)
{
  struct fs_read_job* j = (struct fs_read_job*)vctx;
  if (!j)
    return;
  struct store_fs* fs = j->fs;
  store_fs_release(fs, j->pin);
  fs_submit_state_drop(j->state);
  pool_free(fs->job_pool, j);
}

static struct store_submit_result
fs_submit(struct store* s, const struct store_read* reads, size_t n)
{
  struct store_fs* fs = (struct store_fs*)s;
  struct store_submit_result result = { .status = DAMACY_IO };
  struct fs_submit_state* state = NULL;
  if (n == 0) {
    struct io_event ioev = io_queue_record(fs->q);
    result.status = DAMACY_OK;
    result.event.seq = ioev.seq;
    return result;
  }
  state = fs_submit_state_new();
  if (!state)
    return result;

  for (size_t i = 0; i < n; ++i) {
    struct lru_entry* pin = NULL;
    platform_file* f = store_fs_acquire(fs, reads[i].key, &pin);
    if (!f) {
      log_warn("store_fs: acquire failed for key=%s; draining batch",
               reads[i].key ? reads[i].key : "(null)");
      goto Drain;
    }
    struct fs_read_job* j = (struct fs_read_job*)pool_alloc(fs->job_pool);
    if (!j) {
      log_warn("store_fs: job_pool exhausted (cap=%zu); draining batch",
               pool_capacity(fs->job_pool));
      store_fs_release(fs, pin);
      goto Drain;
    }
    j->fs = fs;
    j->key = reads[i].key;
    j->state = state;
    j->pin = pin;
    j->dst = reads[i].dst;
    j->offset = reads[i].offset;
    j->len = reads[i].len;
    fs_submit_state_ref(state);
    if (io_queue_post(fs->q, fs_read_job_fn, j, fs_read_job_free)) {
      fs_read_job_free(j);
      goto Drain;
    }
  }
  {
    struct io_event ioev = io_queue_record(fs->q);
    result.status = DAMACY_OK;
    result.event.seq = ioev.seq;
    result.event.impl = state;
  }
  return result;

Drain:
  // Drain in-flight jobs so they don't write into caller buffers after return.
  io_event_wait(fs->q, io_queue_record(fs->q));
  fs_submit_state_drop(state);
  return result;
}

static enum damacy_status
fs_event_wait(struct store* s, struct store_event ev)
{
  struct store_fs* fs = (struct store_fs*)s;
  io_event_wait(fs->q, (struct io_event){ .seq = ev.seq });
  return fs_submit_state_finish(ev);
}

static struct store_event_poll
fs_event_query(struct store* s, struct store_event ev)
{
  struct store_fs* fs = (struct store_fs*)s;
  if (!io_event_query(fs->q, (struct io_event){ .seq = ev.seq }))
    return (struct store_event_poll){ .status = DAMACY_OK };
  return (struct store_event_poll){ .status = fs_submit_state_finish(ev),
                                    .ready = 1 };
}

static void
fs_event_discard(struct store* s, struct store_event ev)
{
  (void)s;
  fs_submit_state_drop((struct fs_submit_state*)ev.impl);
}

static enum store_stat_result
fs_stat(struct store* s, const char* key, uint64_t* out)
{
  struct store_fs* fs = (struct store_fs*)s;
  struct strbuf path = { 0 };
  enum store_stat_result rc = STORE_STAT_ERROR;
  CHECK_SILENT(Out, strbuf_join_path(&path, fs->root, key) == 0);
  switch (platform_path_size(strbuf_cstr(&path), out)) {
    case PLATFORM_STAT_OK:
      rc = STORE_STAT_OK;
      break;
    case PLATFORM_STAT_NOT_FOUND:
      rc = STORE_STAT_NOT_FOUND;
      break;
    case PLATFORM_STAT_ERROR:
      rc = STORE_STAT_ERROR;
      break;
  }
Out:
  strbuf_free(&path);
  return rc;
}

static int
fs_map(struct store* s, const char* key, struct store_view* out)
{
  struct store_fs* fs = (struct store_fs*)s;
  struct strbuf path = { 0 };
  struct platform_file_view* pv = NULL;
  int rc = 1;

  CHECK_SILENT(Out, strbuf_join_path(&path, fs->root, key) == 0);
  pv = (struct platform_file_view*)calloc(1, sizeof(*pv));
  CHECK_SILENT(Out, pv);
  rc = platform_file_map_path(strbuf_cstr(&path), pv);
  if (rc) {
    free(pv);
    pv = NULL;
    goto Out;
  }
  out->data = pv->data;
  out->len = pv->len;
  out->backend = pv;

Out:
  strbuf_free(&path);
  return rc;
}

static void
fs_unmap(struct store* s, struct store_view* view)
{
  (void)s;
  if (!view || !view->backend)
    return;
  struct platform_file_view* pv = (struct platform_file_view*)view->backend;
  platform_file_unmap(pv);
  free(pv);
  view->data = NULL;
  view->len = 0;
  view->backend = NULL;
}

static void
fs_destroy(struct store* s)
{
  struct store_fs* fs = (struct store_fs*)s;
  if (!fs)
    return;
  // io_event_wait drains the ring so every job's ctx_free (which calls
  // pool_free and releases pins) has run before we tear down the pool
  // and lru. io_queue_destroy does not drain queued jobs itself.
  if (fs->q) {
    io_event_wait(fs->q, io_queue_record(fs->q));
    io_queue_destroy(fs->q);
  }
  pool_destroy(fs->job_pool);
  lru_destroy(fs->fd_cache);
  platform_mutex_free(fs->cache_lock);
  free(fs->root);
  free(fs);
}

static void
store_fs_free_partial(struct store_fs* fs)
{
  if (!fs)
    return;
  // No jobs posted yet; in_use is 0, so pool_destroy's assert holds.
  io_queue_destroy(fs->q);
  pool_destroy(fs->job_pool);
  lru_destroy(fs->fd_cache);
  platform_mutex_free(fs->cache_lock);
  free(fs->root);
  free(fs);
}

static const struct store_vtable fs_vtable_host = {
  .destroy = fs_destroy,
  .stat = fs_stat,
  .submit = fs_submit,
  .submit_dev = NULL,
  .event_wait = fs_event_wait,
  .event_query = fs_event_query,
  .event_discard = fs_event_discard,
  .map = fs_map,
  .unmap = fs_unmap,
};

struct store*
store_fs_create(const struct store_fs_config* cfg)
{
  struct store_fs* fs = NULL;

  CHECK_SILENT(Fail, cfg);
  CHECK_SILENT(Fail, cfg->root);

  fs = (struct store_fs*)calloc(1, sizeof(*fs));
  CHECK_SILENT(Fail, fs);
  fs->base.vt = &fs_vtable_host;
  fs->cache_lock = platform_mutex_new();
  CHECK_SILENT(Fail, fs->cache_lock);
  fs->root = strdup(cfg->root);
  CHECK_SILENT(Fail, fs->root);

  uint32_t capacity = cfg->fd_cache_capacity
                        ? cfg->fd_cache_capacity
                        : store_default_fd_cache_capacity();
  struct lru_ops ops = {
    .eq = fs_cache_eq,
    .destroy = fs_cache_destroy,
  };
  fs->fd_cache = lru_create(capacity, FS_FD_CACHE_MAX_PROBE, &ops);
  CHECK_SILENT(Fail, fs->fd_cache);

  fs->q = io_queue_create(cfg->nthreads, cfg->affinity);
  CHECK_SILENT(Fail, fs->q);

  // Sized to the io_queue ring's initial cap. The ring grows past this
  // only under heavy burst; in that regime fs_submit fails (seq==0) and
  // the caller drains, rather than spilling to a second allocator.
  fs->job_pool =
    pool_create(sizeof(struct fs_read_job), DAMACY_IO_QUEUE_INITIAL_CAP);
  CHECK_SILENT(Fail, fs->job_pool);

  return &fs->base;

Fail:
  store_fs_free_partial(fs);
  return NULL;
}
