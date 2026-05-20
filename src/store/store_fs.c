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

#include <stdlib.h>
#include <string.h>

#define FS_FD_CACHE_MAX_PROBE 16u

struct fs_cache_entry
{
  char* key;
  platform_file* file;
};

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

  platform_mutex_lock(fs->cache_mu);
  {
    struct lru_entry* hit = lru_get(fs->fd_cache, hash, key);
    if (hit) {
      lru_entry_acquire(hit);
      platform_file* f = ((struct fs_cache_entry*)lru_entry_value(hit))->file;
      *pin_out = hit;
      platform_mutex_unlock(fs->cache_mu);
      return f;
    }
  }
  platform_mutex_unlock(fs->cache_mu);

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

  platform_mutex_lock(fs->cache_mu);
  {
    struct lru_entry* existing = lru_peek(fs->fd_cache, hash, key);
    if (existing) {
      lru_entry_acquire(existing);
      platform_file* f =
        ((struct fs_cache_entry*)lru_entry_value(existing))->file;
      *pin_out = existing;
      platform_mutex_unlock(fs->cache_mu);
      fs_cache_destroy(entry, NULL);
      return f;
    }
    struct lru_entry* inserted = lru_put(fs->fd_cache, hash, key, entry);
    if (!inserted) {
      // lru_put already destroyed entry via fs_cache_destroy on failure.
      platform_mutex_unlock(fs->cache_mu);
      return NULL;
    }
    lru_entry_acquire(inserted);
    platform_file* f =
      ((struct fs_cache_entry*)lru_entry_value(inserted))->file;
    *pin_out = inserted;
    platform_mutex_unlock(fs->cache_mu);
    return f;
  }
}

void
store_fs_release(struct store_fs* fs, struct lru_entry* pin)
{
  if (!fs || !pin)
    return;
  platform_mutex_lock(fs->cache_mu);
  lru_entry_release(pin);
  platform_mutex_unlock(fs->cache_mu);
}

void
store_fs_stats_get(struct store_fs* fs, struct lru_stats* out)
{
  if (!out)
    return;
  platform_mutex_lock(fs->cache_mu);
  lru_stats_get(fs ? fs->fd_cache : NULL, out);
  platform_mutex_unlock(fs->cache_mu);
}

struct fs_read_job
{
  struct store_fs* fs;
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
  (void)platform_file_pread(f, j->dst, j->len, j->offset);
}

static void
fs_read_job_free(void* vctx)
{
  struct fs_read_job* j = (struct fs_read_job*)vctx;
  if (!j)
    return;
  struct store_fs* fs = j->fs;
  store_fs_release(fs, j->pin);
  pool_free(fs->job_pool, j);
}

static struct store_event
fs_submit(struct store* s, const struct store_read* reads, size_t n)
{
  struct store_fs* fs = (struct store_fs*)s;
  struct store_event ev = { 0 };
  if (n == 0) {
    struct io_event ioev = io_queue_record(fs->q);
    ev.seq = ioev.seq;
    return ev;
  }

  // seq == 0 → partial-submit failure. Without it, a half-submitted
  // batch would silently leave dst buffers unfilled. On failure we
  // still drain in-flight jobs so they don't write into caller buffers
  // after we return.
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
    j->pin = pin;
    j->dst = reads[i].dst;
    j->offset = reads[i].offset;
    j->len = reads[i].len;
    if (io_queue_post(fs->q, fs_read_job_fn, j, fs_read_job_free)) {
      fs_read_job_free(j);
      goto Drain;
    }
  }
  {
    struct io_event ioev = io_queue_record(fs->q);
    ev.seq = ioev.seq;
  }
  return ev;

Drain:
  io_event_wait(fs->q, io_queue_record(fs->q));
  return ev; // ev.seq == 0 signals failure
}

static void
fs_event_wait(struct store* s, struct store_event ev)
{
  struct store_fs* fs = (struct store_fs*)s;
  io_event_wait(fs->q, (struct io_event){ .seq = ev.seq });
}

static int
fs_event_query(struct store* s, struct store_event ev)
{
  struct store_fs* fs = (struct store_fs*)s;
  return io_event_query(fs->q, (struct io_event){ .seq = ev.seq });
}

static int
fs_stat(struct store* s, const char* key, uint64_t* out)
{
  struct store_fs* fs = (struct store_fs*)s;
  struct strbuf path = { 0 };
  int rc = 1;
  CHECK_SILENT(Out, strbuf_join_path(&path, fs->root, key) == 0);
  rc = platform_path_size(strbuf_cstr(&path), out);
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
  platform_mutex_free(fs->cache_mu);
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
  platform_mutex_free(fs->cache_mu);
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
  fs->cache_mu = platform_mutex_new();
  CHECK_SILENT(Fail, fs->cache_mu);
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
