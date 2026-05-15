#include "store/store_fs.h"

#include "log/log.h"
#include "store/store.h"
#include "util/prelude.h"
#include "util/strbuf.h"

#ifdef DAMACY_ENABLE_GDS
#include "store/store_fs_gds.h"
#endif

#include <stdlib.h>
#include <string.h>

// Locate or open a file for the given key. Returns NULL on failure.
// Caller does not own the returned handle; the store owns it for its
// lifetime.
static platform_file*
fs_get_file(struct store_fs* fs, const char* key)
{
  pthread_mutex_lock(&fs->cache_mu);
  for (size_t i = 0; i < fs->n_slots; ++i) {
    if (strcmp(fs->slots[i].key, key) == 0) {
      platform_file* f = fs->slots[i].file;
      pthread_mutex_unlock(&fs->cache_mu);
      return f;
    }
  }
  pthread_mutex_unlock(&fs->cache_mu);

  // Compose full path and open outside the lock.
  struct strbuf path = { 0 };
  platform_file* f = NULL;
  CHECK_SILENT(Out, strbuf_join_path(&path, fs->root, key) == 0);
  f = platform_file_open_read(strbuf_cstr(&path), 0);
Out:
  strbuf_free(&path);
  if (!f)
    return NULL;

  // Insert into cache; if a concurrent caller raced us, drop ours and reuse
  // theirs.
  pthread_mutex_lock(&fs->cache_mu);
  for (size_t i = 0; i < fs->n_slots; ++i) {
    if (strcmp(fs->slots[i].key, key) == 0) {
      platform_file* winner = fs->slots[i].file;
      pthread_mutex_unlock(&fs->cache_mu);
      platform_file_close(f);
      return winner;
    }
  }
  if (fs->n_slots == fs->cap_slots) {
    size_t new_cap = fs->cap_slots ? fs->cap_slots * 2 : 16;
    struct fs_cache_slot* p = (struct fs_cache_slot*)realloc(
      fs->slots, new_cap * sizeof(struct fs_cache_slot));
    if (!p) {
      pthread_mutex_unlock(&fs->cache_mu);
      platform_file_close(f);
      return NULL;
    }
    fs->slots = p;
    fs->cap_slots = new_cap;
  }
  char* dup = strdup(key);
  if (!dup) {
    pthread_mutex_unlock(&fs->cache_mu);
    platform_file_close(f);
    return NULL;
  }
  fs->slots[fs->n_slots].key = dup;
  fs->slots[fs->n_slots].file = f;
  fs->n_slots++;
  pthread_mutex_unlock(&fs->cache_mu);
  return f;
}

// io_queue job context.
struct fs_read_job
{
  struct store_fs* fs;
  char* key; // heap-owned
  void* dst;
  uint64_t offset;
  size_t len;
};

static void
fs_read_job_fn(void* vctx)
{
  struct fs_read_job* j = (struct fs_read_job*)vctx;
  platform_file* f = fs_get_file(j->fs, j->key);
  if (!f)
    return; // silently fail; store has no error channel yet
  (void)platform_file_pread(f, j->dst, j->len, j->offset);
}

static void
fs_read_job_free(void* vctx)
{
  struct fs_read_job* j = (struct fs_read_job*)vctx;
  free(j->key);
  free(j);
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

  // Sentinel: seq == 0 means partial-submit failure. Callers (store_read_many)
  // detect this and surface an error; without it, a half-submitted batch
  // would silently leave dst buffers unfilled. On failure we still drain
  // jobs already in flight so they don't write into caller buffers after
  // we return.
  for (size_t i = 0; i < n; ++i) {
    struct fs_read_job* j =
      (struct fs_read_job*)calloc(1, sizeof(struct fs_read_job));
    CHECK_SILENT(Drain, j);
    j->fs = fs;
    j->key = strdup(reads[i].key);
    j->dst = reads[i].dst;
    j->offset = reads[i].offset;
    j->len = reads[i].len;
    if (!j->key) {
      free(j);
      goto Drain;
    }
    if (io_queue_post(fs->q, fs_read_job_fn, j, fs_read_job_free))
      goto Drain;
  }
  struct io_event ioev = io_queue_record(fs->q);
  ev.seq = ioev.seq;
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
  // Drain pending I/O before tearing down file handles.
  if (fs->q) {
    io_event_wait(fs->q, io_queue_record(fs->q));
    io_queue_destroy(fs->q);
  }
#ifdef DAMACY_ENABLE_GDS
  // Deregister cuFile handles + close the driver before we close the
  // underlying fds — order matters: cuFile handles reference the fd.
  store_fs_gds_destroy(fs);
#endif
  for (size_t i = 0; i < fs->n_slots; ++i) {
    free(fs->slots[i].key);
    platform_file_close(fs->slots[i].file);
  }
  free(fs->slots);
  pthread_mutex_destroy(&fs->cache_mu);
  free(fs->root);
  free(fs);
}

// Helper: free a partially-constructed store_fs from store_fs_create's
// Fail label. fs->cache_mu is initialized inline before any branch that
// could send us here, so we can always destroy it.
static void
store_fs_free_partial(struct store_fs* fs)
{
  if (!fs)
    return;
  io_queue_destroy(fs->q);
  pthread_mutex_destroy(&fs->cache_mu);
  free(fs->root);
  free(fs);
}

static struct store_vtable fs_vtable = {
  .destroy = fs_destroy,
  .stat = fs_stat,
  .submit = fs_submit,
  .submit_dev = NULL, // patched by store_fs_gds_init when GDS enabled
  .event_wait = fs_event_wait,
  .event_query = fs_event_query,
  .map = fs_map,
  .unmap = fs_unmap,
};

// Mutable-vtable accessor used by store_fs_gds_init to install
// submit_dev once cuFile driver init succeeds. Single shared vtable;
// patching is idempotent (writes the same pointer if called twice).
struct store_vtable*
store_fs_vtable_mut(void)
{
  return &fs_vtable;
}

// Bridge from store_fs_gds.c — it needs to ensure the file is open
// (cache slot exists) before registering its FD with cuFile.
platform_file*
store_fs_get_file_external(struct store_fs* fs, const char* key)
{
  return fs_get_file(fs, key);
}

struct store*
store_fs_create(const struct store_fs_config* cfg)
{
  struct store_fs* fs = NULL;

  CHECK_SILENT(Fail, cfg);
  CHECK_SILENT(Fail, cfg->root);

  fs = (struct store_fs*)calloc(1, sizeof(*fs));
  CHECK_SILENT(Fail, fs);
  fs->base.vt = &fs_vtable;
  // Init the mutex up front so store_fs_free_partial can always
  // unconditionally destroy it.
  pthread_mutex_init(&fs->cache_mu, NULL);
  fs->root = strdup(cfg->root);
  CHECK_SILENT(Fail, fs->root);
  fs->q = io_queue_create(cfg->nthreads, cfg->affinity);
  CHECK_SILENT(Fail, fs->q);
#ifdef DAMACY_ENABLE_GDS
  // Best-effort: if cuFile init fails the store still works through
  // the host-staging submit; callers introspect store_supports_gds.
  if (store_fs_gds_init(fs) == 0)
    log_info("damacy: cuFile driver opened; GDS submit_dev available");
#endif
  return &fs->base;

Fail:
  store_fs_free_partial(fs);
  return NULL;
}
