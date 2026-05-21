#include "store/store_fs_gds.h"

#include "log/log.h"
#include "platform/platform.h"
#include "platform/platform_io.h"
#include "store/store.h"
#include "store/store_internal.h"
#include "util/hash.h"
#include "util/lru.h"
#include "util/prelude.h"
#include "util/strbuf.h"

#include <cufile.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef CUfileError_t (*pfn_cuFileDriverOpen)(void);
typedef CUfileError_t (*pfn_cuFileDriverClose)(void);
typedef CUfileError_t (*pfn_cuFileHandleRegister)(CUfileHandle_t*,
                                                  CUfileDescr_t*);
typedef void (*pfn_cuFileHandleDeregister)(CUfileHandle_t);
typedef CUfileError_t (*pfn_cuFileReadAsync)(CUfileHandle_t,
                                             void*,
                                             size_t*,
                                             off_t*,
                                             off_t*,
                                             ssize_t*,
                                             CUstream);

// platform_call_once establishes a happens-before edge so unsynchronized
// reads of g_libcufile.* are safe afterward. dlopen handle leaked at
// process exit.
static struct
{
  pfn_cuFileDriverOpen cuFileDriverOpen;
  pfn_cuFileDriverClose cuFileDriverClose;
  pfn_cuFileHandleRegister cuFileHandleRegister;
  pfn_cuFileHandleDeregister cuFileHandleDeregister;
  pfn_cuFileReadAsync cuFileReadAsync;
} g_libcufile;

#define DLSYM_BIND(handle, table, sym)                                         \
  (*(void**)&(table).sym = platform_dlsym((handle), #sym))

static platform_once g_libcufile_once = PLATFORM_ONCE_INIT;
static int g_libcufile_ok;

static void
libcufile_once_init(void)
{
  void* h = platform_dlopen("libcufile.so.0");
  if (!h) {
    log_error(
      "cuFile: libcufile.so.0 not loadable — install the cuFile runtime "
      "(part of CUDA toolkit / nvidia-fs) and ensure it is on the dynamic "
      "loader path; GDS unavailable");
    return;
  }
  DLSYM_BIND(h, g_libcufile, cuFileDriverOpen);
  DLSYM_BIND(h, g_libcufile, cuFileDriverClose);
  DLSYM_BIND(h, g_libcufile, cuFileHandleRegister);
  DLSYM_BIND(h, g_libcufile, cuFileHandleDeregister);
  DLSYM_BIND(h, g_libcufile, cuFileReadAsync);
  if (!g_libcufile.cuFileDriverOpen || !g_libcufile.cuFileDriverClose ||
      !g_libcufile.cuFileHandleRegister ||
      !g_libcufile.cuFileHandleDeregister || !g_libcufile.cuFileReadAsync) {
    log_error("cuFile: missing required symbol in libcufile.so.0; GDS "
              "unavailable");
    platform_dlclose(h);
    memset(&g_libcufile, 0, sizeof(g_libcufile));
    return;
  }
  g_libcufile_ok = 1;
}

static int
try_load_libcufile(void)
{
  platform_call_once(&g_libcufile_once, libcufile_once_init);
  return g_libcufile_ok;
}

// io_queue seqs are monotonic from 1; UINT64_MAX never collides.
#define GDS_SENTINEL_SEQ ((uint64_t)~(uint64_t)0)

struct fs_gds_cache_entry
{
  char* key;
  platform_file* file;
  CUfileHandle_t handle;
};

struct store_fs_gds
{
  struct store base;
  char* root;       // owned
  void* gds_stream; // CUstream as void*
  uint8_t driver_opened;
  struct platform_mutex* cache_lock;
  struct lru* cache;
};

static int
fs_gds_entry_eq(const void* value, const void* probe_key, void* user)
{
  (void)user;
  const struct fs_gds_cache_entry* e = (const struct fs_gds_cache_entry*)value;
  return strcmp(e->key, (const char*)probe_key) == 0;
}

// cuFile is undefined if the fd closes while a handle is still
// registered against it — deregister must precede close.
static void
fs_gds_entry_destroy(void* value, void* user)
{
  (void)user;
  struct fs_gds_cache_entry* e = (struct fs_gds_cache_entry*)value;
  if (!e)
    return;
  if (e->handle && g_libcufile.cuFileHandleDeregister)
    g_libcufile.cuFileHandleDeregister(e->handle);
  platform_file_close(e->file);
  free(e->key);
  free(e);
}

static struct lru_entry*
fs_gds_acquire(struct store_fs_gds* g, const char* key)
{
  uint64_t hash = hash_fnv1a_str(key);

  platform_mutex_lock(g->cache_lock);
  struct lru_entry* hit = lru_get(g->cache, hash, key);
  if (hit) {
    lru_entry_acquire_locked(hit);
    platform_mutex_unlock(g->cache_lock);
    return hit;
  }
  platform_mutex_unlock(g->cache_lock);

  struct strbuf path = { 0 };
  platform_file* f = NULL;
  CUfileHandle_t h = NULL;
  struct fs_gds_cache_entry* entry = NULL;
  char* dup = NULL;

  CHECK_SILENT(Fail, strbuf_join_path(&path, g->root, key) == 0);
  f = platform_file_open_read(strbuf_cstr(&path), 0);
  CHECK_SILENT(Fail, f);

  {
    int fd = platform_file_fd(f);
    CHECK_SILENT(Fail, fd >= 0);
    CUfileDescr_t descr;
    memset(&descr, 0, sizeof(descr));
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descr.handle.fd = fd;
    CUfileError_t e = g_libcufile.cuFileHandleRegister(&h, &descr);
    if (e.err != CU_FILE_SUCCESS) {
      log_error("cuFileHandleRegister(%s) failed: err=%d", key, (int)e.err);
      h = NULL;
      goto Fail;
    }
  }

  entry = (struct fs_gds_cache_entry*)calloc(1, sizeof(*entry));
  CHECK_SILENT(Fail, entry);
  dup = strdup(key);
  CHECK_SILENT(Fail, dup);
  entry->key = dup;
  entry->file = f;
  entry->handle = h;

  platform_mutex_lock(g->cache_lock);
  {
    struct lru_entry* existing = lru_peek(g->cache, hash, key);
    if (existing) {
      lru_entry_acquire_locked(existing);
      platform_mutex_unlock(g->cache_lock);
      fs_gds_entry_destroy(entry, NULL);
      strbuf_free(&path);
      return existing;
    }
    struct lru_entry* inserted = lru_put(g->cache, hash, key, entry);
    if (!inserted) {
      // lru_put already invoked fs_gds_entry_destroy on `entry`.
      platform_mutex_unlock(g->cache_lock);
      strbuf_free(&path);
      return NULL;
    }
    lru_entry_acquire_locked(inserted);
    platform_mutex_unlock(g->cache_lock);
    strbuf_free(&path);
    return inserted;
  }

Fail:
  if (entry) {
    free(entry);
    entry = NULL;
  }
  free(dup);
  if (h && g_libcufile.cuFileHandleDeregister)
    g_libcufile.cuFileHandleDeregister(h);
  if (f)
    platform_file_close(f);
  strbuf_free(&path);
  return NULL;
}

// cuFileReadAsync stashes pointers to these fields and dereferences
// them when the stream reaches the read, so the array must outlive
// submit_dev's return. Freed via cuLaunchHostFunc on the same stream.
struct fs_gds_async_params
{
  size_t size;
  off_t file_off;
  off_t buf_off;
  ssize_t bytes_read;
  struct lru_entry* pin;
};

// rc=2 protocol: submit-owner and host-func callback each own one ref;
// last drop frees. `claimed` latches the owner-side drop into a single
// CAS so repeated event_query calls don't double-free and an unqueried
// event is still reclaimed by the callback alone.
struct fs_gds_done
{
  _Atomic uint8_t flag;
  _Atomic uint8_t claimed;
  _Atomic int rc;
};

static void
fs_gds_done_drop(struct fs_gds_done* d)
{
  if (!d)
    return;
  if (atomic_fetch_sub_explicit(&d->rc, 1, memory_order_acq_rel) == 1)
    free(d);
}

struct fs_gds_async_ctx
{
  struct store_fs_gds* g;
  size_t n;
  struct fs_gds_async_params* params;
  struct fs_gds_done* done;
};

static void CUDA_CB
fs_gds_free_params_cb(void* userdata)
{
  struct fs_gds_async_ctx* ctx = (struct fs_gds_async_ctx*)userdata;
  if (!ctx)
    return;
  if (ctx->params) {
    for (size_t i = 0; i < ctx->n; ++i) {
      if (ctx->params[i].pin)
        lru_entry_release(ctx->params[i].pin);
    }
    free(ctx->params);
  }
  if (ctx->done) {
    atomic_store_explicit(&ctx->done->flag, 1, memory_order_release);
    fs_gds_done_drop(ctx->done);
  }
  free(ctx);
}

static struct store_event
gds_submit_dev(struct store* s, const struct store_read* reads, size_t n)
{
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  // seq == 0 is the failure sentinel consumed by wave_pool (DAMACY_IO).
  struct store_event ev = { 0 };

  CUstream stream = (CUstream)g->gds_stream;
  if (!stream) {
    log_error("store_fs_gds: submit_dev called before stream was set");
    return ev;
  }

  if (n == 0) {
    ev.seq = GDS_SENTINEL_SEQ;
    return ev;
  }

  struct fs_gds_async_ctx* ctx =
    (struct fs_gds_async_ctx*)calloc(1, sizeof(*ctx));
  if (!ctx)
    return ev;
  ctx->g = g;
  ctx->n = n;
  // calloc: SubmitFail's drain loop relies on params[i].pin == NULL for
  // unacquired entries; switching to malloc would silently break it.
  ctx->params = (struct fs_gds_async_params*)calloc(n, sizeof(*ctx->params));
  if (!ctx->params) {
    free(ctx);
    return ev;
  }
  ctx->done = (struct fs_gds_done*)calloc(1, sizeof(*ctx->done));
  if (!ctx->done) {
    free(ctx->params);
    free(ctx);
    return ev;
  }
  atomic_init(&ctx->done->rc, 2);

  size_t submitted = 0;
  for (size_t i = 0; i < n; ++i) {
    struct lru_entry* pin = fs_gds_acquire(g, reads[i].key);
    if (!pin) {
      // Reads 0..i-1 are already queued on the stream; tell SubmitFail
      // to drain before freeing params (cuFile holds raw pointers).
      submitted = i;
      goto SubmitFail;
    }
    ctx->params[i].pin = pin;
    ctx->params[i].size = reads[i].len;
    ctx->params[i].file_off = (off_t)reads[i].offset;
    ctx->params[i].buf_off = 0;
    ctx->params[i].bytes_read = 0;
    const struct fs_gds_cache_entry* entry =
      (const struct fs_gds_cache_entry*)lru_entry_value(pin);
    CUfileError_t e = g_libcufile.cuFileReadAsync(entry->handle,
                                                  reads[i].dst,
                                                  &ctx->params[i].size,
                                                  &ctx->params[i].file_off,
                                                  &ctx->params[i].buf_off,
                                                  &ctx->params[i].bytes_read,
                                                  stream);
    if (e.err != CU_FILE_SUCCESS) {
      log_error("cuFileReadAsync(%s,off=%llu,len=%zu) failed: err=%d",
                reads[i].key,
                (unsigned long long)reads[i].offset,
                reads[i].len,
                (int)e.err);
      submitted = i;
      goto SubmitFail;
    }
  }

  if (cuLaunchHostFunc(stream, fs_gds_free_params_cb, ctx) != CUDA_SUCCESS) {
    log_error("cuLaunchHostFunc(free params) failed");
    submitted = n;
    goto SubmitFail;
  }

  ev.seq = GDS_SENTINEL_SEQ;
  ev.impl = ctx->done;
  return ev;

SubmitFail:
  // Drain so cuFile is done dereferencing params before we free them.
  // The host-func callback never ran, so pin-release is ours.
  if (submitted > 0)
    cuStreamSynchronize(stream);
  for (size_t i = 0; i < n; ++i) {
    if (ctx->params[i].pin)
      lru_entry_release(ctx->params[i].pin);
  }
  free(ctx->params);
  fs_gds_done_drop(ctx->done);
  fs_gds_done_drop(ctx->done);
  free(ctx);
  return ev;
}

static void
fs_gds_try_claim(struct fs_gds_done* d)
{
  if (!d)
    return;
  uint8_t expected = 0;
  if (atomic_compare_exchange_strong_explicit(
        &d->claimed, &expected, 1, memory_order_acq_rel, memory_order_relaxed))
    fs_gds_done_drop(d);
}

// Spin on the per-batch flag; cuStreamSynchronize would drain unrelated
// concurrent submits on the same stream and kill IO/compute overlap.
static void
gds_event_wait(struct store* s, struct store_event ev)
{
  (void)s;
  if (ev.seq != GDS_SENTINEL_SEQ)
    return;
  struct fs_gds_done* d = (struct fs_gds_done*)ev.impl;
  if (!d)
    return;
  while (!atomic_load_explicit(&d->flag, memory_order_acquire))
    platform_cpu_pause();
  fs_gds_try_claim(d);
}

static int
gds_event_query(struct store* s, struct store_event ev)
{
  (void)s;
  if (ev.seq != GDS_SENTINEL_SEQ)
    return 0;
  struct fs_gds_done* d = (struct fs_gds_done*)ev.impl;
  if (!d)
    return 1; // n==0 fast-path: nothing to wait on
  if (!atomic_load_explicit(&d->flag, memory_order_acquire))
    return 0;
  fs_gds_try_claim(d);
  return 1;
}

static void
gds_event_discard(struct store* s, struct store_event ev)
{
  (void)s;
  if (ev.seq != GDS_SENTINEL_SEQ)
    return;
  fs_gds_try_claim((struct fs_gds_done*)ev.impl);
}

// Also reached from store_fs_gds_create's Fail label on partial
// construction; every member is calloc-zeroed and every freed callee is
// NULL-safe.
static void
gds_destroy(struct store* s)
{
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  if (!g)
    return;
  // Drain in-flight host-funcs so a late pin-release can't land on a
  // freed mutex.
  if (g->gds_stream)
    cuStreamSynchronize((CUstream)g->gds_stream);
  lru_destroy(g->cache);
  platform_mutex_free(g->cache_lock);
  if (g->driver_opened && g_libcufile.cuFileDriverClose)
    g_libcufile.cuFileDriverClose();
  free(g->root);
  free(g);
}

static const struct store_vtable gds_vtable = {
  .destroy = gds_destroy,
  .submit_dev = gds_submit_dev,
  .event_wait = gds_event_wait,
  .event_query = gds_event_query,
  .event_discard = gds_event_discard,
};

struct store*
store_fs_gds_create(const struct store_fs_gds_config* cfg)
{
  struct store_fs_gds* g = NULL;

  if (!cfg || !cfg->root)
    return NULL;
  if (!try_load_libcufile())
    return NULL;

  {
    CUfileError_t e = g_libcufile.cuFileDriverOpen();
    if (e.err != CU_FILE_SUCCESS) {
      log_error("cuFileDriverOpen failed (err=%d); GDS unavailable",
                (int)e.err);
      return NULL;
    }
  }

  g = (struct store_fs_gds*)calloc(1, sizeof(*g));
  CHECK_SILENT(Fail, g);
  g->base.vt = &gds_vtable;
  g->driver_opened = 1;
  g->root = strdup(cfg->root);
  CHECK_SILENT(Fail, g->root);
  g->cache_lock = platform_mutex_new();
  CHECK_SILENT(Fail, g->cache_lock);
  {
    uint32_t cap = cfg->fd_cache_capacity ? cfg->fd_cache_capacity
                                          : store_default_fd_cache_capacity();
    struct lru_ops ops = {
      .eq = fs_gds_entry_eq,
      .destroy = fs_gds_entry_destroy,
    };
    g->cache = lru_create(cap, 16, &ops);
    CHECK_SILENT(Fail, g->cache);
  }
  log_info("damacy: cuFile driver opened; GDS submit_dev available");
  return &g->base;

Fail:
  if (g)
    gds_destroy(&g->base);
  else
    g_libcufile.cuFileDriverClose();
  return NULL;
}

void
store_fs_gds_stats_get(struct store* s, struct lru_stats* out)
{
  if (!out)
    return;
  if (!s) {
    lru_stats_get(NULL, out);
    return;
  }
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  platform_mutex_lock(g->cache_lock);
  lru_stats_get(g->cache, out);
  platform_mutex_unlock(g->cache_lock);
}

void
store_fs_gds_set_stream(struct store* s, void* stream)
{
  if (!s)
    return;
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  g->gds_stream = stream;
}
