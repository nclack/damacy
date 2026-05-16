#include "store/store_fs_gds.h"

#include "log/log.h"
#include "platform/platform_io.h"
#include "store/store.h"
#include "store/store_fs.h"

#include <cufile.h>
#include <dlfcn.h>
#include <pthread.h>
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

// pthread_once happens-before establishes safe unsynchronized reads of
// g_libcufile.* afterward. dlopen handle leaked at process exit.
static struct
{
  pfn_cuFileDriverOpen cuFileDriverOpen;
  pfn_cuFileDriverClose cuFileDriverClose;
  pfn_cuFileHandleRegister cuFileHandleRegister;
  pfn_cuFileHandleDeregister cuFileHandleDeregister;
  pfn_cuFileReadAsync cuFileReadAsync;
} g_libcufile;

#define DLSYM_BIND(handle, table, sym)                                         \
  (*(void**)&(table).sym = dlsym((handle), #sym))

static pthread_once_t g_libcufile_once = PTHREAD_ONCE_INIT;
static int g_libcufile_ok;

static void
libcufile_once_init(void)
{
  void* h = dlopen("libcufile.so.0", RTLD_LAZY | RTLD_LOCAL);
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
    dlclose(h);
    memset(&g_libcufile, 0, sizeof(g_libcufile));
    return;
  }
  g_libcufile_ok = 1;
}

static int
try_load_libcufile(void)
{
  pthread_once(&g_libcufile_once, libcufile_once_init);
  return g_libcufile_ok;
}

struct gds_handle_slot
{
  char* key;
  void* handle; // CUfileHandle_t
};

// io_queue seqs are monotonic from 1; UINT64_MAX never collides.
#define GDS_SENTINEL_SEQ ((uint64_t)~(uint64_t)0)

struct store_fs_gds
{
  struct store base;
  struct store* host; // owns; underlying store_fs for host-path forwards
  void* gds_stream;   // CUstream as void*
  uint8_t driver_opened;
  pthread_mutex_t handle_mu;
  struct gds_handle_slot* handles;
  size_t n_handles, cap_handles;
};

static CUfileHandle_t
gds_get_handle(struct store_fs_gds* g, const char* key)
{
  struct store_fs* host_fs = (struct store_fs*)g->host;
  platform_file* f = store_fs_get_file_external(host_fs, key);
  if (!f)
    return NULL;

  pthread_mutex_lock(&g->handle_mu);
  for (size_t i = 0; i < g->n_handles; ++i) {
    if (strcmp(g->handles[i].key, key) == 0) {
      CUfileHandle_t h = (CUfileHandle_t)g->handles[i].handle;
      pthread_mutex_unlock(&g->handle_mu);
      return h;
    }
  }
  int fd = platform_file_fd(f);
  if (fd < 0) {
    pthread_mutex_unlock(&g->handle_mu);
    return NULL;
  }
  CUfileDescr_t descr;
  memset(&descr, 0, sizeof(descr));
  descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  descr.handle.fd = fd;
  CUfileHandle_t h = NULL;
  CUfileError_t e = g_libcufile.cuFileHandleRegister(&h, &descr);
  if (e.err != CU_FILE_SUCCESS) {
    log_error("cuFileHandleRegister(%s) failed: err=%d", key, (int)e.err);
    pthread_mutex_unlock(&g->handle_mu);
    return NULL;
  }
  if (g->n_handles == g->cap_handles) {
    size_t new_cap = g->cap_handles ? g->cap_handles * 2 : 16;
    struct gds_handle_slot* p = (struct gds_handle_slot*)realloc(
      g->handles, new_cap * sizeof(struct gds_handle_slot));
    if (!p) {
      g_libcufile.cuFileHandleDeregister(h);
      pthread_mutex_unlock(&g->handle_mu);
      return NULL;
    }
    g->handles = p;
    g->cap_handles = new_cap;
  }
  char* dup = strdup(key);
  if (!dup) {
    g_libcufile.cuFileHandleDeregister(h);
    pthread_mutex_unlock(&g->handle_mu);
    return NULL;
  }
  g->handles[g->n_handles].key = dup;
  g->handles[g->n_handles].handle = (void*)h;
  g->n_handles++;
  pthread_mutex_unlock(&g->handle_mu);
  return h;
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
};

static void CUDA_CB
fs_gds_free_params_cb(void* userdata)
{
  free(userdata);
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

  struct fs_gds_async_params* params =
    (struct fs_gds_async_params*)calloc(n, sizeof(*params));
  if (!params)
    return ev;

  for (size_t i = 0; i < n; ++i) {
    CUfileHandle_t h = gds_get_handle(g, reads[i].key);
    if (!h) {
      free(params);
      return ev;
    }
    params[i].size = reads[i].len;
    params[i].file_off = (off_t)reads[i].offset;
    params[i].buf_off = 0;
    params[i].bytes_read = 0;
    CUfileError_t e = g_libcufile.cuFileReadAsync(h,
                                                  reads[i].dst,
                                                  &params[i].size,
                                                  &params[i].file_off,
                                                  &params[i].buf_off,
                                                  &params[i].bytes_read,
                                                  stream);
    if (e.err != CU_FILE_SUCCESS) {
      log_error("cuFileReadAsync(%s,off=%llu,len=%zu) failed: err=%d",
                reads[i].key,
                (unsigned long long)reads[i].offset,
                reads[i].len,
                (int)e.err);
      // Drain already-submitted reads before freeing their params.
      cuStreamSynchronize(stream);
      free(params);
      return ev;
    }
  }

  CUresult cr = cuLaunchHostFunc(stream, fs_gds_free_params_cb, params);
  if (cr != CUDA_SUCCESS) {
    log_error("cuLaunchHostFunc(free params) failed: %d", (int)cr);
    cuStreamSynchronize(stream);
    free(params);
    return ev;
  }

  ev.seq = GDS_SENTINEL_SEQ;
  return ev;
}

static int
gds_stat(struct store* s, const char* key, uint64_t* out)
{
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  return g->host->vt->stat(g->host, key, out);
}

static struct store_event
gds_submit(struct store* s, const struct store_read* reads, size_t n)
{
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  return g->host->vt->submit(g->host, reads, n);
}

static void
gds_event_wait(struct store* s, struct store_event ev)
{
  if (ev.seq == GDS_SENTINEL_SEQ)
    return;
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  g->host->vt->event_wait(g->host, ev);
}

static int
gds_event_query(struct store* s, struct store_event ev)
{
  if (ev.seq == GDS_SENTINEL_SEQ)
    return 1;
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  return g->host->vt->event_query(g->host, ev);
}

static int
gds_map(struct store* s, const char* key, struct store_view* out)
{
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  return g->host->vt->map(g->host, key, out);
}

static void
gds_unmap(struct store* s, struct store_view* view)
{
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  g->host->vt->unmap(g->host, view);
}

static void
gds_destroy(struct store* s)
{
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  if (!g)
    return;
  if (g->gds_stream)
    cuStreamSynchronize((CUstream)g->gds_stream);
  pthread_mutex_lock(&g->handle_mu);
  for (size_t i = 0; i < g->n_handles; ++i) {
    if (g->handles[i].handle && g_libcufile.cuFileHandleDeregister)
      g_libcufile.cuFileHandleDeregister((CUfileHandle_t)g->handles[i].handle);
    free(g->handles[i].key);
  }
  pthread_mutex_unlock(&g->handle_mu);
  free(g->handles);
  pthread_mutex_destroy(&g->handle_mu);
  if (g->driver_opened && g_libcufile.cuFileDriverClose)
    g_libcufile.cuFileDriverClose();
  // Deregister handles before tearing down host (handles reference its FDs).
  if (g->host)
    g->host->vt->destroy(g->host);
  free(g);
}

static const struct store_vtable gds_vtable = {
  .destroy = gds_destroy,
  .stat = gds_stat,
  .submit = gds_submit,
  .submit_dev = gds_submit_dev,
  .event_wait = gds_event_wait,
  .event_query = gds_event_query,
  .map = gds_map,
  .unmap = gds_unmap,
};

struct store*
store_fs_gds_create(const struct store_fs_config* cfg)
{
  if (!try_load_libcufile())
    return NULL;
  CUfileError_t e = g_libcufile.cuFileDriverOpen();
  if (e.err != CU_FILE_SUCCESS) {
    log_error("cuFileDriverOpen failed (err=%d); GDS unavailable", (int)e.err);
    return NULL;
  }

  struct store_fs_gds* g = (struct store_fs_gds*)calloc(1, sizeof(*g));
  if (!g) {
    g_libcufile.cuFileDriverClose();
    return NULL;
  }
  g->base.vt = &gds_vtable;
  g->driver_opened = 1;
  pthread_mutex_init(&g->handle_mu, NULL);

  g->host = store_fs_create(cfg);
  if (!g->host) {
    pthread_mutex_destroy(&g->handle_mu);
    g_libcufile.cuFileDriverClose();
    free(g);
    return NULL;
  }
  log_info("damacy: cuFile driver opened; GDS submit_dev available");
  return &g->base;
}

void
store_fs_gds_set_stream(struct store* s, void* stream)
{
  if (!s)
    return;
  struct store_fs_gds* g = (struct store_fs_gds*)s;
  g->gds_stream = stream;
}
