// libcufile is dlopen'd lazily — see store_fs_gds.h. Including
// cufile.h gives us types/enums only; no symbols are referenced
// directly, so libcufile is not linked.

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

// One-time init via pthread_once. After pthread_once returns, readers
// can use g_libcufile.* unsynchronized: the once_init body happens-before
// every subsequent caller. g_libcufile_ok is the sole readiness flag;
// the dlopen handle is leaked at process exit (mirrors src/numa/numa.c).
static struct
{
  pfn_cuFileDriverOpen cuFileDriverOpen;
  pfn_cuFileDriverClose cuFileDriverClose;
  pfn_cuFileHandleRegister cuFileHandleRegister;
  pfn_cuFileHandleDeregister cuFileHandleDeregister;
  pfn_cuFileReadAsync cuFileReadAsync;
} g_libcufile;

// Field name MUST equal the symbol name; the stringified token is the
// dlsym key. Compiles cleanly under -Wpedantic via the void** round-trip.
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

// Caller holds fs->cache_mu.
static CUfileHandle_t
fs_gds_get_handle_locked(struct store_fs* fs, const char* key)
{
  for (size_t i = 0; i < fs->n_slots; ++i) {
    if (strcmp(fs->slots[i].key, key) == 0) {
      if (fs->slots[i].gds_handle)
        return (CUfileHandle_t)fs->slots[i].gds_handle;
      int fd = platform_file_fd(fs->slots[i].file);
      if (fd < 0)
        return NULL;
      CUfileDescr_t descr;
      memset(&descr, 0, sizeof(descr));
      descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
      descr.handle.fd = fd;
      CUfileHandle_t h = NULL;
      CUfileError_t e = g_libcufile.cuFileHandleRegister(&h, &descr);
      if (e.err != CU_FILE_SUCCESS) {
        log_error("cuFileHandleRegister(%s) failed: err=%d", key, (int)e.err);
        return NULL;
      }
      fs->slots[i].gds_handle = (void*)h;
      return h;
    }
  }
  return NULL;
}

extern platform_file*
store_fs_get_file_external(struct store_fs* fs, const char* key);

static CUfileHandle_t
fs_gds_get_handle(struct store_fs* fs, const char* key)
{
  if (!store_fs_get_file_external(fs, key))
    return NULL;
  CUfileHandle_t h = NULL;
  pthread_mutex_lock(&fs->cache_mu);
  h = fs_gds_get_handle_locked(fs, key);
  pthread_mutex_unlock(&fs->cache_mu);
  return h;
}

// Per-read scratch for cuFileReadAsync. The API stashes pointers to
// these fields and dereferences them when the stream reaches the read,
// so the storage must outlive submit_dev's return. We malloc one array
// per submit_dev call and free it via cuLaunchHostFunc on the same
// stream — that callback fires after every read in the batch has
// retired, so the malloc lifetime is exactly bounded by the IO.
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

struct store_event
store_fs_gds_submit_dev(struct store* s,
                        const struct store_read* reads,
                        size_t n)
{
  struct store_fs* fs = (struct store_fs*)s;
  // seq == 0 is the failure sentinel consumed by wave_pool (DAMACY_IO).
  struct store_event ev = { 0 };

  CUstream stream = (CUstream)fs->gds_stream;
  if (!stream) {
    log_error("store_fs_gds: submit_dev called before stream was set");
    return ev;
  }

  if (n == 0) {
    ev.seq = STORE_FS_GDS_SENTINEL_SEQ;
    return ev;
  }

  struct fs_gds_async_params* params =
    (struct fs_gds_async_params*)calloc(n, sizeof(*params));
  if (!params)
    return ev;

  // Two-pass: resolve every handle first so we never half-submit. The
  // GDS handle table grows under fs->cache_mu inside fs_gds_get_handle;
  // pulling it out of the submit loop keeps the lock-acquire cost off
  // the IO submission path.
  for (size_t i = 0; i < n; ++i) {
    CUfileHandle_t h = fs_gds_get_handle(fs, reads[i].key);
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
      // Already-submitted reads will still execute against `params`.
      // Sync the stream to drain them before freeing the storage; the
      // caller treats seq=0 as fatal and will tear down.
      cuStreamSynchronize(stream);
      free(params);
      return ev;
    }
  }

  // Release the params array after every read in this batch retires.
  // On scheduler failure we fall back to a synchronous drain — slow,
  // but only reachable on resource exhaustion.
  CUresult cr = cuLaunchHostFunc(stream, fs_gds_free_params_cb, params);
  if (cr != CUDA_SUCCESS) {
    log_error("cuLaunchHostFunc(free params) failed: %d", (int)cr);
    cuStreamSynchronize(stream);
    free(params);
    return ev;
  }

  ev.seq = STORE_FS_GDS_SENTINEL_SEQ;
  return ev;
}

void
store_fs_gds_set_stream(struct store* s, void* stream)
{
  if (!s)
    return;
  struct store_fs* fs = (struct store_fs*)s;
  fs->gds_stream = stream;
}

int
store_fs_gds_init(struct store_fs* fs)
{
  if (!fs)
    return 1;
  if (!try_load_libcufile())
    return 1;
  CUfileError_t e = g_libcufile.cuFileDriverOpen();
  if (e.err != CU_FILE_SUCCESS) {
    log_error("cuFileDriverOpen failed (err=%d); GDS unavailable", (int)e.err);
    return 1;
  }
  fs->gds_driver_opened = 1;
  return 0;
}

void
store_fs_gds_destroy(struct store_fs* fs)
{
  if (!fs)
    return;
  // Drain any in-flight cuFileReadAsync + their free callbacks before
  // tearing handles down. The store's stream is owned by wave_pool and
  // is already synced by the time fs_destroy runs, but sync defensively
  // in case a future caller invokes destroy out of order.
  if (fs->gds_stream)
    cuStreamSynchronize((CUstream)fs->gds_stream);
  // Be defensive: a store_fs that never went through init has a NULL
  // function-pointer table.
  pthread_mutex_lock(&fs->cache_mu);
  for (size_t i = 0; i < fs->n_slots; ++i) {
    if (fs->slots[i].gds_handle) {
      if (g_libcufile.cuFileHandleDeregister)
        g_libcufile.cuFileHandleDeregister(
          (CUfileHandle_t)fs->slots[i].gds_handle);
      fs->slots[i].gds_handle = NULL;
    }
  }
  pthread_mutex_unlock(&fs->cache_mu);
  if (fs->gds_driver_opened) {
    if (g_libcufile.cuFileDriverClose)
      g_libcufile.cuFileDriverClose();
    fs->gds_driver_opened = 0;
  }
}
