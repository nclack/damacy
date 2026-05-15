// libcufile is dlopen'd lazily — see store_fs_gds.h. Including
// cufile.h gives us types/enums only; no symbols are referenced
// directly, so libcufile is not linked.

#include "store/store_fs_gds.h"

#include "io_queue/io_queue.h"
#include "log/log.h"
#include "platform/platform_io.h"
#include "store/store.h"
#include "store/store_fs.h"

#include <cufile.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

typedef CUfileError_t (*pfn_cuFileDriverOpen)(void);
typedef CUfileError_t (*pfn_cuFileDriverClose)(void);
typedef CUfileError_t (*pfn_cuFileHandleRegister)(CUfileHandle_t*,
                                                  CUfileDescr_t*);
typedef void (*pfn_cuFileHandleDeregister)(CUfileHandle_t);
typedef ssize_t (*pfn_cuFileRead)(CUfileHandle_t, void*, size_t, off_t, off_t);

// `handle` doubles as a lazy-load state machine: NULL = unattempted,
// LIBCUFILE_FAILED = previously failed (don't retry), else live handle.
#define LIBCUFILE_FAILED ((void*)-1)
static struct
{
  _Atomic(void*) handle;
  pfn_cuFileDriverOpen driver_open;
  pfn_cuFileDriverClose driver_close;
  pfn_cuFileHandleRegister handle_register;
  pfn_cuFileHandleDeregister handle_deregister;
  pfn_cuFileRead read;
} g_libcufile;

static _Atomic int g_libcufile_unavailable_logged = 0;

static int
try_load_libcufile(void)
{
  void* cur = atomic_load_explicit(&g_libcufile.handle, memory_order_acquire);
  if (cur == LIBCUFILE_FAILED)
    return 0;
  if (cur != NULL)
    return 1;

  void* h = dlopen("libcufile.so.0", RTLD_LAZY | RTLD_LOCAL);
  if (!h) {
    atomic_store_explicit(
      &g_libcufile.handle, LIBCUFILE_FAILED, memory_order_release);
    return 0;
  }

  g_libcufile.driver_open = (pfn_cuFileDriverOpen)dlsym(h, "cuFileDriverOpen");
  g_libcufile.driver_close =
    (pfn_cuFileDriverClose)dlsym(h, "cuFileDriverClose");
  g_libcufile.handle_register =
    (pfn_cuFileHandleRegister)dlsym(h, "cuFileHandleRegister");
  g_libcufile.handle_deregister =
    (pfn_cuFileHandleDeregister)dlsym(h, "cuFileHandleDeregister");
  g_libcufile.read = (pfn_cuFileRead)dlsym(h, "cuFileRead");

  if (!g_libcufile.driver_open || !g_libcufile.driver_close ||
      !g_libcufile.handle_register || !g_libcufile.handle_deregister ||
      !g_libcufile.read) {
    dlclose(h);
    atomic_store_explicit(
      &g_libcufile.handle, LIBCUFILE_FAILED, memory_order_release);
    return 0;
  }

  void* expected = NULL;
  if (!atomic_compare_exchange_strong_explicit(&g_libcufile.handle,
                                               &expected,
                                               h,
                                               memory_order_acq_rel,
                                               memory_order_acquire)) {
    // Lost the race; both racers resolved the same SO, so close ours.
    dlclose(h);
  }
  return 1;
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
      CUfileError_t e = g_libcufile.handle_register(&h, &descr);
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

struct fs_gds_read_job
{
  CUfileHandle_t handle;
  void* dst;       // device pointer
  uint64_t offset; // file offset
  size_t len;
  const char* key_for_log; // borrowed pointer into a heap copy held by the job
  char* key_owned;
};

static void
fs_gds_read_job_fn(void* vctx)
{
  struct fs_gds_read_job* j = (struct fs_gds_read_job*)vctx;
  ssize_t got =
    g_libcufile.read(j->handle, j->dst, j->len, (off_t)j->offset, 0);
  if (got < 0) {
    log_error("cuFileRead(%s,off=%llu,len=%zu) failed: ret=%lld",
              j->key_for_log ? j->key_for_log : "?",
              (unsigned long long)j->offset,
              j->len,
              (long long)got);
  }
}

static void
fs_gds_read_job_free(void* vctx)
{
  struct fs_gds_read_job* j = (struct fs_gds_read_job*)vctx;
  free(j->key_owned);
  free(j);
}

struct store_event
store_fs_gds_submit_dev(struct store* s,
                        const struct store_read* reads,
                        size_t n)
{
  struct store_fs* fs = (struct store_fs*)s;
  struct store_event ev = { 0 };
  if (n == 0) {
    struct io_event ioev = io_queue_record(fs->q);
    ev.seq = ioev.seq;
    return ev;
  }
  for (size_t i = 0; i < n; ++i) {
    CUfileHandle_t h = fs_gds_get_handle(fs, reads[i].key);
    if (!h)
      goto Drain;
    struct fs_gds_read_job* j =
      (struct fs_gds_read_job*)calloc(1, sizeof(struct fs_gds_read_job));
    if (!j)
      goto Drain;
    j->handle = h;
    j->dst = reads[i].dst;
    j->offset = reads[i].offset;
    j->len = reads[i].len;
    j->key_owned = strdup(reads[i].key);
    j->key_for_log = j->key_owned;
    if (!j->key_owned) {
      free(j);
      goto Drain;
    }
    if (io_queue_post(fs->q, fs_gds_read_job_fn, j, fs_gds_read_job_free))
      goto Drain;
  }
  struct io_event ioev = io_queue_record(fs->q);
  ev.seq = ioev.seq;
  return ev;
Drain:
  io_event_wait(fs->q, io_queue_record(fs->q));
  return ev; // ev.seq == 0 → failure
}

int
store_fs_gds_init(struct store_fs* fs)
{
  if (!fs)
    return 1;
  if (!try_load_libcufile()) {
    int expected = 0;
    if (atomic_compare_exchange_strong(
          &g_libcufile_unavailable_logged, &expected, 1))
      log_error(
        "cuFile: libcufile.so.0 not loadable — install the cuFile runtime "
        "(part of CUDA toolkit / nvidia-fs) and ensure it is on the dynamic "
        "loader path; GDS unavailable");
    return 1;
  }
  CUfileError_t e = g_libcufile.driver_open();
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
  // Be defensive: a store_fs that never went through init has a NULL
  // function-pointer table.
  pthread_mutex_lock(&fs->cache_mu);
  for (size_t i = 0; i < fs->n_slots; ++i) {
    if (fs->slots[i].gds_handle) {
      if (g_libcufile.handle_deregister)
        g_libcufile.handle_deregister((CUfileHandle_t)fs->slots[i].gds_handle);
      fs->slots[i].gds_handle = NULL;
    }
  }
  pthread_mutex_unlock(&fs->cache_mu);
  if (fs->gds_driver_opened) {
    if (g_libcufile.driver_close)
      g_libcufile.driver_close();
    fs->gds_driver_opened = 0;
  }
}
