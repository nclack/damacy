#include "store/store_fs_gds.h"

#include "io_queue/io_queue.h"
#include "log/log.h"
#include "platform/platform_io.h"
#include "store/store.h"
#include "store/store_fs.h"
#include "util/prelude.h"

#include <cufile.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

// Stamp out the writable vtable that store_fs.c installed when its
// fs_vtable is loaded; we patch submit_dev on it after gds init.
extern struct store_vtable*
store_fs_vtable_mut(void);

// Look up (or open + register) the cuFileHandle_t for `key`. Returns a
// CUfileHandle_t (NULL on failure). The handle is owned by the
// fs_cache_slot's gds_handle field; lifetime is the slot's.
static CUfileHandle_t
fs_gds_get_handle_locked(struct store_fs* fs, const char* key)
{
  // fs->cache_mu held by caller.
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
      CUfileError_t e = cuFileHandleRegister(&h, &descr);
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

// Helper used by fs_submit_dev: locate (or open) the file for `key`,
// register a cuFileHandle if not yet, return the handle. NULL on
// failure.
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
  // cuFileRead returns bytes read on success, -1 on error.
  ssize_t got = cuFileRead(j->handle, j->dst, j->len, (off_t)j->offset, 0);
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

static struct store_event
fs_submit_dev(struct store* s, const struct store_read* reads, size_t n)
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
  CUfileError_t e = cuFileDriverOpen();
  if (e.err != CU_FILE_SUCCESS) {
    log_info("cuFileDriverOpen failed (err=%d); GDS unavailable", (int)e.err);
    return 1;
  }
  fs->gds_driver_opened = 1;
  fs->gds_enabled = 1;
  // Patch submit_dev into the vtable; non-GDS callers retain the
  // host-staging submit.
  store_fs_vtable_mut()->submit_dev = fs_submit_dev;
  return 0;
}

void
store_fs_gds_destroy(struct store_fs* fs)
{
  if (!fs)
    return;
  pthread_mutex_lock(&fs->cache_mu);
  for (size_t i = 0; i < fs->n_slots; ++i) {
    if (fs->slots[i].gds_handle) {
      cuFileHandleDeregister((CUfileHandle_t)fs->slots[i].gds_handle);
      fs->slots[i].gds_handle = NULL;
    }
  }
  pthread_mutex_unlock(&fs->cache_mu);
  if (fs->gds_driver_opened) {
    cuFileDriverClose();
    fs->gds_driver_opened = 0;
  }
  fs->gds_enabled = 0;
}
