// Internal: filesystem store. Public callers use store.h only.
#pragma once

#include "store/store.h"

#include "io_queue/io_queue.h"
#include "platform/platform_io.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

// vtable for backend dispatch. store_destroy and the stat/submit/map
// dispatchers go through this. submit_dev is NULL on backends that
// don't support direct device reads; store_supports_gds returns 0 in
// that case.
struct store_vtable
{
  void (*destroy)(struct store* s);
  int (*stat)(struct store* s, const char* key, uint64_t* out);
  struct store_event (*submit)(struct store* s,
                               const struct store_read* reads,
                               size_t n);
  struct store_event (*submit_dev)(struct store* s,
                                   const struct store_read* reads,
                                   size_t n);
  void (*event_wait)(struct store* s, struct store_event ev);
  int (*event_query)(struct store* s, struct store_event ev);
  int (*map)(struct store* s, const char* key, struct store_view* out);
  void (*unmap)(struct store* s, struct store_view* view);
};

struct store
{
  const struct store_vtable* vt;
};

// Cache slot: key string + open file handle. gds_handle is the registered
// CUfileHandle_t when the store was created with enable_gds and the file
// has been touched on the device path; NULL otherwise. Owned by
// store_fs_gds when populated.
struct fs_cache_slot
{
  char* key;
  platform_file* file;
  void* gds_handle;
};

struct store_fs
{
  struct store base;
  char* root; // owned
  struct io_queue* q;

  // Open-file cache. Linear scan; small until we find we need better.
  pthread_mutex_t cache_mu;
  struct fs_cache_slot* slots;
  size_t n_slots;
  size_t cap_slots;

  // GDS lifecycle. Set to 1 by store_fs_gds_init after a successful
  // cuFileDriverOpen so destroy knows to call cuFileDriverClose. The
  // "is GDS enabled?" question is answered by base.vt: store_supports_gds
  // returns true iff vt->submit_dev is non-NULL.
  uint8_t gds_driver_opened;

  // CUstream (as void* to keep this header CUDA-free) that cuFileReadAsync
  // submissions ride on. Set by wave_pool_init via store_fs_gds_set_stream
  // after the wave pool creates its stream_h2d; submit_dev returns
  // seq=0 if a caller racing init invokes it before the stream is set.
  void* gds_stream;
};
