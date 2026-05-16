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

struct fs_cache_slot
{
  char* key;
  platform_file* file;
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
};

// Bridge for store_fs_gds.c — returns a cached handle (do not close).
platform_file*
store_fs_get_file_external(struct store_fs* fs, const char* key);
