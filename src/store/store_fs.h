// Internal: filesystem store. Public callers use store.h only.
#pragma once

#include "store/store.h"

#include "io_queue/io_queue.h"
#include "platform/platform.h"
#include "platform/platform_io.h"

#include <stddef.h>
#include <stdint.h>

struct lru;
struct lru_entry;

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

struct store_fs
{
  struct store base;
  char* root; // owned
  struct io_queue* q;

  struct platform_mutex* cache_mu;
  struct lru* fd_cache;
};

// Returns a borrowed handle and writes the pin to *pin_out. The pin
// blocks LRU eviction of the entry; must be balanced with
// store_fs_release. On failure returns NULL with *pin_out = NULL.
platform_file*
store_fs_acquire(struct store_fs* fs,
                 const char* key,
                 struct lru_entry** pin_out);

void
store_fs_release(struct store_fs* fs, struct lru_entry* pin);

// Bridge for store_fs_gds.c — returns a cached handle (do not close).
platform_file*
store_fs_get_file_external(struct store_fs* fs, const char* key);
