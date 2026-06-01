// Backend-author header: store layout + vtable. Public consumers use store.h.
#pragma once

#include "store/store.h"

#include <stddef.h>
#include <stdint.h>

struct store_vtable
{
  void (*destroy)(struct store* s);
  enum store_stat_result (*stat)(struct store* s,
                                 const char* key,
                                 uint64_t* out);
  struct store_submit_result (*submit)(struct store* s,
                                       const struct store_read* reads,
                                       size_t n);
  struct store_submit_result (*submit_dev)(struct store* s,
                                           const struct store_read* reads,
                                           size_t n);
  void (*event_wait)(struct store* s, struct store_event ev);
  int (*event_query)(struct store* s, struct store_event ev);
  void (*event_discard)(struct store* s, struct store_event ev);
  int (*map)(struct store* s, const char* key, struct store_view* out);
  void (*unmap)(struct store* s, struct store_view* view);
};

struct store
{
  const struct store_vtable* vt;
};

// Splits the RLIMIT_NOFILE budget so host + GDS stores can coexist.
uint32_t
store_default_fd_cache_capacity(void);
