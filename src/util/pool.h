// Fixed-size object pool. Intrusive LIFO freelist through unused slots.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct pool;

  // elem_size >= sizeof(void*) so the freelist link fits in a slot.
  struct pool* pool_create(size_t elem_size, size_t capacity);
  void pool_destroy(struct pool* p);

  // NULL on exhaustion.
  void* pool_alloc(struct pool* p);
  void pool_free(struct pool* p, void* ptr);

  size_t pool_in_use(struct pool* p);
  size_t pool_capacity(const struct pool* p);

#ifdef __cplusplus
}
#endif
