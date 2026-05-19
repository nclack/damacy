// Fixed-size object pool. Intrusive LIFO freelist through unused slots.
#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct pool;

  // Slots are sized up to max(elem_size, sizeof(void*)) and rounded to
  // _Alignof(max_align_t) so any standard scalar type stays aligned.
  struct pool* pool_create(size_t elem_size, size_t capacity);
  void pool_destroy(struct pool* p);

  // NULL on exhaustion.
  void* pool_alloc(struct pool* p);
  void pool_free(struct pool* p, void* ptr);

  bool pool_owns(const struct pool* p, const void* ptr);

  size_t pool_in_use(const struct pool* p);
  size_t pool_capacity(const struct pool* p);

#ifdef __cplusplus
}
#endif
