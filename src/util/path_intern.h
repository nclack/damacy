// Equal strings map to the same const char*. Pointers stay valid
// until the last matching release drops refcount to zero, or until
// path_intern_reset / path_intern_free.
//
// Not thread-safe; caller serializes.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct path_intern_slot;

  struct path_intern
  {
    struct path_intern_slot* slots;
    size_t cap;
    size_t n;
  };

  void path_intern_free(struct path_intern* pi);

  // Drop every entry regardless of refcount.
  void path_intern_reset(struct path_intern* pi);

  const char* path_intern_acquire(struct path_intern* pi, const char* s);

  // `s` must be the pointer returned from acquire — release looks up by
  // pointer identity, not string content.
  void path_intern_release(struct path_intern* pi, const char* s);

  // Linear scan; for debug asserts on cache-key contracts.
  int path_intern_owns(const struct path_intern* pi, const char* p);

#ifdef __cplusplus
}
#endif
