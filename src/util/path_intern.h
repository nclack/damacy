// Equal strings map to the same const char*. Pointers stay valid
// until the last matching release drops refcount to zero, or until
// path_intern_reset / path_intern_free.
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
  void path_intern_release(struct path_intern* pi, const char* s);

#ifdef __cplusplus
}
#endif
