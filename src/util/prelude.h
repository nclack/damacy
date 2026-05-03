/// PRIVATE: never include in other headers.
///
/// Common helpers used throughout the implementation files: logging
/// wrappers (log_*), checked-precondition macros (CHECK / CHECK_SILENT),
/// and a few small inline utilities. The CHECK macros log the failing
/// expression at LOG_ERROR and goto a caller-provided label so error
/// paths read top-to-bottom and free their resources in one place.
#pragma once

#include "log/log.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define container_of(ptr, type, member)                                        \
  ((type*)((char*)(ptr) - offsetof(type, member)))

#define countof(e) (sizeof(e) / sizeof((e)[0]))

#define CHECK(lbl, e)                                                          \
  do {                                                                         \
    if (!(e)) {                                                                \
      log_error("check failed: %s", #e);                                       \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

#define CHECK_SILENT(lbl, e)                                                   \
  do {                                                                         \
    if (!(e))                                                                  \
      goto lbl;                                                                \
  } while (0)

#define CHECK_MUL_OVERFLOW(lbl, a, b, max_val)                                 \
  do {                                                                         \
    if ((b) != 0 && (a) > (max_val) / (b)) {                                   \
      log_error("overflow: %llu * %llu > %llu",                                \
                (unsigned long long)(a),                                       \
                (unsigned long long)(b),                                       \
                (unsigned long long)(max_val));                                \
      goto lbl;                                                                \
    }                                                                          \
  } while (0)

  static inline size_t align_up(size_t x, size_t alignment)
  {
    return (x + alignment - 1) / alignment * alignment;
  }

  static inline int ceil_log2(uint64_t v)
  {
    int p = 0;
    while ((1ull << p) < v)
      ++p;
    return p;
  }

#ifdef __cplusplus
}
#endif
