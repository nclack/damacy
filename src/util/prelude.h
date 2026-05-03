/// PRIVATE: never include in other headers.
///
/// Small utility macros: countof, container_of, and the
/// checked-precondition macros (CHECK / CHECK_SILENT / CHECK_MUL_OVERFLOW).
/// The CHECK family logs the failing expression at LOG_ERROR and gotos a
/// caller-provided label so error paths read top-to-bottom and free their
/// resources in one place.
///
/// CHECK expansions reference log_error from "log/log.h" — sources that
/// use CHECK must include that header themselves. Sources that only need
/// countof / container_of can include prelude.h alone (e.g. parse-only
/// modules that don't want a logger dependency).
///
/// Macros only — no `static inline` definitions. Anything that needs a
/// real function lives in its own .c file (see util/hash.c).
#pragma once

#include <stddef.h>

#define container_of(ptr, type, member)                                        \
  ((type*)((char*)(ptr) - offsetof(type, member)))

#define countof(arr) (sizeof(arr) / sizeof((arr)[0]))

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
