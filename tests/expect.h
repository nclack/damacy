// EXPECT / RUN macros for self-checking test executables. Both lower
// onto the project logger so test output looks like all the other
// stderr we produce.
#pragma once

#include "log/log.h"

#define EXPECT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      log_error("expect failed: %s", #cond);                                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define RUN(t)                                                                 \
  do {                                                                         \
    int r = t();                                                               \
    if (r != 0) {                                                              \
      log_error("FAIL %s", #t);                                                \
      return r;                                                                \
    }                                                                          \
    log_info("ok   %s", #t);                                                   \
  } while (0)
