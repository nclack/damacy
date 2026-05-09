#pragma once

#include "util/cuda_check.h" // CU + CUDPTR

#include <nvcomp.h>

#define NV(label, expr)                                                        \
  do {                                                                         \
    nvcompStatus_t _s = (expr);                                                \
    if (_s != nvcompSuccess) {                                                 \
      log_error("nvcomp: %s -> nvcompStatus=%d", #expr, (int)_s);              \
      goto label;                                                              \
    }                                                                          \
  } while (0)
