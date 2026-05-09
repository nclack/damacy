#pragma once

#include "log/log.h"

#include <cuda.h>
#include <nvcomp.h>

#define CU(label, expr)                                                        \
  do {                                                                         \
    CUresult _r = (expr);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      const char* _msg = NULL;                                                 \
      cuGetErrorString(_r, &_msg);                                             \
      log_error("cu: %s -> %s", #expr, _msg ? _msg : "?");                     \
      goto label;                                                              \
    }                                                                          \
  } while (0)

#define NV(label, expr)                                                        \
  do {                                                                         \
    nvcompStatus_t _s = (expr);                                                \
    if (_s != nvcompSuccess) {                                                 \
      log_error("nvcomp: %s -> nvcompStatus=%d", #expr, (int)_s);              \
      goto label;                                                              \
    }                                                                          \
  } while (0)

// CUdeviceptr is a 64-bit handle; cast through uintptr_t to silence
// pointer-to-int diagnostics. Use the inverse cast at allocation sites.
#define CUDPTR(p) ((CUdeviceptr)(uintptr_t)(p))
