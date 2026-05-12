// CUDA driver-API check macros + the CUdeviceptr cast helper.
// decoder/decoder_check.h re-includes this and adds NV() for nvcomp.
#pragma once

#include "log/log.h"

#include <cuda.h>

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

// CUdeviceptr is a 64-bit handle; cast through uintptr_t to silence
// pointer-to-int diagnostics. Use the inverse cast at allocation sites.
#define CUDPTR(p) ((CUdeviceptr)(uintptr_t)(p))
