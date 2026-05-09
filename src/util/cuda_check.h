// CUDA driver-API check macros + the CUdeviceptr cast helper. Split out
// of decoder/decoder_check.h so non-decoder modules (damacy.c,
// batch_pool, wave) don't have to drag in nvcomp.h just to log a
// driver error. decoder/decoder_check.h re-includes this and adds the
// NV() macro on top.
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
