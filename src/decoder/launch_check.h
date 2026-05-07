// Inline helper for kernel-launch error reporting. Used by all decoder
// kernels (blosc1, shuffle, bitshuffle, memcpy). Kept header-only and
// `static inline` so each TU gets its own copy without ODR concerns.
//
// Returns 0 on success, 1 on any cuda error so callers can
// `return decoder_launch_status_check("tag");` directly.
#pragma once

#include "log/log.h"

#include <cuda_runtime.h>

static inline int
decoder_launch_status_check(const char* tag)
{
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    log_error("%s: %s", tag, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}
