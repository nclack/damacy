// Tiny CUDA bootstrap for tests/bench. damacy_create requires a
// CUcontext current on the calling thread; under PyTorch this is
// already true, but pure-C callers have to do it themselves. Header-
// only so each test/bench main can drop it in without CMake gymnastics.
#pragma once

#include <cuda.h>

static inline int
cuda_init_primary(void)
{
  if (cuInit(0) != CUDA_SUCCESS)
    return 1;
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS)
    return 1;
  CUcontext ctx;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS)
    return 1;
  if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS)
    return 1;
  return 0;
}
