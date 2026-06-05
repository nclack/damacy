#include "cuda_init.h"

#include <cuda.h>

int
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
