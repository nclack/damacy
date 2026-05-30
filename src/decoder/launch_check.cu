#include "decoder/launch_check.h"

#include "log/log.h"

#include <cuda_runtime.h>

int
decoder_launch_status_check(const char* tag)
{
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    log_error("%s: %s", tag, cudaGetErrorString(e));
    return 1;
  }
  return 0;
}
