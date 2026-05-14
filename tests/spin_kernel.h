#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif
  cudaError_t test_spin_launch(cudaStream_t stream, int64_t cycles);
#ifdef __cplusplus
}
#endif
