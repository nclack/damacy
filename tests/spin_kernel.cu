#include "spin_kernel.h"
__global__ static void
spin_kernel(int64_t cycles)
{
  const int64_t start = clock64();
  while (clock64() - start < cycles) {
  }
}
extern "C" cudaError_t
test_spin_launch(cudaStream_t stream, int64_t cycles)
{
  spin_kernel<<<1, 1, 0, stream>>>(cycles);
  return cudaGetLastError();
}
