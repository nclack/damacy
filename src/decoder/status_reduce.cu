#include "decoder/status_reduce.h"

#include "decoder/launch_check.h"

#include <cuda_runtime.h>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;

// Block-stride scan; each thread accumulates into a register, warp-reduces,
// then thread 0 of each warp atomicAdds into the global counter. Cheap
// at the sizes we care about (typically <= 8K statuses per codec batch).
__global__ void
status_reduce_kernel(const int* __restrict__ d_statuses,
                     uint32_t* __restrict__ d_counter,
                     uint32_t n)
{
  uint32_t local = 0;
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    if (d_statuses[i] != 0)
      ++local;
  }
#pragma unroll
  for (int s = 16; s > 0; s >>= 1)
    local += __shfl_down_sync(0xffffffffu, local, s);
  if ((threadIdx.x & 31u) == 0u && local != 0)
    atomicAdd(d_counter, local);
}

} // namespace

extern "C" int
decoder_status_reduce_launch(CUstream stream,
                             const int* d_statuses,
                             uint32_t* d_error_counter,
                             uint32_t n)
{
  if (n == 0)
    return 0;
  // One block per ~kThreadsPerBlock entries; cap at 64 to keep launch
  // overhead bounded for very large batches. Block-stride loop covers
  // the rest.
  uint32_t blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
  if (blocks > 64u)
    blocks = 64u;
  status_reduce_kernel<<<blocks,
                         kThreadsPerBlock,
                         0,
                         (cudaStream_t)stream>>>(d_statuses, d_error_counter, n);
  return decoder_launch_status_check("decoder_status_reduce_launch");
}
