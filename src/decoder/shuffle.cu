#include "decoder/shuffle.h"

#include "damacy_limits.h"
#include "log/log.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;

// Reverse byte-shuffle for one blosc block. Loads the block (shuffled
// layout) into shared memory, then writes back transposed:
//   d_block[i*T + b] = smem[b*N + i],  N = blocksize / typesize.
//
// Bank conflicts are tolerated: stride b*N has N divisible by 32 in
// typical blocksizes (64 KB / typesize 4 → N=16384 → stride a multiple
// of 32). Could be padded out at the cost of shmem; not done here.
__global__ void
gpu_unshuffle_kernel(const struct gpu_shuffle_op* __restrict__ ops,
                     uint32_t blocksize)
{
  const struct gpu_shuffle_op op = ops[blockIdx.y];
  const uint32_t bi = blockIdx.x;
  if (bi >= op.nblocks_full)
    return;

  extern __shared__ uint8_t smem[];
  uint8_t* d_block = static_cast<uint8_t*>(op.d_buf) + (size_t)bi * blocksize;

  // Stage 1: copy d_block (shuffled) → smem.
  if ((blocksize & 0xfu) == 0u && ((uintptr_t)d_block & 0xfu) == 0u) {
    const uint4* src4 = reinterpret_cast<const uint4*>(d_block);
    uint4* dst4 = reinterpret_cast<uint4*>(smem);
    const uint32_t n4 = blocksize >> 4;
    for (uint32_t i = threadIdx.x; i < n4; i += kThreadsPerBlock)
      dst4[i] = src4[i];
  } else {
    for (uint32_t i = threadIdx.x; i < blocksize; i += kThreadsPerBlock)
      smem[i] = d_block[i];
  }
  __syncthreads();

  // Stage 2: scatter smem → d_block (natural layout).
  const uint32_t T = op.typesize;
  const uint32_t N = blocksize / T;
  for (uint32_t off = threadIdx.x; off < blocksize; off += kThreadsPerBlock) {
    const uint32_t i = off / T;
    const uint32_t b = off - i * T;
    d_block[off] = smem[b * N + i];
  }
}

cudaError_t
launch_status_check(const char* tag)
{
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess)
    log_error("%s: %s", tag, cudaGetErrorString(e));
  return e;
}

} // namespace

extern "C" int
gpu_unshuffle_launch(CUstream stream,
                     const struct gpu_shuffle_op* d_ops,
                     uint32_t n_ops,
                     uint32_t blocksize)
{
  if (n_ops == 0)
    return 0;
  if (blocksize == 0) {
    log_error("gpu_unshuffle_launch: blocksize=0");
    return 1;
  }
  // Opt in to up to 64 KB of dynamic shared memory per block; sm_75's
  // default cap is 48 KB. Idempotent once set.
  static int s_smem_opt_in_done = 0;
  if (!s_smem_opt_in_done) {
    cudaFuncSetAttribute(
      gpu_unshuffle_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
    s_smem_opt_in_done = 1;
  }
  if (blocksize > 65536u) {
    log_error("gpu_unshuffle_launch: blocksize=%u exceeds 64 KB shmem cap",
              blocksize);
    return 1;
  }
  dim3 grid(DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK, n_ops, 1);
  gpu_unshuffle_kernel<<<grid,
                         kThreadsPerBlock,
                         blocksize,
                         (cudaStream_t)stream>>>(d_ops, blocksize);
  return launch_status_check("gpu_unshuffle_launch") == cudaSuccess ? 0 : 1;
}
