#include "decoder/shuffle.h"

#include "damacy_limits.h"
#include "decoder/launch_check.h"
#include "log/log.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;

// Reverse byte-shuffle for one blosc block, using a per-wave global
// scratch buffer. Phase 1: copy d_block (shuffled) → scratch_block.
// Phase 2: read scratch_block at transposed offsets, write d_block.
// __syncthreads guarantees global-mem ordering between phases for
// threads in this CUDA block; each chunk's scratch slice is private to
// this block (chunks don't share decompressed offsets).
__global__ void
gpu_unshuffle_kernel(const struct gpu_shuffle_op* __restrict__ ops,
                     const uint8_t* __restrict__ dev_decompressed_base,
                     uint8_t* __restrict__ scratch_base)
{
  const struct gpu_shuffle_op op = ops[blockIdx.y];
  const uint32_t bi = blockIdx.x;
  if (bi >= op.nblocks_full)
    return;

  const uint32_t blocksize = op.blocksize;
  const uint32_t T = op.typesize;
  const uint32_t N = blocksize / T;
  uint8_t* d_block = static_cast<uint8_t*>(op.d_buf) + (size_t)bi * blocksize;
  const size_t chunk_off =
    (size_t)((const uint8_t*)op.d_buf - dev_decompressed_base);
  uint8_t* s_block = scratch_base + chunk_off + (size_t)bi * blocksize;

  // Phase 1: copy d_block → s_block.
  if ((blocksize & 0xfu) == 0u && ((uintptr_t)d_block & 0xfu) == 0u &&
      ((uintptr_t)s_block & 0xfu) == 0u) {
    const uint4* src4 = reinterpret_cast<const uint4*>(d_block);
    uint4* dst4 = reinterpret_cast<uint4*>(s_block);
    const uint32_t n4 = blocksize >> 4;
    for (uint32_t i = threadIdx.x; i < n4; i += kThreadsPerBlock)
      dst4[i] = src4[i];
  } else {
    for (uint32_t i = threadIdx.x; i < blocksize; i += kThreadsPerBlock)
      s_block[i] = d_block[i];
  }
  __syncthreads();

  // Phase 2: scatter from shuffled s_block to natural d_block.
  for (uint32_t off = threadIdx.x; off < blocksize; off += kThreadsPerBlock) {
    const uint32_t i = off / T;
    const uint32_t b = off - i * T;
    d_block[off] = s_block[b * N + i];
  }
}

} // namespace

extern "C" int
gpu_unshuffle_launch(CUstream stream,
                     const struct gpu_shuffle_op* d_ops,
                     uint32_t n_ops,
                     const void* dev_decompressed_base,
                     void* scratch_base)
{
  if (n_ops == 0)
    return 0;
  if (!dev_decompressed_base || !scratch_base) {
    log_error("gpu_unshuffle_launch: scratch pointer is NULL");
    return 1;
  }
  dim3 grid(DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK, n_ops, 1);
  gpu_unshuffle_kernel<<<grid, kThreadsPerBlock, 0, (cudaStream_t)stream>>>(
    d_ops, (const uint8_t*)dev_decompressed_base, (uint8_t*)scratch_base);
  return decoder_launch_status_check("gpu_unshuffle_launch");
}
