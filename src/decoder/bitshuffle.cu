#include "decoder/bitshuffle.h"

#include "damacy_limits.h"
#include "decoder/launch_check.h"
#include "log/log.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;

// Reverse blosc1 bit-shuffle for one block.
//
// Bitshuffle (forward) reorganises N elements of T bytes (T*8 bit-planes
// of N bits each) so that all bits of plane p ∈ [0, T*8) sit
// consecutively, packed LSB-first into N/8 bytes. So plane p occupies
// `[p*(N/8), (p+1)*(N/8))` of the block; bit-position e within the plane
// is at byte `e/8`, bit `e%8`.
//
// Output byte at position `off = e*T + bb` is reassembled from 8 plane
// bits: for bp ∈ [0,8), bit bp of the output byte comes from plane
// (bb*8 + bp), bit-position e.
//
// Same scratch-buffer model as gpu_unshuffle: phase 1 copies d_block →
// scratch; __syncthreads; phase 2 scatters reassembled bytes back.
__global__ void
gpu_bitunshuffle_kernel(const struct gpu_shuffle_op* __restrict__ ops,
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
  const uint32_t bytes_per_plane = N >> 3;
  uint8_t* d_block = static_cast<uint8_t*>(op.d_buf) + (size_t)bi * blocksize;
  const size_t chunk_off =
    (size_t)((const uint8_t*)op.d_buf - dev_decompressed_base);
  uint8_t* s_block = scratch_base + chunk_off + (size_t)bi * blocksize;

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

  for (uint32_t off = threadIdx.x; off < blocksize; off += kThreadsPerBlock) {
    const uint32_t e = off / T;
    const uint32_t bb = off - e * T;
    const uint32_t e_byte = e >> 3;
    const uint32_t e_bit = e & 7u;
    uint8_t out = 0;
#pragma unroll
    for (uint32_t bp = 0; bp < 8; ++bp) {
      const uint8_t plane_byte =
        s_block[(bb * 8u + bp) * bytes_per_plane + e_byte];
      const uint8_t bit = (plane_byte >> e_bit) & 1u;
      out |= (uint8_t)(bit << bp);
    }
    d_block[off] = out;
  }
}

} // namespace

extern "C" int
gpu_bitunshuffle_launch(CUstream stream,
                        const struct gpu_shuffle_op* d_ops,
                        uint32_t n_ops,
                        const void* dev_decompressed_base,
                        void* scratch_base)
{
  if (n_ops == 0)
    return 0;
  if (!dev_decompressed_base || !scratch_base) {
    log_error("gpu_bitunshuffle_launch: scratch pointer is NULL");
    return 1;
  }
  dim3 grid(DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK, n_ops, 1);
  gpu_bitunshuffle_kernel<<<grid, kThreadsPerBlock, 0, (cudaStream_t)stream>>>(
    d_ops, (const uint8_t*)dev_decompressed_base, (uint8_t*)scratch_base);
  return decoder_launch_status_check("gpu_bitunshuffle_launch") == cudaSuccess
           ? 0
           : 1;
}
