#include "decoder/blosc1_parse.h"

#include "assemble/assemble.h"
#include "damacy_limits.h"
#include "decoder/launch_check.h"
#include "planner/planner.h"
#include "zarr/zarr_chunk_layout.h"
#include "zarr/zarr_metadata.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t kBloscHeaderBytes = 16u;
constexpr uint32_t kChunkScanThreadsPerBlock = 128u;
constexpr uint32_t kBlockFanoutThreadsPerBlock = 128u;

__device__ inline uint32_t
read_u32_le(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

__device__ inline void
record_parse_err(uint32_t* d_parse_err, uint32_t code)
{
  atomicCAS(d_parse_err, 0u, code);
}

// Kernel A: one thread per BLOSC_ZSTD chunk. Reads the chunk's flags
// byte and validates cbytes; emits a single memcpy op for memcpyed
// chunks and writes the chunk's assemble shuffle fields. Sets a bit in
// d_is_memcpyed[chunk_idx] for memcpyed chunks so Kernel B's per-block
// fan-out skips them.
__global__ void
blosc1_chunk_scan_kernel(const uint8_t* __restrict__ d_compressed,
                         uint8_t* __restrict__ d_decompressed,
                         const struct gpu_parse_chunk* __restrict__ d_chunks,
                         const struct sample_plan* __restrict__ d_sample_plans,
                         const uint32_t* __restrict__ d_blosc_chunk_indices,
                         struct gpu_memcpy_op* __restrict__ d_memcpy_ops,
                         struct assemble_chunk* __restrict__ d_assemble_chunks,
                         uint32_t* __restrict__ d_is_memcpyed,
                         uint32_t* __restrict__ d_n_memcpy,
                         uint32_t* __restrict__ d_parse_err,
                         uint32_t n_blosc_zstd_chunks)
{
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_blosc_zstd_chunks)
    return;

  const uint32_t chunk_idx = d_blosc_chunk_indices[tid];
  const struct gpu_parse_chunk chunk = d_chunks[chunk_idx];
  const struct sample_plan& sp = d_sample_plans[chunk.sample_idx_in_batch];

  const uint8_t* d_comp = d_compressed + chunk.compressed_offset;
  const uint8_t flags = d_comp[2];
  const uint32_t cbytes = read_u32_le(d_comp + 12);
  if (cbytes != chunk.compressed_nbytes) {
    record_parse_err(d_parse_err, 4u);
    return;
  }

  const uint32_t nblocks = sp.layout.nblocks;
  const bool memcpyed = ((flags >> 1) & 0x1u) != 0u;

  struct assemble_chunk* a = &d_assemble_chunks[chunk_idx];
  if (memcpyed) {
    // Set the per-chunk bit; Kernel B reads it to skip these chunks.
    atomicOr(&d_is_memcpyed[chunk_idx >> 5], 1u << (chunk_idx & 31u));
    const uint32_t overhead = kBloscHeaderBytes + 4u * nblocks;
    const uint32_t slot = atomicAdd(d_n_memcpy, 1u);
    struct gpu_memcpy_op op;
    op.d_src = (uint8_t*)d_comp + overhead;
    op.d_dst = d_decompressed + chunk.decompressed_offset;
    op.nbytes = chunk.decompressed_nbytes;
    d_memcpy_ops[slot] = op;
    a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_NONE;
    a->shuffle_typesize = 0;
    a->shuffle_blocksize = 0;
  } else if (sp.layout.shuffle) {
    a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_BYTE;
    a->shuffle_typesize = sp.layout.typesize;
    a->shuffle_blocksize = sp.layout.blocksize;
  } else if (sp.layout.bitshuffle) {
    a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_BIT;
    a->shuffle_typesize = sp.layout.typesize;
    a->shuffle_blocksize = sp.layout.blocksize;
  } else {
    a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_NONE;
    a->shuffle_typesize = 0;
    a->shuffle_blocksize = 0;
  }
}

// Kernel B: one thread per blosc block across non-memcpyed BLOSC_ZSTD
// chunks. Each thread reads its bstart and the 4B cb prefix, classifies
// the substream as raw (cb == blocksize, emit a memcpy op) or codec
// (emit a zstd-fanout entry). Warp ballot + popc compacts the per-lane
// results into one atomicAdd per warp per (raw/codec) class for slot
// allocation.
__global__ void
blosc1_block_fanout_kernel(
  const uint8_t* __restrict__ d_compressed,
  uint8_t* __restrict__ d_decompressed,
  const struct gpu_parse_chunk* __restrict__ d_chunks,
  const struct sample_plan* __restrict__ d_sample_plans,
  const uint32_t* __restrict__ d_block_chunk_map,
  const uint32_t* __restrict__ d_is_memcpyed,
  const void** __restrict__ zstd_comp_ptrs,
  size_t* __restrict__ zstd_comp_sizes,
  void** __restrict__ zstd_decomp_ptrs,
  size_t* __restrict__ zstd_decomp_buf_sizes,
  struct gpu_memcpy_op* __restrict__ d_memcpy_ops,
  uint32_t* __restrict__ d_n_zstd,
  uint32_t* __restrict__ d_n_memcpy,
  uint32_t* __restrict__ d_parse_err,
  uint32_t n_blosc_zstd_blocks)
{
  static_assert(kBlockFanoutThreadsPerBlock % 32u == 0u,
                "block-fan-out kernel uses warp-level ballot/popc");
  constexpr uint32_t kWarpsPerBlock = kBlockFanoutThreadsPerBlock / 32u;
  __shared__ uint32_t s_codec_base[kWarpsPerBlock];
  __shared__ uint32_t s_raw_base[kWarpsPerBlock];

  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp_id = threadIdx.x >> 5;

  bool is_codec = false;
  bool is_raw = false;
  const uint8_t* src_ptr = nullptr;
  uint8_t* dst_ptr = nullptr;
  uint32_t cb = 0;
  uint32_t blocksize = 0;

  if (tid < n_blosc_zstd_blocks) {
    const uint32_t packed = d_block_chunk_map[tid];
    const uint32_t chunk_idx = packed >> 16;
    const uint32_t block_idx = packed & 0xFFFFu;

    const uint32_t mc_bit =
      (d_is_memcpyed[chunk_idx >> 5] >> (chunk_idx & 31u)) & 1u;
    if (mc_bit == 0u) {
      const struct gpu_parse_chunk chunk = d_chunks[chunk_idx];
      const struct sample_plan& sp = d_sample_plans[chunk.sample_idx_in_batch];
      const uint32_t nblocks = sp.layout.nblocks;
      blocksize = sp.layout.blocksize;
      const uint32_t payload_lo = kBloscHeaderBytes + 4u * nblocks;
      const uint8_t* d_comp = d_compressed + chunk.compressed_offset;
      uint8_t* d_decomp = d_decompressed + chunk.decompressed_offset;

      const uint32_t bstart =
        read_u32_le(d_comp + kBloscHeaderBytes + 4u * block_idx);
      if (bstart < payload_lo || bstart + 4u > chunk.compressed_nbytes) {
        record_parse_err(d_parse_err, 9u);
      } else {
        cb = read_u32_le(d_comp + bstart);
        const uint32_t cur = bstart + 4u;
        if (cb > chunk.compressed_nbytes - cur) {
          record_parse_err(d_parse_err, 11u);
        } else {
          src_ptr = d_comp + cur;
          dst_ptr = d_decomp + block_idx * blocksize;
          if (cb == blocksize)
            is_raw = true;
          else
            is_codec = true;
        }
      }
    }
  }

  const uint32_t mask_codec = __ballot_sync(0xFFFFFFFFu, is_codec);
  const uint32_t mask_raw = __ballot_sync(0xFFFFFFFFu, is_raw);

  if (lane == 0) {
    s_codec_base[warp_id] =
      mask_codec ? atomicAdd(d_n_zstd, __popc(mask_codec)) : 0u;
    s_raw_base[warp_id] =
      mask_raw ? atomicAdd(d_n_memcpy, __popc(mask_raw)) : 0u;
  }
  __syncwarp();

  if (is_codec) {
    const uint32_t rank = __popc(mask_codec & ((1u << lane) - 1u));
    const uint32_t slot = s_codec_base[warp_id] + rank;
    zstd_comp_ptrs[slot] = src_ptr;
    zstd_comp_sizes[slot] = cb;
    zstd_decomp_ptrs[slot] = dst_ptr;
    zstd_decomp_buf_sizes[slot] = blocksize;
  }
  if (is_raw) {
    const uint32_t rank = __popc(mask_raw & ((1u << lane) - 1u));
    const uint32_t slot = s_raw_base[warp_id] + rank;
    struct gpu_memcpy_op op;
    op.d_src = (uint8_t*)src_ptr;
    op.d_dst = dst_ptr;
    op.nbytes = blocksize;
    d_memcpy_ops[slot] = op;
  }
}

} // namespace

extern "C" int
blosc1_parse_launch(CUstream stream, const struct blosc1_parse_args* args)
{
  if (!args || args->n_blosc_zstd_chunks == 0u)
    return 0;

  const uint32_t scan_blocks =
    (args->n_blosc_zstd_chunks + kChunkScanThreadsPerBlock - 1u) /
    kChunkScanThreadsPerBlock;
  blosc1_chunk_scan_kernel<<<scan_blocks,
                             kChunkScanThreadsPerBlock,
                             0,
                             (cudaStream_t)stream>>>(
    args->d_compressed,
    args->d_decompressed,
    args->d_chunks,
    args->d_sample_plans,
    args->d_blosc_chunk_indices,
    args->d_memcpy_ops,
    args->d_assemble_chunks,
    args->d_is_memcpyed,
    args->d_n_memcpy,
    args->d_parse_err,
    args->n_blosc_zstd_chunks);
  int rc = decoder_launch_status_check("blosc1_chunk_scan_kernel");
  if (rc != 0)
    return rc;

  if (args->n_blosc_zstd_blocks == 0u)
    return 0;

  const uint32_t fanout_blocks =
    (args->n_blosc_zstd_blocks + kBlockFanoutThreadsPerBlock - 1u) /
    kBlockFanoutThreadsPerBlock;
  blosc1_block_fanout_kernel<<<fanout_blocks,
                               kBlockFanoutThreadsPerBlock,
                               0,
                               (cudaStream_t)stream>>>(
    args->d_compressed,
    args->d_decompressed,
    args->d_chunks,
    args->d_sample_plans,
    args->d_block_chunk_map,
    args->d_is_memcpyed,
    args->zstd.d_comp_ptrs,
    args->zstd.d_comp_sizes,
    args->zstd.d_decomp_ptrs,
    args->zstd.d_decomp_buf_sizes,
    args->d_memcpy_ops,
    args->d_n_zstd,
    args->d_n_memcpy,
    args->d_parse_err,
    args->n_blosc_zstd_blocks);
  return decoder_launch_status_check("blosc1_block_fanout_kernel");
}
