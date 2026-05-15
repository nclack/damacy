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

constexpr uint32_t kThreadsPerBlock = DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK;
constexpr uint32_t kBloscHeaderBytes = 16u;

__device__ inline uint32_t
read_u32_le(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

__device__ inline void
record_parse_err(uint32_t* d_parse_err, uint32_t code)
{
  // First failure wins. Subsequent CASes no-op.
  atomicCAS(d_parse_err, 0u, code);
}

// One CUDA block handles one chunk. Up to nblocks threads classify the
// chunk's blocks (≤ DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK = 32 so the block
// is one warp). Per-warp ballot + atomicAdd packs each block's
// contribution into the global SOA in two atomic ops per chunk worst
// case.
__global__ void
blosc1_parse_kernel(const uint8_t* __restrict__ d_compressed,
                    uint8_t* __restrict__ d_decompressed,
                    const struct gpu_parse_chunk* __restrict__ d_chunks,
                    const struct sample_plan* __restrict__ d_sample_plans,
                    const void** __restrict__ zstd_comp_ptrs,
                    size_t* __restrict__ zstd_comp_sizes,
                    void** __restrict__ zstd_decomp_ptrs,
                    size_t* __restrict__ zstd_decomp_buf_sizes,
                    struct gpu_memcpy_op* __restrict__ d_memcpy_ops,
                    struct assemble_chunk* __restrict__ d_assemble_chunks,
                    uint32_t* __restrict__ d_n_zstd,
                    uint32_t* __restrict__ d_n_memcpy,
                    uint32_t* __restrict__ d_parse_err,
                    uint32_t n_chunks)
{
  const uint32_t chunk_idx = blockIdx.x;
  if (chunk_idx >= n_chunks)
    return;

  const struct gpu_parse_chunk chunk = d_chunks[chunk_idx];
  const uint32_t tid = threadIdx.x;

  // FILL: nothing to do on the codec stage; assemble broadcasts.
  if (chunk.is_fill || chunk.codec_id == (uint8_t)CODEC_FILL) {
    if (tid == 0) {
      struct assemble_chunk* a = &d_assemble_chunks[chunk_idx];
      a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_NONE;
      a->shuffle_typesize = 0;
      a->shuffle_blocksize = 0;
    }
    return;
  }

  uint8_t* d_comp =
    const_cast<uint8_t*>(d_compressed) + chunk.compressed_offset;
  uint8_t* d_decomp = d_decompressed + chunk.decompressed_offset;

  // CODEC_NONE: whole chunk is a single raw memcpy.
  if (chunk.codec_id == (uint8_t)CODEC_NONE) {
    if (tid == 0) {
      uint32_t slot = atomicAdd(d_n_memcpy, 1u);
      struct gpu_memcpy_op op;
      op.d_src = d_comp;
      op.d_dst = d_decomp;
      op.nbytes = chunk.decompressed_nbytes;
      d_memcpy_ops[slot] = op;
      struct assemble_chunk* a = &d_assemble_chunks[chunk_idx];
      a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_NONE;
      a->shuffle_typesize = 0;
      a->shuffle_blocksize = 0;
    }
    return;
  }

  // CODEC_ZSTD: whole chunk is a single zstd substream.
  if (chunk.codec_id == (uint8_t)CODEC_ZSTD) {
    if (tid == 0) {
      uint32_t slot = atomicAdd(d_n_zstd, 1u);
      zstd_comp_ptrs[slot] = d_comp;
      zstd_comp_sizes[slot] = chunk.compressed_nbytes;
      zstd_decomp_ptrs[slot] = d_decomp;
      zstd_decomp_buf_sizes[slot] = chunk.decompressed_nbytes;
      struct assemble_chunk* a = &d_assemble_chunks[chunk_idx];
      a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_NONE;
      a->shuffle_typesize = 0;
      a->shuffle_blocksize = 0;
    }
    return;
  }

  // From here: CODEC_BLOSC_ZSTD. Validate basic shape using the
  // probed sample-level layout when available, then walk per-block
  // 4B prefixes.
  if (chunk.codec_id != (uint8_t)CODEC_BLOSC_ZSTD) {
    if (tid == 0)
      record_parse_err(d_parse_err, 8u); // unsupported codec_id
    return;
  }
  if (chunk.compressed_nbytes < kBloscHeaderBytes) {
    if (tid == 0)
      record_parse_err(d_parse_err, 1u);
    return;
  }

  __shared__ uint8_t s_header[kBloscHeaderBytes];
  __shared__ uint32_t s_nblocks;
  __shared__ uint32_t s_blocksize;
  __shared__ uint8_t s_typesize;
  __shared__ uint8_t s_shuffle;
  __shared__ uint8_t s_bitshuffle;
  __shared__ uint8_t s_memcpyed;

  if (tid < kBloscHeaderBytes)
    s_header[tid] = d_comp[tid];
  __syncthreads();

  if (tid == 0) {
    const uint8_t flags = s_header[2];
    s_typesize = s_header[3];
    s_shuffle = (uint8_t)(flags & 0x01u);
    s_memcpyed = (uint8_t)((flags >> 1) & 0x01u);
    s_bitshuffle = (uint8_t)((flags >> 2) & 0x01u);
    const uint8_t compformat = (uint8_t)((flags >> 5) & 0x07u);
    const uint32_t nbytes = read_u32_le(s_header + 4);
    s_blocksize = read_u32_le(s_header + 8);
    const uint32_t cbytes = read_u32_le(s_header + 12);
    uint32_t err = 0;
    if (s_blocksize == 0)
      err = 2;
    else if (nbytes != chunk.decompressed_nbytes)
      err = 3;
    else if (cbytes != chunk.compressed_nbytes)
      err = 4;
    else if (s_typesize == 0 || s_typesize > 8u)
      err = 6;
    else if (compformat != 4u)
      err = 7; // blosc1-zstd compformat == 4
    s_nblocks = (s_blocksize == 0)
                  ? 0u
                  : (nbytes / s_blocksize + (nbytes % s_blocksize != 0u));
    if (err == 0 && s_nblocks > DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK)
      err = 5;
    if (err != 0) {
      record_parse_err(d_parse_err, err);
      s_nblocks = 0u;
    }
  }
  __syncthreads();

  if (s_nblocks == 0u)
    return;

  // memcpyed flag: whole chunk is raw bytes after a 16-byte header +
  // 4*nblocks bstarts. Emit one memcpy op pointing past the overhead.
  if (s_memcpyed) {
    if (tid == 0) {
      uint32_t overhead = kBloscHeaderBytes + 4u * s_nblocks;
      uint32_t slot = atomicAdd(d_n_memcpy, 1u);
      struct gpu_memcpy_op op;
      op.d_src = d_comp + overhead;
      op.d_dst = d_decomp;
      op.nbytes = chunk.decompressed_nbytes;
      d_memcpy_ops[slot] = op;
      struct assemble_chunk* a = &d_assemble_chunks[chunk_idx];
      a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_NONE;
      a->shuffle_typesize = 0;
      a->shuffle_blocksize = 0;
    }
    return;
  }

  // Load bstarts[i] and walk per-block prefix.
  const uint32_t payload_lo = kBloscHeaderBytes + 4u * s_nblocks;
  uint32_t bs = 0;
  uint32_t cb = 0;
  uint32_t cur = 0;
  bool is_codec = false;
  bool is_raw = false;
  if (tid < s_nblocks) {
    bs = read_u32_le(d_comp + kBloscHeaderBytes + 4u * tid);
    if (bs < payload_lo || bs >= chunk.compressed_nbytes) {
      record_parse_err(d_parse_err, 9u);
      bs = 0;
    } else {
      // Coarse bounds: cb prefix sits at [bs, bs+4); cb itself is the
      // substream payload length. Tight per-block end requires
      // sort-by-bs (host parser does it); for zstd-only we use the
      // chunk's compressed_nbytes as an upper bound on the substream
      // extent. Pathological writers could pass this check but not the
      // tight one — flagged in the plan as a known limitation.
      if (bs + 4u > chunk.compressed_nbytes) {
        record_parse_err(d_parse_err, 11u);
      } else {
        cb = read_u32_le(d_comp + bs);
        cur = bs + 4u;
        if (cb > chunk.compressed_nbytes - cur) {
          record_parse_err(d_parse_err, 11u);
        } else if (cb == s_blocksize) {
          is_raw = true;
        } else {
          is_codec = true;
        }
      }
    }
  }
  __syncthreads();

  // Warp-wide compaction: each thread (within the warp covering nblocks)
  // either contributes a zstd or memcpy slot, claimed via per-warp
  // atomicAdd of the popcount.
  const uint32_t mask_codec = __ballot_sync(0xffffffffu, is_codec);
  const uint32_t mask_raw = __ballot_sync(0xffffffffu, is_raw);

  __shared__ uint32_t s_codec_base;
  __shared__ uint32_t s_raw_base;
  if (tid == 0) {
    s_codec_base =
      (mask_codec != 0u) ? atomicAdd(d_n_zstd, __popc(mask_codec)) : 0u;
    s_raw_base =
      (mask_raw != 0u) ? atomicAdd(d_n_memcpy, __popc(mask_raw)) : 0u;
  }
  __syncthreads();

  if (is_codec) {
    const uint32_t rank_within = __popc(mask_codec & ((1u << tid) - 1u));
    const uint32_t slot = s_codec_base + rank_within;
    const uint8_t* src = d_comp + cur;
    uint8_t* dst = d_decomp + tid * s_blocksize;
    zstd_comp_ptrs[slot] = src;
    zstd_comp_sizes[slot] = cb;
    zstd_decomp_ptrs[slot] = dst;
    zstd_decomp_buf_sizes[slot] = s_blocksize;
  }
  if (is_raw) {
    const uint32_t rank_within = __popc(mask_raw & ((1u << tid) - 1u));
    const uint32_t slot = s_raw_base + rank_within;
    struct gpu_memcpy_op op;
    op.d_src = d_comp + cur;
    op.d_dst = d_decomp + tid * s_blocksize;
    op.nbytes = s_blocksize;
    d_memcpy_ops[slot] = op;
  }

  if (tid == 0) {
    struct assemble_chunk* a = &d_assemble_chunks[chunk_idx];
    if (s_shuffle) {
      a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_BYTE;
      a->shuffle_typesize = s_typesize;
      a->shuffle_blocksize = s_blocksize;
    } else if (s_bitshuffle) {
      a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_BIT;
      a->shuffle_typesize = s_typesize;
      a->shuffle_blocksize = s_blocksize;
    } else {
      a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_NONE;
      a->shuffle_typesize = 0;
      a->shuffle_blocksize = 0;
    }
  }
  // Suppress unused warning for sample_plans; reserved for a future
  // layout-mismatch assertion against the probed per-array layout.
  (void)d_sample_plans;
}

} // namespace

extern "C" int
blosc1_parse_launch(CUstream stream, const struct blosc1_parse_args* args)
{
  if (!args || args->n_chunks == 0)
    return 0;
  blosc1_parse_kernel<<<args->n_chunks,
                        kThreadsPerBlock,
                        0,
                        (cudaStream_t)stream>>>(args->d_compressed,
                                                args->d_decompressed,
                                                args->d_chunks,
                                                args->d_sample_plans,
                                                args->zstd.d_comp_ptrs,
                                                args->zstd.d_comp_sizes,
                                                args->zstd.d_decomp_ptrs,
                                                args->zstd.d_decomp_buf_sizes,
                                                args->d_memcpy_ops,
                                                args->d_assemble_chunks,
                                                args->d_n_zstd,
                                                args->d_n_memcpy,
                                                args->d_parse_err,
                                                args->n_chunks);
  return decoder_launch_status_check("blosc1_parse_launch");
}
