#include "decoder/blosc1.h"

#include "damacy_limits.h"
#include "log/log.h"
#include "zarr/zarr_metadata.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

// blosc1 chunk header: 16 bytes at offset 0; bstarts[nblocks] follows.
constexpr uint32_t kHeaderBytes = 16;

__device__ uint32_t
read_u32_le(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

__device__ uint8_t
inner_codec_compformat(uint8_t codec)
{
  if (codec == CODEC_BLOSC_LZ4)
    return 1;
  if (codec == CODEC_BLOSC_ZSTD)
    return 4;
  return 0;
}

__global__ void
blosc1_parse_and_count_kernel(const struct blosc1_chunk_input* __restrict__ in,
                              struct blosc1_chunk_hdr* __restrict__ hdrs,
                              struct blosc1_chunk_counts* __restrict__ counts,
                              uint32_t n_chunks)
{
  const uint32_t i = blockIdx.x;
  if (i >= n_chunks || threadIdx.x != 0)
    return;

  const struct blosc1_chunk_input chunk = in[i];
  struct blosc1_chunk_hdr h = {};
  struct blosc1_chunk_counts c = {};
  h.codec_id = chunk.codec_id;

  if (chunk.codec_id == CODEC_NONE) {
    c.n_memcpy = 1;
  } else if (chunk.codec_id == CODEC_ZSTD) {
    c.n_zstd = 1;
  } else if (chunk.codec_id == CODEC_BLOSC_LZ4 ||
             chunk.codec_id == CODEC_BLOSC_ZSTD) {
    if (chunk.compressed_nbytes < kHeaderBytes) {
      h.err = 1;
      goto done;
    }
    const uint8_t* p = (const uint8_t*)chunk.d_compressed;
    const uint8_t flags = p[2];
    h.typesize = p[3];
    h.shuffle = flags & 0x01u;
    h.memcpyed = (flags >> 1) & 0x01u;
    h.bitshuffle = (flags >> 2) & 0x01u;
    h.compformat = (flags >> 5) & 0x07u;
    h.nbytes = read_u32_le(p + 4);
    h.blocksize = read_u32_le(p + 8);
    h.cbytes = read_u32_le(p + 12);
    if (h.blocksize == 0) {
      h.err = 2;
      goto done;
    }
    h.nblocks = (h.nbytes + h.blocksize - 1) / h.blocksize;

    if (h.nbytes != chunk.decompressed_nbytes) {
      h.err = 3;
      goto done;
    }
    if (h.cbytes != chunk.compressed_nbytes) {
      h.err = 4;
      goto done;
    }
    if (h.nblocks > DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK) {
      h.err = 5;
      goto done;
    }
    if (h.typesize == 0 || h.typesize > DAMACY_BLOSC_MAX_TYPESIZE) {
      h.err = 6;
      goto done;
    }
    if (h.compformat != inner_codec_compformat(chunk.codec_id)) {
      h.err = 7;
      goto done;
    }

    if (h.memcpyed) {
      c.n_memcpy = 1;
    } else {
      uint32_t nstreams_per_block = 1;
      if (chunk.codec_id == CODEC_BLOSC_LZ4 && h.typesize > 1)
        nstreams_per_block = h.typesize;
      const uint32_t total = h.nblocks * nstreams_per_block;
      if (chunk.codec_id == CODEC_BLOSC_LZ4)
        c.n_lz4 = total;
      else
        c.n_zstd = total;
      c.has_unshuffle = h.shuffle;
      c.has_bitunshuffle = h.bitshuffle;
    }
  } else {
    h.err = 8;
  }

done:
  hdrs[i] = h;
  counts[i] = c;
}

// Block size of blosc1_scan_offsets_kernel. Equals the max wave size
// (DAMACY_MAX_CHUNKS_PER_WAVE in damacy.c). Hard-coded here to avoid
// a damacy.h dependency from the kernel module; the launcher rejects
// n_chunks > kScanBlockSize.
constexpr uint32_t kScanBlockSize = 512;
constexpr uint32_t kScanWarps = kScanBlockSize / 32u;

__device__ uint32_t
warp_exclusive_scan_u32(uint32_t v, uint32_t lane)
{
  uint32_t x = v;
#pragma unroll
  for (int i = 1; i < 32; i <<= 1) {
    const uint32_t up = __shfl_up_sync(0xffffffffu, x, (uint32_t)i);
    if (lane >= (uint32_t)i)
      x += up;
  }
  return x - v;
}

__global__ void
blosc1_scan_offsets_kernel(
  const struct blosc1_chunk_counts* __restrict__ counts,
  struct blosc1_chunk_offsets* __restrict__ offsets,
  struct blosc1_totals* __restrict__ totals,
  uint32_t n_chunks)
{
  __shared__ uint32_t warp_z[kScanWarps];
  __shared__ uint32_t warp_l[kScanWarps];
  __shared__ uint32_t warp_m[kScanWarps];
  __shared__ uint32_t warp_u[kScanWarps];
  __shared__ uint32_t warp_b[kScanWarps];

  const uint32_t tid = threadIdx.x;
  const uint32_t lane = tid & 31u;
  const uint32_t warp = tid >> 5u;

  struct blosc1_chunk_counts c = { 0, 0, 0, 0, 0 };
  if (tid < n_chunks)
    c = counts[tid];

  uint32_t z_ex = warp_exclusive_scan_u32(c.n_zstd, lane);
  uint32_t l_ex = warp_exclusive_scan_u32(c.n_lz4, lane);
  uint32_t m_ex = warp_exclusive_scan_u32(c.n_memcpy, lane);
  uint32_t u_ex = warp_exclusive_scan_u32(c.has_unshuffle, lane);
  uint32_t b_ex = warp_exclusive_scan_u32(c.has_bitunshuffle, lane);

  if (lane == 31u) {
    warp_z[warp] = z_ex + c.n_zstd;
    warp_l[warp] = l_ex + c.n_lz4;
    warp_m[warp] = m_ex + c.n_memcpy;
    warp_u[warp] = u_ex + c.has_unshuffle;
    warp_b[warp] = b_ex + c.has_bitunshuffle;
  }
  __syncthreads();

  if (warp == 0u) {
    const uint32_t in_range = lane < kScanWarps ? 1u : 0u;
    uint32_t z = in_range ? warp_z[lane] : 0u;
    uint32_t l = in_range ? warp_l[lane] : 0u;
    uint32_t m = in_range ? warp_m[lane] : 0u;
    uint32_t u = in_range ? warp_u[lane] : 0u;
    uint32_t b = in_range ? warp_b[lane] : 0u;
    z = warp_exclusive_scan_u32(z, lane);
    l = warp_exclusive_scan_u32(l, lane);
    m = warp_exclusive_scan_u32(m, lane);
    u = warp_exclusive_scan_u32(u, lane);
    b = warp_exclusive_scan_u32(b, lane);
    if (in_range) {
      warp_z[lane] = z;
      warp_l[lane] = l;
      warp_m[lane] = m;
      warp_u[lane] = u;
      warp_b[lane] = b;
    }
  }
  __syncthreads();

  z_ex += warp_z[warp];
  l_ex += warp_l[warp];
  m_ex += warp_m[warp];
  u_ex += warp_u[warp];
  b_ex += warp_b[warp];

  if (tid < n_chunks) {
    struct blosc1_chunk_offsets o;
    o.zstd_off = z_ex;
    o.lz4_off = l_ex;
    o.memcpy_off = m_ex;
    o.unshuffle_off = u_ex;
    o.bitunshuffle_off = b_ex;
    offsets[tid] = o;
  }

  if (tid + 1u == n_chunks) {
    totals->n_zstd = z_ex + c.n_zstd;
    totals->n_lz4 = l_ex + c.n_lz4;
    totals->n_memcpy = m_ex + c.n_memcpy;
    totals->n_unshuffle = u_ex + c.has_unshuffle;
    totals->n_bitunshuffle = b_ex + c.has_bitunshuffle;
  }
}

// blockDim.x is a full warp so warp scheduling isn't half-utilised.
// Active lanes are 0 .. h.nblocks-1.
constexpr uint32_t kEmitBlockDim = 32;
static_assert(DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK <= kEmitBlockDim, "");

struct EmitSmem
{
  struct blosc1_chunk_input chunk;
  struct blosc1_chunk_hdr hdr;
  struct blosc1_chunk_offsets offsets;
  uint32_t bstarts[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK];
  uint32_t sorted_offsets[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK];
};

__global__ void
blosc1_emit_fanout_kernel(
  const struct blosc1_chunk_input* __restrict__ in,
  const struct blosc1_chunk_hdr* __restrict__ hdrs,
  const struct blosc1_chunk_offsets* __restrict__ offsets,
  struct gpu_substream* __restrict__ zstd_subs,
  struct gpu_substream* __restrict__ lz4_subs,
  struct gpu_memcpy_op* __restrict__ memcpy_ops,
  struct gpu_shuffle_op* __restrict__ unshuffle_ops,
  struct gpu_shuffle_op* __restrict__ bitunshuffle_ops,
  uint32_t n_chunks)
{
  const uint32_t i = blockIdx.x;
  if (i >= n_chunks)
    return;

  // Per-chunk metadata: load once per block into shared memory rather
  // than into every thread's registers.
  __shared__ EmitSmem s;
  if (threadIdx.x == 0) {
    s.chunk = in[i];
    s.hdr = hdrs[i];
    s.offsets = offsets[i];
  }
  __syncthreads();

  if (s.hdr.err != 0)
    return;

  const uint8_t codec = s.chunk.codec_id;
  uint8_t* d_decomp = (uint8_t*)s.chunk.d_decompressed;
  const uint8_t* d_comp = (const uint8_t*)s.chunk.d_compressed;

  if (codec == CODEC_NONE) {
    if (threadIdx.x == 0) {
      struct gpu_memcpy_op* slot = &memcpy_ops[s.offsets.memcpy_off];
      slot->d_src = d_comp;
      slot->d_dst = d_decomp;
      slot->nbytes = s.chunk.decompressed_nbytes;
    }
    return;
  }
  if (codec == CODEC_ZSTD) {
    if (threadIdx.x == 0) {
      struct gpu_substream* slot = &zstd_subs[s.offsets.zstd_off];
      slot->d_src = d_comp;
      slot->d_dst = d_decomp;
      slot->src_nbytes = s.chunk.compressed_nbytes;
      slot->dst_nbytes = s.chunk.decompressed_nbytes;
    }
    return;
  }

  // CODEC_BLOSC_*
  if (s.hdr.memcpyed) {
    if (threadIdx.x == 0) {
      const uint32_t overhead = kHeaderBytes + 4u * s.hdr.nblocks;
      struct gpu_memcpy_op* slot = &memcpy_ops[s.offsets.memcpy_off];
      slot->d_src = d_comp + overhead;
      slot->d_dst = d_decomp;
      slot->nbytes = s.chunk.decompressed_nbytes;
    }
    return;
  }

  // bstarts is in writer-thread completion order; rank-sort to derive
  // each block's compressed-payload extent.
  const uint32_t t = threadIdx.x;
  const uint32_t nblocks = s.hdr.nblocks;
  if (t < nblocks)
    s.bstarts[t] = read_u32_le(d_comp + kHeaderBytes + 4u * t);
  __syncthreads();

  uint32_t my_off = 0;
  uint32_t my_rank = 0;
  if (t < nblocks) {
    my_off = s.bstarts[t];
    for (uint32_t j = 0; j < nblocks; ++j) {
      const uint32_t other = s.bstarts[j];
      if (other < my_off || (other == my_off && j < t))
        ++my_rank;
    }
    s.sorted_offsets[my_rank] = my_off;
  }
  __syncthreads();

  if (t >= nblocks)
    return;

  const uint32_t block_end =
    (my_rank + 1u < nblocks) ? s.sorted_offsets[my_rank + 1u] : s.hdr.cbytes;
  const uint32_t nstreams = (codec == CODEC_BLOSC_LZ4 && s.hdr.typesize > 1)
                              ? (uint32_t)s.hdr.typesize
                              : 1u;
  const uint32_t per_stream_dst =
    (nstreams == 1u) ? s.hdr.blocksize : (s.hdr.blocksize / s.hdr.typesize);
  const uint32_t block_dst_off = t * s.hdr.blocksize;
  struct gpu_substream* dst_table =
    (codec == CODEC_BLOSC_LZ4) ? lz4_subs : zstd_subs;
  const uint32_t fanout_base =
    (codec == CODEC_BLOSC_LZ4 ? s.offsets.lz4_off : s.offsets.zstd_off) +
    t * nstreams;

  // Walk int32-prefixed sub-streams within [my_off, block_end). Stream
  // each descriptor straight to global without holding the temporary in
  // registers.
  uint32_t cur = my_off;
  for (uint32_t k = 0; k < nstreams; ++k) {
    if (cur + 4u > block_end)
      break;
    const uint32_t cb = read_u32_le(d_comp + cur);
    cur += 4u;
    if (cur + cb > block_end)
      break;
    struct gpu_substream* slot = &dst_table[fanout_base + k];
    slot->d_src = d_comp + cur;
    slot->d_dst = d_decomp + block_dst_off + k * per_stream_dst;
    slot->src_nbytes = cb;
    slot->dst_nbytes = per_stream_dst;
    cur += cb;
  }

  if (t == 0 && (s.hdr.shuffle || s.hdr.bitshuffle)) {
    if (s.hdr.shuffle) {
      struct gpu_shuffle_op* slot = &unshuffle_ops[s.offsets.unshuffle_off];
      slot->d_buf = d_decomp;
      slot->blocksize = s.hdr.blocksize;
      slot->typesize = s.hdr.typesize;
      slot->nblocks_full = nblocks;
      slot->tail_nbytes = 0;
    }
    if (s.hdr.bitshuffle) {
      struct gpu_shuffle_op* slot =
        &bitunshuffle_ops[s.offsets.bitunshuffle_off];
      slot->d_buf = d_decomp;
      slot->blocksize = s.hdr.blocksize;
      slot->typesize = s.hdr.typesize;
      slot->nblocks_full = nblocks;
      slot->tail_nbytes = 0;
    }
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
blosc1_parse_and_count_launch(CUstream stream,
                              const struct blosc1_chunk_input* d_inputs,
                              struct blosc1_chunk_hdr* d_hdrs,
                              struct blosc1_chunk_counts* d_counts,
                              uint32_t n_chunks)
{
  if (n_chunks == 0)
    return 0;
  blosc1_parse_and_count_kernel<<<n_chunks, 32, 0, (cudaStream_t)stream>>>(
    d_inputs, d_hdrs, d_counts, n_chunks);
  return launch_status_check("blosc1_parse_and_count_launch") == cudaSuccess
           ? 0
           : 1;
}

extern "C" int
blosc1_scan_offsets_launch(CUstream stream,
                           const struct blosc1_chunk_counts* d_counts,
                           struct blosc1_chunk_offsets* d_offsets,
                           struct blosc1_totals* d_totals,
                           uint32_t n_chunks)
{
  if (n_chunks == 0) {
    cudaMemsetAsync(d_totals, 0, sizeof(*d_totals), (cudaStream_t)stream);
    return 0;
  }
  if (n_chunks > kScanBlockSize) {
    log_error(
      "blosc1_scan_offsets_launch: n_chunks=%u > %u", n_chunks, kScanBlockSize);
    return 1;
  }
  blosc1_scan_offsets_kernel<<<1, kScanBlockSize, 0, (cudaStream_t)stream>>>(
    d_counts, d_offsets, d_totals, n_chunks);
  return launch_status_check("blosc1_scan_offsets_launch") == cudaSuccess ? 0
                                                                          : 1;
}

extern "C" int
blosc1_emit_fanout_launch(CUstream stream,
                          const struct blosc1_chunk_input* d_inputs,
                          const struct blosc1_chunk_hdr* d_hdrs,
                          const struct blosc1_chunk_offsets* d_offsets,
                          struct gpu_substream* d_zstd_subs,
                          struct gpu_substream* d_lz4_subs,
                          struct gpu_memcpy_op* d_memcpy_ops,
                          struct gpu_shuffle_op* d_unshuffle_ops,
                          struct gpu_shuffle_op* d_bitunshuffle_ops,
                          uint32_t n_chunks)
{
  if (n_chunks == 0)
    return 0;
  blosc1_emit_fanout_kernel<<<n_chunks,
                              kEmitBlockDim,
                              0,
                              (cudaStream_t)stream>>>(d_inputs,
                                                      d_hdrs,
                                                      d_offsets,
                                                      d_zstd_subs,
                                                      d_lz4_subs,
                                                      d_memcpy_ops,
                                                      d_unshuffle_ops,
                                                      d_bitunshuffle_ops,
                                                      n_chunks);
  return launch_status_check("blosc1_emit_fanout_launch") == cudaSuccess ? 0
                                                                         : 1;
}
