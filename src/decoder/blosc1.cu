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

// Single-block exclusive-scan over up to MAX_CHUNKS_PER_WAVE entries.
// Five parallel scans (zstd, lz4, memcpy, unshuffle, bitunshuffle) each
// run in a separate warp-strided pass through shared memory.
__global__ void
blosc1_scan_offsets_kernel(
  const struct blosc1_chunk_counts* __restrict__ counts,
  struct blosc1_chunk_offsets* __restrict__ offsets,
  struct blosc1_totals* __restrict__ totals,
  uint32_t n_chunks)
{
  if (threadIdx.x != 0)
    return;

  uint32_t z = 0, l = 0, m = 0, u = 0, b = 0;
  for (uint32_t i = 0; i < n_chunks; ++i) {
    const struct blosc1_chunk_counts c = counts[i];
    struct blosc1_chunk_offsets o;
    o.zstd_off = z;
    o.lz4_off = l;
    o.memcpy_off = m;
    o.unshuffle_off = u;
    o.bitunshuffle_off = b;
    offsets[i] = o;
    z += c.n_zstd;
    l += c.n_lz4;
    m += c.n_memcpy;
    u += c.has_unshuffle;
    b += c.has_bitunshuffle;
  }
  totals->n_zstd = z;
  totals->n_lz4 = l;
  totals->n_memcpy = m;
  totals->n_unshuffle = u;
  totals->n_bitunshuffle = b;
}

// One CUDA block per chunk; blockDim.x = DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK.
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

  const struct blosc1_chunk_input chunk = in[i];
  const struct blosc1_chunk_hdr h = hdrs[i];
  const struct blosc1_chunk_offsets o = offsets[i];

  if (h.err != 0)
    return;

  uint8_t* d_decomp = (uint8_t*)chunk.d_decompressed;
  const uint8_t* d_comp = (const uint8_t*)chunk.d_compressed;

  if (chunk.codec_id == CODEC_NONE) {
    if (threadIdx.x == 0) {
      struct gpu_memcpy_op op;
      op.d_src = d_comp;
      op.d_dst = d_decomp;
      op.nbytes = chunk.decompressed_nbytes;
      memcpy_ops[o.memcpy_off] = op;
    }
    return;
  }
  if (chunk.codec_id == CODEC_ZSTD) {
    if (threadIdx.x == 0) {
      struct gpu_substream s;
      s.d_src = d_comp;
      s.d_dst = d_decomp;
      s.src_nbytes = chunk.compressed_nbytes;
      s.dst_nbytes = chunk.decompressed_nbytes;
      zstd_subs[o.zstd_off] = s;
    }
    return;
  }

  // CODEC_BLOSC_*
  if (h.memcpyed) {
    if (threadIdx.x == 0) {
      const uint32_t overhead = kHeaderBytes + 4u * h.nblocks;
      struct gpu_memcpy_op op;
      op.d_src = d_comp + overhead;
      op.d_dst = d_decomp;
      op.nbytes = chunk.decompressed_nbytes;
      memcpy_ops[o.memcpy_off] = op;
    }
    return;
  }

  // bstarts is in writer-thread completion order; each block's
  // compressed-payload extent is bounded by the next-higher offset
  // (or cbytes for the trailing block). Compute each thread's rank
  // among bstarts; sorted_offsets[rank] gives sorted ascending list.
  __shared__ uint32_t bstarts[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK];
  __shared__ uint32_t sorted_offsets[DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK];
  const uint32_t t = threadIdx.x;
  if (t < h.nblocks)
    bstarts[t] = read_u32_le(d_comp + kHeaderBytes + 4u * t);
  __syncthreads();

  uint32_t my_off = 0;
  uint32_t my_end = 0;
  uint32_t my_rank = 0;
  if (t < h.nblocks) {
    my_off = bstarts[t];
    for (uint32_t j = 0; j < h.nblocks; ++j) {
      const uint32_t other = bstarts[j];
      if (other < my_off || (other == my_off && j < t))
        ++my_rank;
    }
    sorted_offsets[my_rank] = my_off;
  }
  __syncthreads();
  if (t < h.nblocks) {
    my_end =
      (my_rank + 1u < h.nblocks) ? sorted_offsets[my_rank + 1u] : h.cbytes;
  }

  if (t >= h.nblocks)
    return;

  const uint32_t block_start = my_off;
  const uint32_t block_end = my_end;
  const uint32_t block_uncompressed = h.blocksize;
  const uint32_t block_dst_off = t * h.blocksize;

  uint32_t nstreams_per_block = 1;
  if (chunk.codec_id == CODEC_BLOSC_LZ4 && h.typesize > 1)
    nstreams_per_block = h.typesize;
  const uint32_t per_stream_dst = (nstreams_per_block == 1)
                                    ? block_uncompressed
                                    : (block_uncompressed / h.typesize);

  // Walk int32-prefixed sub-streams within [block_start, block_end).
  uint32_t cur = block_start;
  uint32_t emitted = 0;
  uint32_t fanout_base =
    (chunk.codec_id == CODEC_BLOSC_LZ4) ? o.lz4_off : o.zstd_off;
  // Each block writes its substreams contiguously in the chunk's slice;
  // within the chunk, block t starts at fanout_base + t * nstreams_per_block.
  fanout_base += t * nstreams_per_block;

  while (cur + 4u <= block_end && emitted < nstreams_per_block) {
    const uint32_t cb = read_u32_le(d_comp + cur);
    cur += 4u;
    if (cur + cb > block_end)
      break;
    struct gpu_substream s;
    s.d_src = d_comp + cur;
    s.d_dst = d_decomp + block_dst_off + emitted * per_stream_dst;
    s.src_nbytes = cb;
    s.dst_nbytes = per_stream_dst;
    if (chunk.codec_id == CODEC_BLOSC_LZ4)
      lz4_subs[fanout_base + emitted] = s;
    else
      zstd_subs[fanout_base + emitted] = s;
    cur += cb;
    ++emitted;
  }

  if (t == 0) {
    if (h.shuffle) {
      struct gpu_shuffle_op u;
      u.d_buf = d_decomp;
      u.blocksize = h.blocksize;
      u.typesize = h.typesize;
      u.nblocks_full = h.nblocks;
      u.tail_nbytes = 0;
      unshuffle_ops[o.unshuffle_off] = u;
    }
    if (h.bitshuffle) {
      struct gpu_shuffle_op u;
      u.d_buf = d_decomp;
      u.blocksize = h.blocksize;
      u.typesize = h.typesize;
      u.nblocks_full = h.nblocks;
      u.tail_nbytes = 0;
      bitunshuffle_ops[o.bitunshuffle_off] = u;
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
  blosc1_scan_offsets_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(
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
                              DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK,
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
