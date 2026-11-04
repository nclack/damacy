#include "decoder/blosc1_parse.h"

#include "assemble/assemble.h"
#include "decoder/launch_check.h"
#include "executor/dispatch.h"
#include "zarr/zarr_metadata.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr uint32_t kHeaderBytes = 16u;
constexpr uint32_t kThreads = 128u;

__device__ uint32_t
read_u32_le(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

__device__ void
skip_chunk(const blosc1_parse_args& args, uint32_t chunk, uint32_t error)
{
  atomicOr(&args.d_skip_blocks[chunk >> 5], 1u << (chunk & 31u));
  if (error)
    atomicCAS(args.d_parse_err, 0u, error);
}

__global__ void
blosc1_chunk_scan_kernel(blosc1_parse_args args)
{
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= args.n_blosc_chunks)
    return;

  const uint32_t index = args.d_blosc_chunk_indices[tid];
  const gpu_parse_chunk chunk = args.d_chunks[index];
  if (chunk.sample_idx_in_batch >= args.n_sample_plans) {
    skip_chunk(args, index, 12u);
    return;
  }
  const sample_plan& sample = args.d_sample_plans[chunk.sample_idx_in_batch];
  const uint8_t* source = args.d_compressed + chunk.compressed_offset;
  if (chunk.compressed_nbytes < kHeaderBytes) {
    skip_chunk(args, index, 1u);
    return;
  }
  const uint8_t flags = source[2];
  const uint32_t typesize = source[3];
  const uint32_t nbytes = read_u32_le(source + 4);
  const uint32_t blocksize = read_u32_le(source + 8);
  if (source[0] < 1 || source[0] > 2 ||
      read_u32_le(source + 12) != chunk.compressed_nbytes ||
      nbytes != chunk.decompressed_nbytes || !nbytes || !blocksize ||
      blocksize > nbytes || !typesize || typesize != sample.layout.typesize ||
      nbytes % typesize || blocksize % typesize) {
    skip_chunk(args, index, 4u);
    return;
  }

  assemble_chunk& output = args.d_assemble_chunks[index];
  if (flags & 2u) {
    skip_chunk(args, index, 0u);
    if (chunk.compressed_nbytes - kHeaderBytes != nbytes) {
      skip_chunk(args, index, 4u);
      return;
    }
    const uint32_t slot = atomicAdd(args.d_n_memcpy, 1u);
    if (slot >= args.fanout_cap) {
      skip_chunk(args, index, 5u);
      return;
    }
    args.fanout.d_memcpy_ops[slot] = {
      source + kHeaderBytes,
      args.d_decompressed + chunk.decompressed_offset,
      nbytes,
    };
    output.shuffle_mode = ASSEMBLE_SHUFFLE_NONE;
    output.shuffle_typesize = 0;
    output.shuffle_blocksize = 0;
    return;
  }

  const uint8_t format = chunk.codec_id == CODEC_BLOSC_LZ4 ? 1u : 4u;
  const uint32_t nblocks = nbytes / blocksize + (nbytes % blocksize != 0);
  const uint32_t splits =
    !(flags & 0x10u) && blocksize / typesize >= 128u ? typesize : 1u;
  const uint32_t substreams =
    (nbytes / blocksize) * splits + (nbytes % blocksize != 0);
  if (source[1] != 1 || (flags >> 5) != format || (flags & 8u) ||
      blocksize != sample.layout.blocksize ||
      nblocks != sample.layout.nblocks ||
      substreams > sample.layout.nsubstreams ||
      nblocks > (chunk.compressed_nbytes - kHeaderBytes) / 4u) {
    skip_chunk(args, index, 7u);
    return;
  }
  output.shuffle_typesize = (uint8_t)typesize;
  output.shuffle_blocksize = blocksize;
  output.shuffle_mode = (flags & 1u) && typesize > 1u ? ASSEMBLE_SHUFFLE_BYTE
                        : (flags & 4u)                ? ASSEMBLE_SHUFFLE_BIT
                                                      : ASSEMBLE_SHUFFLE_NONE;
}

__global__ void
blosc1_block_fanout_kernel(blosc1_parse_args args)
{
  __shared__ uint32_t zstd_base[kThreads / 32u];
  __shared__ uint32_t lz4_base[kThreads / 32u];
  __shared__ uint32_t raw_base[kThreads / 32u];
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;

  const uint8_t* source = nullptr;
  uint8_t* destination = nullptr;
  uint32_t cbytes = 0, cursor = 0, stream_bytes = 0, splits = 0;
  bool lz4 = false;
  if (tid < args.n_blosc_blocks) {
    const uint32_t packed = args.d_block_chunk_map[tid];
    const uint32_t index = packed >> 16;
    const uint32_t block = packed & 0xFFFFu;
    if (!(args.d_skip_blocks[index >> 5] & (1u << (index & 31u)))) {
      const gpu_parse_chunk chunk = args.d_chunks[index];
      const chunk_layout& layout =
        args.d_sample_plans[chunk.sample_idx_in_batch].layout;
      source = args.d_compressed + chunk.compressed_offset;
      cbytes = chunk.compressed_nbytes;
      const uint32_t offset = block * layout.blocksize;
      const uint32_t block_bytes =
        min(layout.blocksize, layout.nbytes - offset);
      splits = !(source[2] & 0x10u) && block_bytes == layout.blocksize &&
                   block_bytes / layout.typesize >= 128u
                 ? layout.typesize
                 : 1u;
      stream_bytes = block_bytes / splits;
      destination = args.d_decompressed + chunk.decompressed_offset + offset;
      lz4 = chunk.codec_id == CODEC_BLOSC_LZ4;
      cursor = read_u32_le(source + kHeaderBytes + 4u * block);
      if (cursor < kHeaderBytes + 4u * layout.nblocks || cursor > cbytes - 4u) {
        atomicCAS(args.d_parse_err, 0u, 9u);
        splits = 0;
      }
    }
  }

  for (uint32_t split = 0; __any_sync(0xFFFFFFFFu, split < splits); ++split) {
    uint32_t size = 0;
    bool valid = split < splits;
    if (valid) {
      if (cursor > cbytes - 4u) {
        valid = false;
      } else {
        size = read_u32_le(source + cursor);
        cursor += 4u;
        valid = size && size <= stream_bytes && size <= cbytes - cursor;
      }
      if (!valid) {
        atomicCAS(args.d_parse_err, 0u, 11u);
        splits = 0;
      }
    }
    const bool raw = valid && size == stream_bytes;
    const bool compressed = valid && !raw;
    const uint32_t zstd_mask = __ballot_sync(0xFFFFFFFFu, compressed && !lz4);
    const uint32_t lz4_mask = __ballot_sync(0xFFFFFFFFu, compressed && lz4);
    const uint32_t raw_mask = __ballot_sync(0xFFFFFFFFu, raw);
    if (lane == 0) {
      zstd_base[warp] =
        zstd_mask ? atomicAdd(args.d_n_zstd, __popc(zstd_mask)) : 0u;
      lz4_base[warp] =
        lz4_mask ? atomicAdd(args.d_n_lz4, __popc(lz4_mask)) : 0u;
      raw_base[warp] =
        raw_mask ? atomicAdd(args.d_n_memcpy, __popc(raw_mask)) : 0u;
    }
    __syncwarp();

    if (compressed) {
      const uint32_t mask = lz4 ? lz4_mask : zstd_mask;
      const uint32_t rank = __popc(mask & ((1u << lane) - 1u));
      const uint32_t count = (lz4 ? lz4_base[warp] : zstd_base[warp]) + rank;
      if (count >= args.fanout_cap) {
        atomicCAS(args.d_parse_err, 0u, 5u);
      } else {
        // Zstd fills from the front and LZ4 from the back of shared storage.
        const uint32_t slot = lz4 ? args.fanout_cap - count - 1u : count;
        args.fanout.d_comp_ptrs[slot] = source + cursor;
        args.fanout.d_comp_sizes[slot] = size;
        args.fanout.d_decomp_ptrs[slot] = destination + split * stream_bytes;
        args.fanout.d_decomp_buf_sizes[slot] = stream_bytes;
      }
    }
    if (raw) {
      const uint32_t slot =
        raw_base[warp] + __popc(raw_mask & ((1u << lane) - 1u));
      if (slot >= args.fanout_cap) {
        atomicCAS(args.d_parse_err, 0u, 5u);
      } else {
        args.fanout.d_memcpy_ops[slot] = {
          source + cursor,
          destination + split * stream_bytes,
          stream_bytes,
        };
      }
    }
    cursor += size;
    __syncwarp();
  }
}

}

extern "C" int
blosc1_parse_launch(CUstream stream, const blosc1_parse_args* args)
{
  if (!args || !args->n_blosc_chunks)
    return 0;
  blosc1_chunk_scan_kernel<<<(args->n_blosc_chunks + kThreads - 1) / kThreads,
                             kThreads,
                             0,
                             (cudaStream_t)stream>>>(*args);
  int status = decoder_launch_status_check("blosc1_chunk_scan_kernel");
  if (status || !args->n_blosc_blocks)
    return status;
  blosc1_block_fanout_kernel<<<(args->n_blosc_blocks + kThreads - 1) / kThreads,
                               kThreads,
                               0,
                               (cudaStream_t)stream>>>(*args);
  return decoder_launch_status_check("blosc1_block_fanout_kernel");
}
