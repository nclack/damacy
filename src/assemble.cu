#include "assemble.h"

#include "log/log.h"
#include "util/prelude.h"

#include <stdint.h>

namespace {

// Block-tile shape T[d] per rank. Innermost dim gets 32 threads for
// coalesced unit-stride access; one outer dim gets 8 to keep a 256-thread
// block; other outer dims get 1.
//
// rank 1: T = {256}              — entire block on the only dim
// rank 2: T = {8, 32}            — Y=8, X=32
// rank R >= 3: T = {…, 1, 8, 32} — innermost 32, second-innermost 8, rest 1

constexpr uint32_t kBlockSize = 256;

template<int RANK>
__device__ __host__ __forceinline__ uint32_t
tile_T(int d)
{
  if (RANK == 1)
    return kBlockSize;
  if (d == RANK - 1)
    return 32u;
  if (d == RANK - 2)
    return 8u;
  return 1u;
}

// Decode a 1D thread index into per-dim coordinates within the block tile.
// For rank R >= 2 the layout is innermost-fastest: tid % 32 → axis R-1,
// (tid / 32) % 8 → axis R-2, 0 elsewhere. For rank 1 the whole thread
// index maps to the single axis.
template<int RANK>
__device__ __forceinline__ uint32_t
thread_axis(uint32_t tid, int d)
{
  if (RANK == 1)
    return tid;
  if (d == RANK - 1)
    return tid & 31u;
  if (d == RANK - 2)
    return (tid >> 5) & 7u;
  return 0u;
}

__device__ __host__ __forceinline__ uint32_t
ceil_div_u32(uint32_t a, uint32_t b)
{
  return (a + b - 1u) / b;
}

template<int RANK, typename copy_t>
__global__ void
assemble_kernel(const struct sample_plan* __restrict__ d_samples,
                const struct assemble_chunk* __restrict__ d_chunks,
                const uint8_t* __restrict__ arena_base,
                uint8_t* __restrict__ output_base)
{
  const uint32_t chunk_idx = blockIdx.y;
  const struct assemble_chunk c = d_chunks[chunk_idx];
  const struct sample_plan& s = d_samples[c.sample_idx_in_batch];

  // tiles_per_chunk[d] = ceil(S[d] / T[d]); blocks_per_chunk = ∏.
  uint32_t tiles_per_chunk[RANK];
  uint32_t blocks_per_chunk = 1u;
#pragma unroll
  for (int d = 0; d < RANK; ++d) {
    tiles_per_chunk[d] = ceil_div_u32(s.dims[d].chunk_shape, tile_T<RANK>(d));
    blocks_per_chunk *= tiles_per_chunk[d];
  }
  if (blockIdx.x >= blocks_per_chunk)
    return;

  // Decode blockIdx.x into per-dim tile-in-chunk coords (row-major).
  uint32_t tile_in_chunk[RANK];
  uint32_t rem = blockIdx.x;
#pragma unroll
  for (int d = RANK - 1; d >= 0; --d) {
    tile_in_chunk[d] = rem % tiles_per_chunk[d];
    rem /= tiles_per_chunk[d];
  }

  const uint32_t tid = threadIdx.x;

  // Per-thread bounds + offset accumulation. Each axis contributes:
  //   intra      = tile_in_chunk[d] * T[d] + thread_axis<RANK>(tid, d)
  //   src_off   += intra * src_stride[d]
  //   u_d        = chunk_d[d] * S[d] + intra
  //   dst_d      = u_d - aabb_lo_relative[d]
  //   dst_off   += dst_d * dst_stride[d]
  //   bounds AND= (intra < S[d]) AND (0 <= dst_d < aabb_extent[d])
  int64_t src_off_elems = 0;
  int64_t dst_off_elems = s.sample_dst_off_elems;
  bool in_bounds = true;
#pragma unroll
  for (int d = 0; d < RANK; ++d) {
    uint32_t intra =
      tile_in_chunk[d] * tile_T<RANK>(d) + thread_axis<RANK>(tid, d);
    in_bounds = in_bounds && (intra < s.dims[d].chunk_shape);
    int64_t u =
      (int64_t)c.chunk_d[d] * (int64_t)s.dims[d].chunk_shape + (int64_t)intra;
    int64_t dst_d = u - s.dims[d].aabb_lo_relative;
    in_bounds = in_bounds && (dst_d >= 0) && (dst_d < s.dims[d].aabb_extent);
    src_off_elems += (int64_t)intra * s.dims[d].src_stride;
    dst_off_elems += dst_d * s.dims[d].dst_stride;
  }
  if (!in_bounds)
    return;

  const uint8_t* src =
    arena_base + c.src_base_byte_off + src_off_elems * (int64_t)sizeof(copy_t);
  uint8_t* dst = output_base + dst_off_elems * (int64_t)sizeof(copy_t);
  *reinterpret_cast<copy_t*>(dst) = *reinterpret_cast<const copy_t*>(src);
}

template<int RANK>
int
launch_for_bpe(CUstream stream,
               const struct sample_plan* d_samples,
               const struct assemble_chunk* d_chunks,
               uint32_t n_chunks,
               uint32_t max_blocks_per_chunk,
               const uint8_t* arena_base,
               uint8_t* output_base,
               uint32_t bpe)
{
  dim3 grid(max_blocks_per_chunk, n_chunks, 1);
  dim3 block(kBlockSize, 1, 1);
  switch (bpe) {
    case 1:
      assemble_kernel<RANK, uint8_t><<<grid, block, 0, stream>>>(
        d_samples, d_chunks, arena_base, output_base);
      return 0;
    case 2:
      assemble_kernel<RANK, uint16_t><<<grid, block, 0, stream>>>(
        d_samples, d_chunks, arena_base, output_base);
      return 0;
    case 4:
      assemble_kernel<RANK, uint32_t><<<grid, block, 0, stream>>>(
        d_samples, d_chunks, arena_base, output_base);
      return 0;
    case 8:
      assemble_kernel<RANK, uint64_t><<<grid, block, 0, stream>>>(
        d_samples, d_chunks, arena_base, output_base);
      return 0;
    default:
      log_error("assemble: unsupported bpe=%u", bpe);
      return 1;
  }
}

template<int RANK>
uint32_t
blocks_per_chunk_for_rank(const struct sample_dim* dims)
{
  uint32_t bpc = 1u;
  for (int d = 0; d < RANK; ++d)
    bpc *= ceil_div_u32(dims[d].chunk_shape, tile_T<RANK>(d));
  return bpc;
}

} // namespace

extern "C" int
assemble_launch(CUstream stream,
                uint8_t rank,
                const struct sample_plan* d_samples,
                uint32_t n_samples,
                const struct assemble_chunk* d_chunks,
                uint32_t n_chunks,
                uint32_t max_blocks_per_chunk,
                const void* arena_base,
                void* output_base,
                uint32_t bpe)
{
  (void)n_samples;
  if (n_chunks == 0 || max_blocks_per_chunk == 0)
    return 0;
  if (!d_samples || !d_chunks || !arena_base || !output_base || bpe == 0)
    return 1;

  const uint8_t* arena_b = (const uint8_t*)arena_base;
  uint8_t* output_b = (uint8_t*)output_base;
  int err;
  switch (rank) {
    case 1:
      err = launch_for_bpe<1>(stream,
                              d_samples,
                              d_chunks,
                              n_chunks,
                              max_blocks_per_chunk,
                              arena_b,
                              output_b,
                              bpe);
      break;
    case 2:
      err = launch_for_bpe<2>(stream,
                              d_samples,
                              d_chunks,
                              n_chunks,
                              max_blocks_per_chunk,
                              arena_b,
                              output_b,
                              bpe);
      break;
    case 3:
      err = launch_for_bpe<3>(stream,
                              d_samples,
                              d_chunks,
                              n_chunks,
                              max_blocks_per_chunk,
                              arena_b,
                              output_b,
                              bpe);
      break;
    case 4:
      err = launch_for_bpe<4>(stream,
                              d_samples,
                              d_chunks,
                              n_chunks,
                              max_blocks_per_chunk,
                              arena_b,
                              output_b,
                              bpe);
      break;
    case 5:
      err = launch_for_bpe<5>(stream,
                              d_samples,
                              d_chunks,
                              n_chunks,
                              max_blocks_per_chunk,
                              arena_b,
                              output_b,
                              bpe);
      break;
    default:
      log_error("assemble: unsupported rank=%u (1..5 supported)", rank);
      return 1;
  }
  if (err)
    return err;
  cudaError_t cerr = cudaGetLastError();
  if (cerr != cudaSuccess) {
    log_error("assemble launch: %s", cudaGetErrorString(cerr));
    return 1;
  }
  return 0;
}

extern "C" uint32_t
assemble_blocks_per_chunk(uint8_t rank, const struct sample_dim* dims)
{
  if (!dims)
    return 0;
  switch (rank) {
    case 1:
      return blocks_per_chunk_for_rank<1>(dims);
    case 2:
      return blocks_per_chunk_for_rank<2>(dims);
    case 3:
      return blocks_per_chunk_for_rank<3>(dims);
    case 4:
      return blocks_per_chunk_for_rank<4>(dims);
    case 5:
      return blocks_per_chunk_for_rank<5>(dims);
    default:
      return 0;
  }
}
