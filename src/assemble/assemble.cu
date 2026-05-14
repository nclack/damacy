#include "assemble/assemble.h"

#include "damacy_limits.h"
#include "dtype/dtype.h"
#include "log/log.h"
#include "util/prelude.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

// Block-tile shape T[d] within a chunk for a rank-`rank` sample. Threads
// run innermost-fastest: 32 along dim rank-1, 8 along dim rank-2, 1
// elsewhere (kBlockSize == 256).
//
//   rank 1: T = {256}
//   rank 2: T = {8, 32}
//   rank R >= 3: T = {..., 1, 8, 32}

constexpr uint32_t kBlockSize = 256;

// Ranks <= kMaxTemplateRank get a fully-templated kernel (loop unrolled,
// tile_T / thread_dim resolved at compile time). Higher ranks fall
// through to a runtime-rank kernel up to DAMACY_MAX_RANK.
constexpr int kMaxTemplateRank = 8;

__device__ __host__ __forceinline__ uint32_t
tile_T(int rank, int d)
{
  if (rank == 1)
    return kBlockSize;
  if (d == rank - 1)
    return 32u;
  if (d == rank - 2)
    return 8u;
  return 1u;
}

// Decode a 1D thread index into per-dim coordinates within the block tile.
__device__ __forceinline__ uint32_t
thread_dim(int rank, uint32_t tid, int d)
{
  if (rank == 1)
    return tid;
  if (d == rank - 1)
    return tid & 31u;
  if (d == rank - 2)
    return (tid >> 5) & 7u;
  return 0u;
}

__device__ __host__ __forceinline__ uint32_t
ceil_div_u32(uint32_t a, uint32_t b)
{
  return (a + b - 1u) / b;
}

// --- per-element read+cast --------------------------------------------------
//
// The cast is a per-thread switch on src_dtype. The branch is uniform
// per block (one chunk → one sample → one src_dtype), so warp divergence
// stays at the kernel-grid boundary, not within a warp.

template<typename dst_t>
__device__ __forceinline__ dst_t
cast_to_dst(float v);

template<>
__device__ __forceinline__ float
cast_to_dst<float>(float v)
{
  return v;
}

template<>
__device__ __forceinline__ __nv_bfloat16
cast_to_dst<__nv_bfloat16>(float v)
{
  return __float2bfloat16(v);
}

// Load one source element as float, given src_dtype. Reads exactly
// sizeof(src_t) bytes from `src` and promotes to float for the cast
// path. Unsupported src_dtype values are treated as zero — the host
// side is expected to reject them at push.
__device__ __forceinline__ float
load_src_as_float(const uint8_t* src, uint8_t src_dtype)
{
  switch ((enum dtype)src_dtype) {
    case dtype_u8:
      return (float)(*src);
    case dtype_u16:
      return (float)(*(const uint16_t*)src);
    case dtype_i16:
      return (float)(*(const int16_t*)src);
    case dtype_u32:
      return (float)(*(const uint32_t*)src);
    case dtype_i32:
      return (float)(*(const int32_t*)src);
    case dtype_f16:
      return __half2float(*(const __half*)src);
    case dtype_f32:
      return *(const float*)src;
    default:
      return 0.0f;
  }
}

// Gather element `elem_idx` from a blosc byte-shuffled chunk. Within
// one blosc block of `blocksize` bytes (= N elements of `typesize`
// bytes), the shuffled layout puts byte b of element e at
// `b*N + e`. Reassemble the T bytes locally, then go through the
// existing dtype dispatch.
__device__ __forceinline__ float
load_src_as_float_unshuffled(const uint8_t* chunk_base,
                             int64_t elem_idx,
                             uint8_t typesize,
                             uint32_t blocksize,
                             uint8_t src_dtype)
{
  const uint64_t elem_byte = (uint64_t)elem_idx * (uint64_t)typesize;
  const uint64_t block_idx = elem_byte / (uint64_t)blocksize;
  const uint32_t intra_byte = (uint32_t)(elem_byte - block_idx * blocksize);
  const uint32_t elems_per_block = blocksize / typesize;
  const uint32_t within = intra_byte / typesize;
  const uint8_t* block = chunk_base + (size_t)block_idx * blocksize;
  uint8_t tmp[8];
#pragma unroll
  for (uint32_t t = 0; t < 8; ++t)
    if (t < typesize)
      tmp[t] = block[t * elems_per_block + within];
  return load_src_as_float(tmp, src_dtype);
}

// Gather element `elem_idx` from a blosc bit-shuffled chunk. Each
// block holds T*8 bit-planes of N bits packed LSB-first into N/8
// bytes. Output byte bb of element e gets its bit bp from plane bb*8+bp
// at byte (e>>3), bit (e&7).
__device__ __forceinline__ float
load_src_as_float_bitunshuffled(const uint8_t* chunk_base,
                                int64_t elem_idx,
                                uint8_t typesize,
                                uint32_t blocksize,
                                uint8_t src_dtype)
{
  const uint32_t elems_per_block = blocksize / typesize;
  const uint32_t bytes_per_plane = elems_per_block >> 3;
  const uint64_t block_idx = (uint64_t)elem_idx / elems_per_block;
  const uint32_t e =
    (uint32_t)((uint64_t)elem_idx - block_idx * (uint64_t)elems_per_block);
  const uint32_t e_byte = e >> 3;
  const uint32_t e_bit = e & 7u;
  const uint8_t* block = chunk_base + (size_t)block_idx * blocksize;
  uint8_t tmp[8];
#pragma unroll
  for (uint32_t bb = 0; bb < 8; ++bb) {
    if (bb >= typesize)
      break;
    uint8_t out = 0;
#pragma unroll
    for (uint32_t bp = 0; bp < 8; ++bp) {
      const uint8_t plane =
        block[(size_t)(bb * 8u + bp) * bytes_per_plane + e_byte];
      out = (uint8_t)(out | (((plane >> e_bit) & 1u) << bp));
    }
    tmp[bb] = out;
  }
  return load_src_as_float(tmp, src_dtype);
}

// Bytes per source element for stride math. Mirrors dtype_bpe but as a
// device-side constexpr-friendly helper. Only the supported casts appear.
__device__ __forceinline__ uint32_t
src_bpe(uint8_t src_dtype)
{
  switch ((enum dtype)src_dtype) {
    case dtype_u8:
      return 1u;
    case dtype_u16:
    case dtype_i16:
    case dtype_f16:
      return 2u;
    case dtype_u32:
    case dtype_i32:
    case dtype_f32:
      return 4u;
    default:
      return 0u;
  }
}

// Per-thread copy. The decode of blockIdx.x → tile-in-chunk and the
// per-dim accumulation of src/dst offsets and bounds fuse into one
// reverse loop: the accumulators commute across d, so the row-major
// modulus chain (innermost dim first) doesn't constrain the order.
template<typename dst_t>
__device__ __forceinline__ void
assemble_body(int rank,
              const struct sample_plan& s,
              const struct assemble_chunk& c,
              const uint8_t* arena_base,
              uint8_t* output_base)
{
  const uint32_t tid = threadIdx.x;
  uint32_t rem = blockIdx.x;
  int64_t src_off_elems = 0;
  int64_t dst_off_elems = s.sample_dst_off_elems;
  bool in_bounds = true;

#pragma unroll
  for (int d = rank - 1; d >= 0; --d) {
    const struct sample_dim sd = s.dims[d];
    const uint32_t T_d = tile_T(rank, d);
    const uint32_t tiles_d = ceil_div_u32(sd.chunk_shape, T_d);
    const uint32_t tile_in_chunk = rem % tiles_d;
    rem /= tiles_d;
    const uint32_t intra = tile_in_chunk * T_d + thread_dim(rank, tid, d);
    in_bounds = in_bounds && (intra < sd.chunk_shape);
    const int64_t u =
      (int64_t)c.chunk_d[d] * (int64_t)sd.chunk_shape + (int64_t)intra;
    const int64_t dst_d = u - sd.aabb_lo_relative;
    in_bounds = in_bounds && (dst_d >= 0) && (dst_d < sd.aabb_extent);
    src_off_elems += (int64_t)intra * sd.src_stride;
    dst_off_elems += dst_d * sd.dst_stride;
  }
  if (rem != 0 || !in_bounds)
    return;

  const uint32_t sbpe = src_bpe(s.src_dtype);
  const uint8_t* chunk_base = arena_base + c.src_base_byte_off;
  dst_t* dst = (dst_t*)(output_base + dst_off_elems * (int64_t)sizeof(dst_t));

  // Each block handles one chunk, so the shuffle switch is uniform
  // across the block — no warp divergence.
  float v;
  switch ((enum assemble_shuffle_mode)c.shuffle_mode) {
    case ASSEMBLE_SHUFFLE_BYTE:
      v = load_src_as_float_unshuffled(chunk_base,
                                       src_off_elems,
                                       c.shuffle_typesize,
                                       c.shuffle_blocksize,
                                       s.src_dtype);
      break;
    case ASSEMBLE_SHUFFLE_BIT:
      v = load_src_as_float_bitunshuffled(chunk_base,
                                          src_off_elems,
                                          c.shuffle_typesize,
                                          c.shuffle_blocksize,
                                          s.src_dtype);
      break;
    case ASSEMBLE_SHUFFLE_NONE:
    default:
      v = load_src_as_float(chunk_base + src_off_elems * (int64_t)sbpe,
                            s.src_dtype);
      break;
  }
  *dst = cast_to_dst<dst_t>(v);
}

// Single kernel template covering both the templated-rank fast path
// (RANK_TPL > 0) and the runtime-rank fallback (RANK_TPL == 0).
template<int RANK_TPL, typename dst_t>
__global__ void
assemble_kernel(const struct sample_plan* __restrict__ d_samples,
                const struct assemble_chunk* __restrict__ d_chunks,
                const uint8_t* __restrict__ arena_base,
                uint8_t* __restrict__ output_base)
{
  const struct assemble_chunk c = d_chunks[blockIdx.y];
  const struct sample_plan& s = d_samples[c.sample_idx_in_batch];
  const int rank = (RANK_TPL == 0) ? (int)s.rank : RANK_TPL;
  assemble_body<dst_t>(rank, s, c, arena_base, output_base);
}

// Dispatch kernel launch over destination dtype. RANK_TPL_VAL == 0
// selects the runtime-rank kernel; otherwise it's the compile-time rank.
#define DST_CASE(DST_VAL, DST_T)                                               \
  case DST_VAL:                                                                \
    assemble_kernel<kRankTpl, DST_T>                                           \
      <<<grid, block, 0, stream>>>(d_samples, d_chunks, arena_b, output_b);    \
    break

#define DISPATCH_DST(RANK_TPL_VAL)                                             \
  do {                                                                         \
    constexpr int kRankTpl = (RANK_TPL_VAL);                                   \
    switch (dst_dtype) {                                                       \
      DST_CASE(DAMACY_F32, float);                                             \
      DST_CASE(DAMACY_BF16, __nv_bfloat16);                                    \
      default:                                                                 \
        log_error("assemble: unsupported dst_dtype=%d", (int)dst_dtype);       \
        return 1;                                                              \
    }                                                                          \
  } while (0)

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
                enum damacy_dtype dst_dtype)
{
  (void)n_samples;
  if (n_chunks == 0 || max_blocks_per_chunk == 0)
    return 0;
  if (!d_samples || !d_chunks || !arena_base || !output_base)
    return 1;
  if (rank == 0 || rank > DAMACY_MAX_RANK) {
    log_error("assemble: unsupported rank=%u (1..%u supported)",
              (unsigned)rank,
              (unsigned)DAMACY_MAX_RANK);
    return 1;
  }

  const uint8_t* arena_b = (const uint8_t*)arena_base;
  uint8_t* output_b = (uint8_t*)output_base;
  dim3 grid(max_blocks_per_chunk, n_chunks, 1);
  dim3 block(kBlockSize, 1, 1);

#define LAUNCH_RANK(R)                                                         \
  case R:                                                                      \
    DISPATCH_DST(R);                                                           \
    break

  static_assert(kMaxTemplateRank == 8, "update LAUNCH_RANK case list");
  switch (rank) {
    LAUNCH_RANK(1);
    LAUNCH_RANK(2);
    LAUNCH_RANK(3);
    LAUNCH_RANK(4);
    LAUNCH_RANK(5);
    LAUNCH_RANK(6);
    LAUNCH_RANK(7);
    LAUNCH_RANK(8);
    default:
      // Runtime-rank fallback for ranks 9..DAMACY_MAX_RANK.
      DISPATCH_DST(0);
      break;
  }
#undef LAUNCH_RANK

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
  if (!dims || rank == 0 || rank > DAMACY_MAX_RANK)
    return 0;
  uint32_t bpc = 1u;
  for (int d = 0; d < (int)rank; ++d)
    bpc *= ceil_div_u32(dims[d].chunk_shape, tile_T((int)rank, d));
  return bpc;
}
