#include "assemble.h"

#include "limits.h"
#include "log/log.h"
#include "util/prelude.h"

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
// For rank R >= 2: tid % 32 → dim R-1, (tid / 32) % 8 → dim R-2, 0
// elsewhere. For rank 1 the whole thread index maps to the single dim.
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

// Per-thread copy. The decode of blockIdx.x → tile-in-chunk and the
// per-dim accumulation of src/dst offsets and bounds fuse into one
// reverse loop: the accumulators commute across d, so the row-major
// modulus chain (innermost dim first) doesn't constrain the order.
// After the loop, rem != 0 iff blockIdx.x >= ∏ tiles_per_chunk[d] —
// the surplus blocks gridDim.x emits because it's the wave-wide max.
template<typename copy_t>
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

  const uint8_t* src =
    arena_base + c.src_base_byte_off + src_off_elems * (int64_t)sizeof(copy_t);
  uint8_t* dst = output_base + dst_off_elems * (int64_t)sizeof(copy_t);
  *(copy_t*)dst = *(const copy_t*)src;
}

// Single kernel template covering both the templated-rank fast path
// (RANK_TPL > 0) and the runtime-rank fallback (RANK_TPL == 0).
template<int RANK_TPL, typename copy_t>
__global__ void
assemble_kernel(const struct sample_plan* __restrict__ d_samples,
                const struct assemble_chunk* __restrict__ d_chunks,
                const uint8_t* __restrict__ arena_base,
                uint8_t* __restrict__ output_base)
{
  const struct assemble_chunk c = d_chunks[blockIdx.y];
  const struct sample_plan& s = d_samples[c.sample_idx_in_batch];
  const int rank = (RANK_TPL == 0) ? (int)s.rank : RANK_TPL;
  assemble_body<copy_t>(rank, s, c, arena_base, output_base);
}

// Dispatch kernel launch over bpe ∈ {1,2,4,8}. RANK_TPL_VAL == 0 selects
// the runtime-rank kernel; otherwise it's the compile-time rank.
// Embedded in a function: the default arm returns 1 from the caller.
// Uses identifiers (bpe, grid, block, stream, d_samples, d_chunks,
// arena_b, output_b) from the enclosing scope.
#define BPE_CASE(BPE_VAL, COPY_T)                                              \
  case BPE_VAL:                                                                \
    assemble_kernel<kRankTpl, COPY_T>                                          \
      <<<grid, block, 0, stream>>>(d_samples, d_chunks, arena_b, output_b);    \
    break

#define DISPATCH_BPE(RANK_TPL_VAL)                                             \
  do {                                                                         \
    constexpr int kRankTpl = (RANK_TPL_VAL);                                   \
    switch (bpe) {                                                             \
      BPE_CASE(1, uint8_t);                                                    \
      BPE_CASE(2, uint16_t);                                                   \
      BPE_CASE(4, uint32_t);                                                   \
      BPE_CASE(8, uint64_t);                                                   \
      default:                                                                 \
        log_error("assemble: unsupported bpe=%u", bpe);                        \
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
                uint32_t bpe)
{
  (void)n_samples;
  if (n_chunks == 0 || max_blocks_per_chunk == 0)
    return 0;
  if (!d_samples || !d_chunks || !arena_base || !output_base || bpe == 0)
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
    DISPATCH_BPE(R);                                                           \
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
      DISPATCH_BPE(0);
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
