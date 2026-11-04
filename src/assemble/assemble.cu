#include "assemble/assemble.h"

#include "damacy_limits.h"
#include "dtype/convert.h"
#include "log/log.h"
#include "util/prelude.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr uint32_t kBlockSize = 256;
constexpr int kMaxTemplateRank = 8;

template<enum damacy_dtype output, typename dst_t>
__device__ __forceinline__ dst_t
load_source(const uint8_t* source, uint8_t input)
{
  dst_t value;
  dtype_convert(&value, output, source, (enum dtype)input);
  return value;
}

// Gather element `elem_idx` from a blosc byte-shuffled chunk. Within
// one blosc block of `blocksize` bytes (= N elements of `typesize`
// bytes), the shuffled layout puts byte b of element e at
// `b*N + e`. Reassemble the T bytes locally, then go through the
// existing dtype dispatch.
template<enum damacy_dtype output, typename dst_t>
__device__ __forceinline__ dst_t
load_unshuffled(const uint8_t* chunk_base,
                uint32_t elem_idx,
                uint8_t typesize,
                uint32_t blocksize,
                uint32_t chunk_bytes,
                uint8_t src_dtype)
{
  const uint32_t elem_byte = elem_idx * typesize;
  const uint32_t intra_byte = elem_byte % blocksize;
  const uint32_t block_offset = elem_byte - intra_byte;
  const uint32_t block_bytes = min(blocksize, chunk_bytes - block_offset);
  const uint32_t elems_per_block = block_bytes / typesize;
  const uint32_t within = intra_byte / typesize;
  const uint8_t* block = chunk_base + block_offset;
  uint8_t tmp[8];
#pragma unroll
  for (uint32_t t = 0; t < 8; ++t)
    if (t < typesize)
      tmp[t] = block[t * elems_per_block + within];
  return load_source<output, dst_t>(tmp, src_dtype);
}

// Gather element `elem_idx` from a blosc bit-shuffled chunk. Each
// block holds T*8 bit-planes of N bits packed LSB-first into N/8
// bytes. Output byte bb of element e gets its bit bp from plane bb*8+bp
// at byte (e>>3), bit (e&7).
template<enum damacy_dtype output, typename dst_t>
__device__ __forceinline__ dst_t
load_bitunshuffled(const uint8_t* chunk_base,
                   uint32_t elem_idx,
                   uint8_t typesize,
                   uint32_t blocksize,
                   uint32_t chunk_bytes,
                   uint8_t src_dtype)
{
  const uint32_t elem_byte = elem_idx * typesize;
  const uint32_t intra_byte = elem_byte % blocksize;
  const uint32_t block_offset = elem_byte - intra_byte;
  const uint32_t block_bytes = min(blocksize, chunk_bytes - block_offset);
  const uint32_t elems_per_block = block_bytes / typesize;
  const uint32_t e = intra_byte / typesize;
  const uint8_t* block = chunk_base + block_offset;
  if (elems_per_block % 8u)
    return load_source<output, dst_t>(block + e * typesize, src_dtype);
  const uint32_t bytes_per_plane = elems_per_block >> 3;
  const uint32_t e_byte = e >> 3;
  const uint32_t e_bit = e & 7u;
  uint8_t tmp[8];
#pragma unroll
  for (uint32_t bb = 0; bb < 8; ++bb) {
    if (bb >= typesize)
      break;
    uint8_t out = 0;
#pragma unroll
    for (uint32_t bp = 0; bp < 8; ++bp) {
      const uint8_t plane = block[(bb * 8u + bp) * bytes_per_plane + e_byte];
      out = (uint8_t)(out | (((plane >> e_bit) & 1u) << bp));
    }
    tmp[bb] = out;
  }
  return load_source<output, dst_t>(tmp, src_dtype);
}

// Bytes per source element for stride math. Mirrors dtype_bpe but as a
// device-side constexpr-friendly helper. Only the supported casts appear.
__device__ __forceinline__ uint32_t
src_bpe(uint8_t src_dtype)
{
  switch ((enum dtype)src_dtype) {
    case dtype_u8:
    case dtype_i8:
      return 1u;
    case dtype_u16:
    case dtype_i16:
    case dtype_f16:
      return 2u;
    case dtype_u32:
    case dtype_i32:
    case dtype_f32:
      return 4u;
    case dtype_u64:
    case dtype_i64:
      return 8u;
    default:
      return 0u;
  }
}

template<enum damacy_dtype output, typename dst_t>
__device__ __forceinline__ dst_t
load_chunk_value(const struct sample_plan& s,
                 const struct assemble_chunk& c,
                 const uint8_t* arena_base,
                 uint32_t source)
{
  if (c.is_fill)
    return load_source<output, dst_t>(s.fill_value, s.src_dtype);
  const uint8_t* chunk_base = arena_base + c.src_base_byte_off;
  switch ((enum assemble_shuffle_mode)c.shuffle_mode) {
    case ASSEMBLE_SHUFFLE_BYTE:
      return load_unshuffled<output, dst_t>(chunk_base,
                                            source,
                                            c.shuffle_typesize,
                                            c.shuffle_blocksize,
                                            s.layout.nbytes,
                                            s.src_dtype);
    case ASSEMBLE_SHUFFLE_BIT:
      return load_bitunshuffled<output, dst_t>(chunk_base,
                                               source,
                                               c.shuffle_typesize,
                                               c.shuffle_blocksize,
                                               s.layout.nbytes,
                                               s.src_dtype);
    default:
      return load_source<output, dst_t>(
        chunk_base + source * src_bpe(s.src_dtype), s.src_dtype);
  }
}

template<enum damacy_dtype output, typename dst_t>
__device__ __forceinline__ void
gather_body(int rank,
            const struct sample_plan& s,
            const struct assemble_chunk& c,
            const uint8_t* arena_base,
            uint8_t* output_base)
{
  const struct gather_dim* gather = s.gather_dims + c.gather_offset;
  for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
       i < c.elements;
       i += (uint64_t)gridDim.x * blockDim.x) {
    uint64_t remaining = i;
    uint32_t source = 0;
    uint64_t destination = s.sample_dst_off_elems;
#pragma unroll
    for (int d = rank - 1; d >= 0; --d) {
      const struct gather_dim dim = gather[d];
      uint32_t position = (uint32_t)(remaining % dim.count);
      remaining /= dim.count;
      uint32_t src = dim.begin + position;
      uint64_t dst = dim.output_begin + position;
      if (s.dims[d].index_count) {
        const struct gather_index index = s.indices[src];
        src = index.source;
        dst = index.output;
      }
      source += src * (uint32_t)s.dims[d].src_stride;
      destination += dst * s.dims[d].dst_stride;
    }
    ((dst_t*)output_base)[destination] =
      load_chunk_value<output, dst_t>(s, c, arena_base, source);
  }
}

template<enum damacy_dtype output, typename dst_t>
__device__ __forceinline__ void
assemble_body(int rank,
              const struct sample_plan& s,
              const struct assemble_chunk& c,
              const uint8_t* arena_base,
              uint8_t* output_base)
{
  const uint32_t element = blockIdx.x * kBlockSize + threadIdx.x;
  if (element >= (uint32_t)c.elements)
    return;
  uint32_t remaining = element;
  uint32_t source = c.src_begin_elem;
  uint64_t destination = c.dst_begin_elem;
#pragma unroll
  for (int d = rank - 1; d >= 0; --d) {
    const uint32_t position = remaining % c.shape[d];
    remaining /= c.shape[d];
    source += position * (uint32_t)s.dims[d].src_stride;
    destination += (uint64_t)position * s.dims[d].dst_stride;
  }
  ((dst_t*)output_base)[destination] =
    load_chunk_value<output, dst_t>(s, c, arena_base, source);
}

// Single kernel template covering both the templated-rank fast path
// (RANK_TPL > 0) and the runtime-rank fallback (RANK_TPL == 0).
template<int RANK_TPL, enum damacy_dtype output, typename dst_t, bool INDEXED>
__global__ void
assemble_kernel(const struct sample_plan* __restrict__ d_samples,
                const struct assemble_chunk* __restrict__ d_chunks,
                const uint8_t* __restrict__ arena_base,
                uint8_t* __restrict__ output_base)
{
  const struct assemble_chunk& c = d_chunks[blockIdx.y];
  if (blockIdx.x >= c.blocks)
    return;
  const struct sample_plan& s = d_samples[c.sample_idx_in_batch];
  if (s.indexed != INDEXED)
    return;
  const int rank = (RANK_TPL == 0) ? (int)s.rank : RANK_TPL;
  if (INDEXED)
    gather_body<output, dst_t>(rank, s, c, arena_base, output_base);
  else
    assemble_body<output, dst_t>(rank, s, c, arena_base, output_base);
}

// Dispatch kernel launch over destination dtype. RANK_TPL_VAL == 0
// selects the runtime-rank kernel; otherwise it's the compile-time rank.
#define DST_CASE(DST_VAL, DST_T)                                               \
  case DST_VAL:                                                                \
    if (indexed)                                                               \
      assemble_kernel<kRankTpl, DST_VAL, DST_T, true>                          \
        <<<grid, block, 0, stream>>>(d_samples, d_chunks, arena_b, output_b);  \
    else                                                                       \
      assemble_kernel<kRankTpl, DST_VAL, DST_T, false>                         \
        <<<grid, block, 0, stream>>>(d_samples, d_chunks, arena_b, output_b);  \
    break

#define DISPATCH_DST(RANK_TPL_VAL)                                             \
  do {                                                                         \
    constexpr int kRankTpl = (RANK_TPL_VAL);                                   \
    switch (dst_dtype) {                                                       \
      DST_CASE(DAMACY_F32, float);                                             \
      DST_CASE(DAMACY_BF16, uint16_t);                                         \
      DST_CASE(DAMACY_U8, uint8_t);                                            \
      DST_CASE(DAMACY_U16, uint16_t);                                          \
      DST_CASE(DAMACY_U32, uint32_t);                                          \
      DST_CASE(DAMACY_U64, uint64_t);                                          \
      DST_CASE(DAMACY_I8, int8_t);                                             \
      DST_CASE(DAMACY_I16, int16_t);                                           \
      DST_CASE(DAMACY_I32, int32_t);                                           \
      DST_CASE(DAMACY_I64, int64_t);                                           \
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
                uint8_t indexed,
                const struct assemble_chunk* d_chunks,
                uint32_t n_chunks,
                uint32_t max_blocks_per_chunk,
                const void* arena_base,
                void* output_base,
                enum damacy_dtype dst_dtype)
{
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

extern "C" uint64_t
assemble_prepare_chunk(struct assemble_chunk* out,
                       const struct sample_plan* s,
                       const struct chunk_plan* c,
                       const struct gather_dim* gather_dims)
{
  *out = {};
  out->src_base_byte_off = c->dev_decompressed_offset;
  out->sample_idx_in_batch = c->sample_idx_in_batch;
  out->is_fill = c->is_fill;
  out->gather_offset = c->gather_offset;
  out->dst_begin_elem = s->sample_dst_off_elems;
  uint64_t elements = 1;
  for (int d = 0; d < s->rank; ++d) {
    if (s->indexed) {
      elements *= gather_dims[c->gather_offset + d].count;
      continue;
    }
    const struct sample_dim sd = s->dims[d];
    const int64_t origin =
      (int64_t)c->chunk_d[d] * sd.chunk_shape - sd.aabb_lo_relative;
    const int64_t lo = origin > 0 ? origin : 0;
    const int64_t hi = origin + sd.chunk_shape < sd.aabb_extent
                         ? origin + sd.chunk_shape
                         : sd.aabb_extent;
    if (hi <= lo)
      return 0;
    out->shape[d] = (uint32_t)(hi - lo);
    out->src_begin_elem += (uint32_t)((lo - origin) * sd.src_stride);
    out->dst_begin_elem += (uint64_t)lo * sd.dst_stride;
    elements *= out->shape[d];
  }
  out->elements = elements;
  const uint64_t blocks = elements / kBlockSize + (elements % kBlockSize != 0);
  out->blocks = (uint32_t)(s->indexed && blocks > 65535 ? 65535 : blocks);
  return elements;
}
