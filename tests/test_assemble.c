// Standalone assemble-kernel test. Builds an in-memory chunk arena
// with deterministic typed content, runs assemble_launch directly
// against a host-allocated AABB-tight output tensor, and verifies the
// result element-by-element. Exercises ranks 3, 4, 5 plus a matrix of
// (src_dtype, dst_dtype) cast pairs.

#include "assemble/assemble.h"
#include "dtype/dtype.h"
#include "expect.h"
#include "planner/planner.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RANK 5

// f32 ↔ bf16 helpers, expressed as bit ops to avoid pulling cuda_bf16.h
// into a C TU. Round-to-nearest-even matches __float2bfloat16's
// rounding (NaNs aside, which we don't generate here).
static uint16_t
f32_to_bf16(float v)
{
  uint32_t bits;
  memcpy(&bits, &v, sizeof bits);
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t rounding_bias = 0x7FFFu + lsb;
  bits += rounding_bias;
  return (uint16_t)(bits >> 16);
}

static float
bf16_to_f32(uint16_t b)
{
  uint32_t bits = (uint32_t)b << 16;
  float v;
  memcpy(&v, &bits, sizeof v);
  return v;
}

// f32 → f16 (binary16, RNE) for finite values. Mirrors __float2half for
// the small-magnitude integers this test feeds through.
static uint16_t
f32_to_f16(float v)
{
  uint32_t bits;
  memcpy(&bits, &v, sizeof bits);
  uint32_t sign = (bits >> 31) & 1u;
  int32_t exp = (int32_t)((bits >> 23) & 0xFFu) - 127;
  uint32_t mant = bits & 0x7FFFFFu;
  if ((bits & 0x7FFFFFFFu) == 0)
    return (uint16_t)(sign << 15);
  if (exp > 15)
    return (uint16_t)((sign << 15) | (31u << 10));
  if (exp < -14)
    return (uint16_t)(sign << 15);
  uint32_t lsb = (mant >> 13) & 1u;
  uint32_t rounded = mant + 0xFFFu + lsb;
  int32_t exp_out = exp + (int32_t)(rounded >> 23) + 15;
  if (exp_out >= 31)
    return (uint16_t)((sign << 15) | (31u << 10));
  return (uint16_t)((sign << 15) | ((uint32_t)exp_out << 10) |
                    ((rounded >> 13) & 0x3FFu));
}

// Deterministic source value, XOR'd with `xor_mask` so a scenario can
// push values into negative ranges of signed src dtypes.
static uint32_t
expected_src_u32(uint32_t chunk_idx, uint64_t elem_in_chunk, uint32_t xor_mask)
{
  return (chunk_idx * 17u + (uint32_t)elem_in_chunk) ^ xor_mask;
}

// Mirrors the kernel's load_src_as_float promotion path.
static float
src_to_float(uint32_t src_u32, enum dtype src_dtype)
{
  switch (src_dtype) {
    case dtype_u8:
      return (float)(uint8_t)src_u32;
    case dtype_u16:
      return (float)(uint16_t)src_u32;
    case dtype_i16:
      return (float)(int16_t)(uint16_t)src_u32;
    case dtype_u32:
      return (float)src_u32;
    case dtype_i32:
      return (float)(int32_t)src_u32;
    case dtype_f16:
      return (float)(uint16_t)src_u32; // exact for our small ints
    case dtype_f32:
      return (float)src_u32;
    default:
      return 0.0f;
  }
}

static uint32_t
src_dtype_bpe(enum dtype src_dtype)
{
  switch (src_dtype) {
    case dtype_u8:
      return 1;
    case dtype_u16:
    case dtype_i16:
    case dtype_f16:
      return 2;
    case dtype_u32:
    case dtype_i32:
    case dtype_f32:
      return 4;
    default:
      return 0;
  }
}

static uint32_t
dst_dtype_bpe(enum damacy_dtype dst)
{
  switch (dst) {
    case DAMACY_BF16:
      return 2;
    case DAMACY_F32:
      return 4;
  }
  return 0;
}

static void
write_src_value(uint8_t* slot, uint32_t v, enum dtype src_dtype)
{
  switch (src_dtype) {
    case dtype_u8:
      *slot = (uint8_t)v;
      return;
    case dtype_u16:
      *(uint16_t*)slot = (uint16_t)v;
      return;
    case dtype_i16:
      *(int16_t*)slot = (int16_t)(uint16_t)v;
      return;
    case dtype_u32:
      *(uint32_t*)slot = v;
      return;
    case dtype_i32:
      *(int32_t*)slot = (int32_t)v;
      return;
    case dtype_f16:
      *(uint16_t*)slot = f32_to_f16((float)(uint16_t)v);
      return;
    case dtype_f32:
      *(float*)slot = (float)v;
      return;
    default:
      return;
  }
}

static float
read_dst_as_float(const uint8_t* slot, enum damacy_dtype dst)
{
  switch (dst) {
    case DAMACY_F32:
      return *(const float*)slot;
    case DAMACY_BF16:
      return bf16_to_f32(*(const uint16_t*)slot);
  }
  return NAN;
}

// Row-major linear index over `extents`.
static uint64_t
ravel(const uint32_t* coord, const uint32_t* extents, int rank)
{
  uint64_t lin = 0;
  for (int d = 0; d < rank; ++d)
    lin = lin * extents[d] + coord[d];
  return lin;
}

static int
run_scenario(int rank,
             const uint32_t* S,
             const uint32_t* N,
             const int64_t* aabb_lo_relative,
             const int64_t* aabb_extent,
             enum dtype src_dtype,
             enum damacy_dtype dst_dtype,
             uint32_t value_xor)
{
  const uint32_t sbpe = src_dtype_bpe(src_dtype);
  const uint32_t dbpe = dst_dtype_bpe(dst_dtype);
  EXPECT(sbpe > 0);
  EXPECT(dbpe > 0);

  uint64_t out_volume_elems = 1;
  for (int d = 0; d < rank; ++d)
    out_volume_elems *= (uint64_t)aabb_extent[d];
  size_t out_bytes = (size_t)out_volume_elems * dbpe;

  uint64_t chunk_volume_elems = 1;
  for (int d = 0; d < rank; ++d)
    chunk_volume_elems *= S[d];
  uint64_t chunk_bytes = chunk_volume_elems * sbpe;

  uint32_t n_chunks = 1;
  for (int d = 0; d < rank; ++d)
    n_chunks *= N[d];
  size_t arena_bytes = (size_t)n_chunks * chunk_bytes;

  // Typed arena fill.
  uint8_t* h_arena = (uint8_t*)malloc(arena_bytes);
  EXPECT(h_arena);
  for (uint32_t cidx = 0; cidx < n_chunks; ++cidx) {
    uint8_t* base = h_arena + (size_t)cidx * chunk_bytes;
    for (uint64_t e = 0; e < chunk_volume_elems; ++e)
      write_src_value(
        base + e * sbpe, expected_src_u32(cidx, e, value_xor), src_dtype);
  }

  // Row-major dst strides over [aabb_extent[0..rank-1]].
  int64_t dst_strides[MAX_RANK];
  if (rank > 0) {
    dst_strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; --d)
      dst_strides[d] = dst_strides[d + 1] * aabb_extent[d + 1];
  }
  // Row-major src strides over chunk shape S (in elements).
  int64_t src_strides[MAX_RANK];
  if (rank > 0) {
    src_strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; --d)
      src_strides[d] = src_strides[d + 1] * (int64_t)S[d + 1];
  }

  struct sample_plan sp = { 0 };
  sp.batch_pool_slot = 0;
  sp.sample_idx_in_batch = 0;
  sp.rank = (uint8_t)rank;
  sp.src_dtype = (uint8_t)src_dtype;
  sp.sample_dst_off_elems = 0;
  sp.chunk_offset = 0;
  sp.chunk_count = n_chunks;
  for (int d = 0; d < rank; ++d) {
    sp.dims[d].chunk_shape = S[d];
    sp.dims[d].chunk_grid_extent = N[d];
    sp.dims[d].aabb_lo_relative = aabb_lo_relative[d];
    sp.dims[d].aabb_extent = aabb_extent[d];
    sp.dims[d].dst_stride = dst_strides[d];
    sp.dims[d].src_stride = src_strides[d];
  }

  struct assemble_chunk* h_chunks =
    (struct assemble_chunk*)calloc(n_chunks, sizeof(struct assemble_chunk));
  EXPECT(h_chunks);
  for (uint32_t cidx = 0; cidx < n_chunks; ++cidx) {
    uint32_t chunk_d[MAX_RANK] = { 0 };
    uint32_t rem = cidx;
    for (int d = rank - 1; d >= 0; --d) {
      chunk_d[d] = rem % N[d];
      rem /= N[d];
    }
    h_chunks[cidx].src_base_byte_off = (uint64_t)cidx * chunk_bytes;
    h_chunks[cidx].sample_idx_in_batch = 0;
    for (int d = 0; d < rank; ++d)
      h_chunks[cidx].chunk_d[d] = chunk_d[d];
  }

  CUdeviceptr d_arena = 0, d_output = 0, d_sp = 0, d_chunks = 0;
  EXPECT(cuMemAlloc(&d_arena, arena_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_output, out_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_sp, sizeof(struct sample_plan)) == CUDA_SUCCESS);
  EXPECT(
    cuMemAlloc(&d_chunks, (size_t)n_chunks * sizeof(struct assemble_chunk)) ==
    CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoD(d_arena, h_arena, arena_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemsetD8(d_output, 0xCC, out_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoD(d_sp, &sp, sizeof(sp)) == CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoD(d_chunks,
                      h_chunks,
                      (size_t)n_chunks * sizeof(struct assemble_chunk)) ==
         CUDA_SUCCESS);

  uint32_t max_bpc = assemble_blocks_per_chunk((uint8_t)rank, sp.dims);
  EXPECT(max_bpc > 0);
  EXPECT(assemble_launch(0,
                         (uint8_t)rank,
                         (const struct sample_plan*)(uintptr_t)d_sp,
                         1,
                         (const struct assemble_chunk*)(uintptr_t)d_chunks,
                         n_chunks,
                         max_bpc,
                         (const void*)(uintptr_t)d_arena,
                         (void*)(uintptr_t)d_output,
                         dst_dtype) == 0);
  EXPECT(cuStreamSynchronize(0) == CUDA_SUCCESS);

  uint8_t* h_output = (uint8_t*)malloc(out_bytes);
  EXPECT(h_output);
  EXPECT(cuMemcpyDtoH(h_output, d_output, out_bytes) == CUDA_SUCCESS);

  // Verify each AABB voxel.
  uint32_t v[MAX_RANK] = { 0 };
  int finished = 0;
  while (!finished) {
    uint64_t out_elem_off = 0;
    for (int d = 0; d < rank; ++d)
      out_elem_off += (uint64_t)v[d] * (uint64_t)dst_strides[d];

    uint32_t chunk_d[MAX_RANK] = { 0 };
    uint32_t intra[MAX_RANK] = { 0 };
    for (int d = 0; d < rank; ++d) {
      int64_t src_abs = (int64_t)v[d] + aabb_lo_relative[d];
      chunk_d[d] = (uint32_t)(src_abs / S[d]);
      intra[d] = (uint32_t)(src_abs - (int64_t)chunk_d[d] * S[d]);
    }
    uint32_t cidx = (uint32_t)ravel(chunk_d, N, rank);
    uint64_t elem_idx = ravel(intra, S, rank);
    float want_f =
      src_to_float(expected_src_u32(cidx, elem_idx, value_xor), src_dtype);
    if (dst_dtype == DAMACY_BF16)
      want_f = bf16_to_f32(f32_to_bf16(want_f));

    float got = read_dst_as_float(h_output + out_elem_off * dbpe, dst_dtype);
    if (got != want_f) {
      log_error("rank=%d src=%d dst=%d v=(%u,%u,%u,%u,%u) cidx=%u "
                "elem=%llu got=%f want=%f",
                rank,
                (int)src_dtype,
                (int)dst_dtype,
                v[0],
                rank > 1 ? v[1] : 0,
                rank > 2 ? v[2] : 0,
                rank > 3 ? v[3] : 0,
                rank > 4 ? v[4] : 0,
                cidx,
                (unsigned long long)elem_idx,
                got,
                want_f);
      free(h_output);
      free(h_chunks);
      free(h_arena);
      return 1;
    }

    finished = 1;
    for (int d = rank - 1; d >= 0; --d) {
      v[d]++;
      if ((int64_t)v[d] < aabb_extent[d]) {
        finished = 0;
        break;
      }
      v[d] = 0;
    }
  }

  free(h_output);
  free(h_chunks);
  free(h_arena);
  cuMemFree(d_arena);
  cuMemFree(d_output);
  cuMemFree(d_sp);
  cuMemFree(d_chunks);
  return 0;
}

static int
test_rank3_full(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  int64_t aabb_extent[3] = { 8, 8, 8 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_u16, DAMACY_F32, 0);
}

static int
test_rank3_partial(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 3, 3, 3 };
  int64_t aabb_lo_relative[3] = { 1, 2, 3 };
  int64_t aabb_extent[3] = { 9, 8, 7 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_u16, DAMACY_F32, 0);
}

static int
test_rank3_u8_to_f32(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 1, 0, 2 };
  int64_t aabb_extent[3] = { 6, 8, 5 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_u8, DAMACY_F32, 0);
}

static int
test_rank3_u32_to_f32(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 1, 0 };
  int64_t aabb_extent[3] = { 8, 6, 7 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_u32, DAMACY_F32, 0);
}

static int
test_rank3_f32_to_f32(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  int64_t aabb_extent[3] = { 8, 8, 8 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_f32, DAMACY_F32, 0);
}

static int
test_rank3_u16_to_bf16(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  int64_t aabb_extent[3] = { 8, 8, 8 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_u16, DAMACY_BF16, 0);
}

// XOR with 0xFF80 forces the i16 sign bit and high mantissa bits high,
// producing values like 0xFF91→-111, exercising signed promotion.
static int
test_rank3_i16_neg_to_f32(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  int64_t aabb_extent[3] = { 8, 8, 8 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_i16, DAMACY_F32, 0xFF80u);
}

static int
test_rank3_f16_to_f32(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  int64_t aabb_extent[3] = { 8, 8, 8 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_f16, DAMACY_F32, 0);
}

static int
test_rank3_f32_to_bf16(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  int64_t aabb_extent[3] = { 8, 8, 8 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_f32, DAMACY_BF16, 0);
}

static int
test_rank3_boundary_one_voxel(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  int64_t aabb_extent[3] = { 5, 5, 5 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_u16, DAMACY_F32, 0);
}

static int
test_rank4(void)
{
  uint32_t S[4] = { 2, 4, 4, 4 };
  uint32_t N[4] = { 2, 2, 2, 2 };
  int64_t aabb_lo_relative[4] = { 1, 0, 1, 2 };
  int64_t aabb_extent[4] = { 3, 7, 6, 5 };
  return run_scenario(
    4, S, N, aabb_lo_relative, aabb_extent, dtype_u16, DAMACY_F32, 0);
}

static int
test_rank5(void)
{
  uint32_t S[5] = { 2, 2, 4, 4, 4 };
  uint32_t N[5] = { 2, 2, 2, 2, 2 };
  int64_t aabb_lo_relative[5] = { 0, 1, 0, 2, 1 };
  int64_t aabb_extent[5] = { 3, 3, 7, 5, 6 };
  return run_scenario(
    5, S, N, aabb_lo_relative, aabb_extent, dtype_u16, DAMACY_F32, 0);
}

// Run a rank-3 scenario with the middle chunk in a 2x2x2 grid marked
// is_fill; the assemble kernel should broadcast `fill_byte_pattern` across
// the chunk region for src_dtype (which we choose so the bit pattern
// maps to a known float value). Returns 0 on full match.
static int
run_fill_scenario(enum dtype src_dtype,
                  enum damacy_dtype dst_dtype,
                  uint32_t fill_xor,
                  uint32_t fill_chunk_idx)
{
  const int rank = 3;
  const uint32_t S[3] = { 4, 4, 4 };
  const uint32_t N[3] = { 2, 2, 2 };
  const int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  const int64_t aabb_extent[3] = { 8, 8, 8 };
  const uint32_t sbpe = src_dtype_bpe(src_dtype);
  const uint32_t dbpe = dst_dtype_bpe(dst_dtype);
  EXPECT(sbpe > 0 && dbpe > 0);

  uint64_t out_volume_elems = 1;
  for (int d = 0; d < rank; ++d)
    out_volume_elems *= (uint64_t)aabb_extent[d];
  size_t out_bytes = (size_t)out_volume_elems * dbpe;

  uint64_t chunk_volume_elems = 1;
  for (int d = 0; d < rank; ++d)
    chunk_volume_elems *= S[d];
  uint64_t chunk_bytes = chunk_volume_elems * sbpe;

  uint32_t n_chunks = 1;
  for (int d = 0; d < rank; ++d)
    n_chunks *= N[d];
  size_t arena_bytes = (size_t)n_chunks * chunk_bytes;

  uint8_t* h_arena = (uint8_t*)malloc(arena_bytes);
  EXPECT(h_arena);
  for (uint32_t cidx = 0; cidx < n_chunks; ++cidx) {
    uint8_t* base = h_arena + (size_t)cidx * chunk_bytes;
    for (uint64_t e = 0; e < chunk_volume_elems; ++e)
      write_src_value(base + e * sbpe, expected_src_u32(cidx, e, 0), src_dtype);
  }

  int64_t dst_strides[MAX_RANK];
  dst_strides[rank - 1] = 1;
  for (int d = rank - 2; d >= 0; --d)
    dst_strides[d] = dst_strides[d + 1] * aabb_extent[d + 1];
  int64_t src_strides[MAX_RANK];
  src_strides[rank - 1] = 1;
  for (int d = rank - 2; d >= 0; --d)
    src_strides[d] = src_strides[d + 1] * (int64_t)S[d + 1];

  struct sample_plan sp = { 0 };
  sp.rank = (uint8_t)rank;
  sp.src_dtype = (uint8_t)src_dtype;
  sp.chunk_count = n_chunks;
  for (int d = 0; d < rank; ++d) {
    sp.dims[d].chunk_shape = S[d];
    sp.dims[d].chunk_grid_extent = N[d];
    sp.dims[d].aabb_lo_relative = aabb_lo_relative[d];
    sp.dims[d].aabb_extent = aabb_extent[d];
    sp.dims[d].dst_stride = dst_strides[d];
    sp.dims[d].src_stride = src_strides[d];
  }

  struct assemble_chunk* h_chunks =
    (struct assemble_chunk*)calloc(n_chunks, sizeof(struct assemble_chunk));
  EXPECT(h_chunks);
  uint32_t fill_u32 = expected_src_u32(0, 0, fill_xor);
  for (uint32_t cidx = 0; cidx < n_chunks; ++cidx) {
    uint32_t chunk_d[MAX_RANK] = { 0 };
    uint32_t rem = cidx;
    for (int d = rank - 1; d >= 0; --d) {
      chunk_d[d] = rem % N[d];
      rem /= N[d];
    }
    h_chunks[cidx].src_base_byte_off = (uint64_t)cidx * chunk_bytes;
    h_chunks[cidx].sample_idx_in_batch = 0;
    for (int d = 0; d < rank; ++d)
      h_chunks[cidx].chunk_d[d] = chunk_d[d];
    if (cidx == fill_chunk_idx) {
      h_chunks[cidx].is_fill = 1;
      write_src_value(h_chunks[cidx].fill_value, fill_u32, src_dtype);
    }
  }

  CUdeviceptr d_arena = 0, d_output = 0, d_sp = 0, d_chunks = 0;
  EXPECT(cuMemAlloc(&d_arena, arena_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_output, out_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_sp, sizeof(struct sample_plan)) == CUDA_SUCCESS);
  EXPECT(
    cuMemAlloc(&d_chunks, (size_t)n_chunks * sizeof(struct assemble_chunk)) ==
    CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoD(d_arena, h_arena, arena_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemsetD8(d_output, 0xCC, out_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoD(d_sp, &sp, sizeof(sp)) == CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoD(d_chunks,
                      h_chunks,
                      (size_t)n_chunks * sizeof(struct assemble_chunk)) ==
         CUDA_SUCCESS);

  uint32_t max_bpc = assemble_blocks_per_chunk((uint8_t)rank, sp.dims);
  EXPECT(max_bpc > 0);
  EXPECT(assemble_launch(0,
                         (uint8_t)rank,
                         (const struct sample_plan*)(uintptr_t)d_sp,
                         1,
                         (const struct assemble_chunk*)(uintptr_t)d_chunks,
                         n_chunks,
                         max_bpc,
                         (const void*)(uintptr_t)d_arena,
                         (void*)(uintptr_t)d_output,
                         dst_dtype) == 0);
  EXPECT(cuStreamSynchronize(0) == CUDA_SUCCESS);

  uint8_t* h_output = (uint8_t*)malloc(out_bytes);
  EXPECT(h_output);
  EXPECT(cuMemcpyDtoH(h_output, d_output, out_bytes) == CUDA_SUCCESS);

  // Decode the expected fill value as float (mirrors kernel path).
  float fill_f_want = src_to_float(fill_u32, src_dtype);
  if (dst_dtype == DAMACY_BF16)
    fill_f_want = bf16_to_f32(f32_to_bf16(fill_f_want));

  uint32_t v[MAX_RANK] = { 0 };
  int finished = 0;
  while (!finished) {
    uint64_t out_elem_off = 0;
    for (int d = 0; d < rank; ++d)
      out_elem_off += (uint64_t)v[d] * (uint64_t)dst_strides[d];

    uint32_t chunk_d[MAX_RANK] = { 0 };
    uint32_t intra[MAX_RANK] = { 0 };
    for (int d = 0; d < rank; ++d) {
      int64_t src_abs = (int64_t)v[d] + aabb_lo_relative[d];
      chunk_d[d] = (uint32_t)(src_abs / S[d]);
      intra[d] = (uint32_t)(src_abs - (int64_t)chunk_d[d] * S[d]);
    }
    uint32_t cidx = (uint32_t)ravel(chunk_d, N, rank);
    uint64_t elem_idx = ravel(intra, S, rank);
    float want =
      (cidx == fill_chunk_idx)
        ? fill_f_want
        : src_to_float(expected_src_u32(cidx, elem_idx, 0), src_dtype);
    if (dst_dtype == DAMACY_BF16 && cidx != fill_chunk_idx)
      want = bf16_to_f32(f32_to_bf16(want));
    float got = read_dst_as_float(h_output + out_elem_off * dbpe, dst_dtype);
    int ok = (got == want);
    if (!ok && (want != want) && (got != got))
      ok = 1; // both NaN
    if (!ok) {
      log_error("fill rank=%d v=(%u,%u,%u) cidx=%u elem=%llu got=%f want=%f",
                rank,
                v[0],
                v[1],
                v[2],
                cidx,
                (unsigned long long)elem_idx,
                got,
                want);
      free(h_output);
      free(h_chunks);
      free(h_arena);
      return 1;
    }
    finished = 1;
    for (int d = rank - 1; d >= 0; --d) {
      v[d]++;
      if ((int64_t)v[d] < aabb_extent[d]) {
        finished = 0;
        break;
      }
      v[d] = 0;
    }
  }

  free(h_output);
  free(h_chunks);
  free(h_arena);
  cuMemFree(d_arena);
  cuMemFree(d_output);
  cuMemFree(d_sp);
  cuMemFree(d_chunks);
  return 0;
}

// Mixed AABB: 7 real chunks + 1 fill chunk in a 2x2x2 grid.
static int
test_rank3_fill_int16_neg1(void)
{
  // int16 = 2 bytes; -1 → 0xFFFF; expected float -1.
  return run_fill_scenario(dtype_i16, DAMACY_F32, 0xFFFFu, /*fill_chunk*/ 3);
}

// Float NaN fill_value — the kernel broadcasts NaN across the fill chunk.
static int
test_rank3_fill_f32_nan(void)
{
  // dtype_f32 with the bit pattern of NAN.
  uint32_t nan_bits;
  float nan_f = NAN;
  memcpy(&nan_bits, &nan_f, sizeof nan_bits);
  // expected_src_u32(0,0,xor) = 0 ^ xor = xor; pick xor = nan_bits.
  return run_fill_scenario(dtype_f32, DAMACY_F32, nan_bits, /*fill_chunk*/ 5);
}

// All-fill case: every chunk in the wave is fill, exercises the kernel
// branch with no real chunks. Builds the scenario inline (the mixed
// helper picks exactly one fill chunk).
static int
test_rank3_all_fill(void)
{
  const int rank = 3;
  const uint32_t S[3] = { 4, 4, 4 };
  const uint32_t N[3] = { 2, 2, 2 };
  const int64_t aabb_extent[3] = { 8, 8, 8 };
  const enum dtype src_dtype = dtype_u16;
  const enum damacy_dtype dst_dtype = DAMACY_F32;
  const uint32_t fill_u32 = 7u;
  const uint32_t sbpe = src_dtype_bpe(src_dtype);
  const uint32_t dbpe = dst_dtype_bpe(dst_dtype);

  uint32_t n_chunks = 1;
  for (int d = 0; d < rank; ++d)
    n_chunks *= N[d];
  uint64_t out_volume_elems = 1;
  for (int d = 0; d < rank; ++d)
    out_volume_elems *= (uint64_t)aabb_extent[d];
  size_t out_bytes = (size_t)out_volume_elems * dbpe;
  uint64_t chunk_volume_elems = 1;
  for (int d = 0; d < rank; ++d)
    chunk_volume_elems *= S[d];
  size_t arena_bytes = (size_t)n_chunks * chunk_volume_elems * sbpe;

  int64_t dst_strides[MAX_RANK];
  dst_strides[rank - 1] = 1;
  for (int d = rank - 2; d >= 0; --d)
    dst_strides[d] = dst_strides[d + 1] * aabb_extent[d + 1];
  int64_t src_strides[MAX_RANK];
  src_strides[rank - 1] = 1;
  for (int d = rank - 2; d >= 0; --d)
    src_strides[d] = src_strides[d + 1] * (int64_t)S[d + 1];

  struct sample_plan sp = { 0 };
  sp.rank = (uint8_t)rank;
  sp.src_dtype = (uint8_t)src_dtype;
  sp.chunk_count = n_chunks;
  for (int d = 0; d < rank; ++d) {
    sp.dims[d].chunk_shape = S[d];
    sp.dims[d].chunk_grid_extent = N[d];
    sp.dims[d].aabb_lo_relative = 0;
    sp.dims[d].aabb_extent = aabb_extent[d];
    sp.dims[d].dst_stride = dst_strides[d];
    sp.dims[d].src_stride = src_strides[d];
  }

  struct assemble_chunk* h_chunks =
    (struct assemble_chunk*)calloc(n_chunks, sizeof(struct assemble_chunk));
  EXPECT(h_chunks);
  for (uint32_t cidx = 0; cidx < n_chunks; ++cidx) {
    uint32_t chunk_d[MAX_RANK] = { 0 };
    uint32_t rem = cidx;
    for (int d = rank - 1; d >= 0; --d) {
      chunk_d[d] = rem % N[d];
      rem /= N[d];
    }
    for (int d = 0; d < rank; ++d)
      h_chunks[cidx].chunk_d[d] = chunk_d[d];
    h_chunks[cidx].is_fill = 1;
    write_src_value(h_chunks[cidx].fill_value, fill_u32, src_dtype);
  }

  CUdeviceptr d_arena = 0, d_output = 0, d_sp = 0, d_chunks = 0;
  EXPECT(cuMemAlloc(&d_arena, arena_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_output, out_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemAlloc(&d_sp, sizeof sp) == CUDA_SUCCESS);
  EXPECT(
    cuMemAlloc(&d_chunks, (size_t)n_chunks * sizeof(struct assemble_chunk)) ==
    CUDA_SUCCESS);
  // Poison the arena so any accidental arena read shows up as garbage.
  EXPECT(cuMemsetD8(d_arena, 0xAA, arena_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemsetD8(d_output, 0xCC, out_bytes) == CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoD(d_sp, &sp, sizeof sp) == CUDA_SUCCESS);
  EXPECT(cuMemcpyHtoD(d_chunks,
                      h_chunks,
                      (size_t)n_chunks * sizeof(struct assemble_chunk)) ==
         CUDA_SUCCESS);

  uint32_t max_bpc = assemble_blocks_per_chunk((uint8_t)rank, sp.dims);
  EXPECT(assemble_launch(0,
                         (uint8_t)rank,
                         (const struct sample_plan*)(uintptr_t)d_sp,
                         1,
                         (const struct assemble_chunk*)(uintptr_t)d_chunks,
                         n_chunks,
                         max_bpc,
                         (const void*)(uintptr_t)d_arena,
                         (void*)(uintptr_t)d_output,
                         dst_dtype) == 0);
  EXPECT(cuStreamSynchronize(0) == CUDA_SUCCESS);

  uint8_t* h_output = (uint8_t*)malloc(out_bytes);
  EXPECT(h_output);
  EXPECT(cuMemcpyDtoH(h_output, d_output, out_bytes) == CUDA_SUCCESS);

  float want = src_to_float(fill_u32, src_dtype);
  for (uint64_t e = 0; e < out_volume_elems; ++e) {
    float got = read_dst_as_float(h_output + e * dbpe, dst_dtype);
    if (got != want) {
      log_error(
        "all_fill: elem %llu got=%f want=%f", (unsigned long long)e, got, want);
      free(h_output);
      free(h_chunks);
      return 1;
    }
  }
  free(h_output);
  free(h_chunks);
  cuMemFree(d_arena);
  cuMemFree(d_output);
  cuMemFree(d_sp);
  cuMemFree(d_chunks);
  return 0;
}

static int
test_rank3_non_pow2(void)
{
  uint32_t S[3] = { 3, 5, 7 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 1, 2, 3 };
  int64_t aabb_extent[3] = { 5, 7, 10 };
  return run_scenario(
    3, S, N, aabb_lo_relative, aabb_extent, dtype_u16, DAMACY_F32, 0);
}

int
main(void)
{
  if (cuInit(0) != CUDA_SUCCESS) {
    log_error("cuInit failed");
    return 1;
  }
  CUdevice dev = 0;
  if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
    log_error("cuDeviceGet failed");
    return 1;
  }
  CUcontext ctx = NULL;
  if (cuDevicePrimaryCtxRetain(&ctx, dev) != CUDA_SUCCESS) {
    log_error("cuDevicePrimaryCtxRetain failed");
    return 1;
  }
  if (cuCtxSetCurrent(ctx) != CUDA_SUCCESS) {
    log_error("cuCtxSetCurrent failed");
    return 1;
  }

  RUN(test_rank3_full);
  RUN(test_rank3_partial);
  RUN(test_rank3_u8_to_f32);
  RUN(test_rank3_u32_to_f32);
  RUN(test_rank3_f32_to_f32);
  RUN(test_rank3_u16_to_bf16);
  RUN(test_rank3_i16_neg_to_f32);
  RUN(test_rank3_f16_to_f32);
  RUN(test_rank3_f32_to_bf16);
  RUN(test_rank3_boundary_one_voxel);
  RUN(test_rank3_non_pow2);
  RUN(test_rank3_fill_int16_neg1);
  RUN(test_rank3_fill_f32_nan);
  RUN(test_rank3_all_fill);
  RUN(test_rank4);
  RUN(test_rank5);

  cuDevicePrimaryCtxRelease(dev);
  log_info("all tests passed");
  return 0;
}
