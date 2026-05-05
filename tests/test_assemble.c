// Standalone assemble-kernel test. Builds an in-memory chunk arena
// with deterministic byte content, runs assemble_launch directly
// against a host-allocated AABB-tight output tensor, and verifies the
// result byte-for-byte. Exercises rank 3, 4, 5 plus bpe ∈ {1, 2, 4}
// without going through zarr/IO.

#include "assemble/assemble.h"
#include "expect.h"
#include "planner/planner.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_RANK 5

// Deterministic byte at (chunk_idx, byte_in_chunk).
static uint8_t
expected_byte(uint32_t chunk_idx, uint64_t byte_in_chunk)
{
  return (uint8_t)((chunk_idx * 17u + (uint32_t)byte_in_chunk) & 0xFFu);
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

// Run one (rank, S, N, aabb_lo_relative, aabb_extent, bpe) scenario.
// Returns 0 on success.
static int
run_scenario(int rank,
             const uint32_t* S,
             const uint32_t* N,
             const int64_t* aabb_lo_relative,
             const int64_t* aabb_extent,
             uint32_t bpe)
{
  // dst output tensor shape = aabb_extent (sample slot only; n_samples=1).
  uint64_t out_volume_elems = 1;
  for (int d = 0; d < rank; ++d)
    out_volume_elems *= (uint64_t)aabb_extent[d];
  size_t out_bytes = (size_t)out_volume_elems * bpe;

  uint64_t chunk_volume_elems = 1;
  for (int d = 0; d < rank; ++d)
    chunk_volume_elems *= S[d];
  uint64_t chunk_bytes = chunk_volume_elems * bpe;

  uint32_t n_chunks = 1;
  for (int d = 0; d < rank; ++d)
    n_chunks *= N[d];
  size_t arena_bytes = (size_t)n_chunks * chunk_bytes;

  // Build host arena with deterministic content.
  uint8_t* h_arena = (uint8_t*)malloc(arena_bytes);
  EXPECT(h_arena);
  for (uint32_t cidx = 0; cidx < n_chunks; ++cidx) {
    uint8_t* base = h_arena + (size_t)cidx * chunk_bytes;
    for (uint64_t b = 0; b < chunk_bytes; ++b)
      base[b] = expected_byte(cidx, b);
  }

  // Compute row-major dst strides for [aabb_extent[0..rank-1]].
  int64_t dst_strides[MAX_RANK];
  if (rank > 0) {
    dst_strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; --d)
      dst_strides[d] = dst_strides[d + 1] * aabb_extent[d + 1];
  }
  // Row-major src strides over chunk shape S.
  int64_t src_strides[MAX_RANK];
  if (rank > 0) {
    src_strides[rank - 1] = 1;
    for (int d = rank - 2; d >= 0; --d)
      src_strides[d] = src_strides[d + 1] * (int64_t)S[d + 1];
  }

  // Build sample_plan and assemble_chunk[].
  struct sample_plan sp = { 0 };
  sp.batch_pool_slot = 0;
  sp.sample_idx_in_batch = 0;
  sp.rank = (uint8_t)rank;
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
    // Decode cidx → chunk_d (row-major over N).
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

  // Upload to device.
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
                         bpe) == 0);
  EXPECT(cuStreamSynchronize(0) == CUDA_SUCCESS);

  uint8_t* h_output = (uint8_t*)malloc(out_bytes);
  EXPECT(h_output);
  EXPECT(cuMemcpyDtoH(h_output, d_output, out_bytes) == CUDA_SUCCESS);

  // Verify byte-by-byte. For each spatial position v in AABB, find the
  // chunk it came from and the expected byte at the corresponding
  // arena position.
  uint32_t v[MAX_RANK] = { 0 };
  int finished = 0;
  while (!finished) {
    // Compute output byte offset for this v (sample-local).
    uint64_t out_elem_off = 0;
    for (int d = 0; d < rank; ++d)
      out_elem_off += (uint64_t)v[d] * (uint64_t)dst_strides[d];

    // Source absolute position = v + aabb_lo_relative.
    // Chunk holding this voxel: chunk_d[d] = src_abs[d] / S[d].
    // intra[d] = src_abs[d] - chunk_d[d] * S[d].
    uint32_t chunk_d[MAX_RANK] = { 0 };
    uint32_t intra[MAX_RANK] = { 0 };
    for (int d = 0; d < rank; ++d) {
      int64_t src_abs = (int64_t)v[d] + aabb_lo_relative[d];
      chunk_d[d] = (uint32_t)(src_abs / S[d]);
      intra[d] = (uint32_t)(src_abs - (int64_t)chunk_d[d] * S[d]);
    }
    uint32_t cidx = (uint32_t)ravel(chunk_d, N, rank);
    uint64_t elem_idx = ravel(intra, S, rank);
    uint64_t byte_in_chunk_base = elem_idx * bpe;

    uint64_t out_byte_off = out_elem_off * bpe;
    for (uint32_t b = 0; b < bpe; ++b) {
      uint8_t got = h_output[out_byte_off + b];
      uint8_t want = expected_byte(cidx, byte_in_chunk_base + b);
      if (got != want) {
        log_error("rank=%d bpe=%u v=(%u,%u,%u,%u,%u) cidx=%u elem=%llu "
                  "byte=%u got=0x%02x want=0x%02x",
                  rank,
                  bpe,
                  v[0],
                  rank > 1 ? v[1] : 0,
                  rank > 2 ? v[2] : 0,
                  rank > 3 ? v[3] : 0,
                  rank > 4 ? v[4] : 0,
                  cidx,
                  (unsigned long long)elem_idx,
                  b,
                  got,
                  want);
        free(h_output);
        free(h_chunks);
        free(h_arena);
        return 1;
      }
    }

    // Advance v row-major over aabb_extent.
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

// rank-3, full AABB, bpe=2.
static int
test_rank3_full(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  int64_t aabb_extent[3] = { 8, 8, 8 };
  return run_scenario(3, S, N, aabb_lo_relative, aabb_extent, 2);
}

// rank-3, partial AABB crossing chunk boundaries on multiple axes.
static int
test_rank3_partial(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 3, 3, 3 };
  int64_t aabb_lo_relative[3] = { 1, 2, 3 };
  int64_t aabb_extent[3] = { 9, 8, 7 };
  return run_scenario(3, S, N, aabb_lo_relative, aabb_extent, 2);
}

// rank-3 bpe matrix.
static int
test_rank3_bpe1(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 1, 0, 2 };
  int64_t aabb_extent[3] = { 6, 8, 5 };
  return run_scenario(3, S, N, aabb_lo_relative, aabb_extent, 1);
}

static int
test_rank3_bpe4(void)
{
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 1, 0 };
  int64_t aabb_extent[3] = { 8, 6, 7 };
  return run_scenario(3, S, N, aabb_lo_relative, aabb_extent, 4);
}

// Boundary off-by-one cases.
static int
test_rank3_boundary_one_voxel(void)
{
  // AABB exactly one voxel wider than one chunk in each axis.
  uint32_t S[3] = { 4, 4, 4 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 0, 0, 0 };
  int64_t aabb_extent[3] = { 5, 5, 5 };
  return run_scenario(3, S, N, aabb_lo_relative, aabb_extent, 2);
}

// rank-4.
static int
test_rank4(void)
{
  uint32_t S[4] = { 2, 4, 4, 4 };
  uint32_t N[4] = { 2, 2, 2, 2 };
  int64_t aabb_lo_relative[4] = { 1, 0, 1, 2 };
  int64_t aabb_extent[4] = { 3, 7, 6, 5 };
  return run_scenario(4, S, N, aabb_lo_relative, aabb_extent, 2);
}

// rank-5.
static int
test_rank5(void)
{
  uint32_t S[5] = { 2, 2, 4, 4, 4 };
  uint32_t N[5] = { 2, 2, 2, 2, 2 };
  int64_t aabb_lo_relative[5] = { 0, 1, 0, 2, 1 };
  int64_t aabb_extent[5] = { 3, 3, 7, 5, 6 };
  return run_scenario(5, S, N, aabb_lo_relative, aabb_extent, 2);
}

// Non-power-of-two chunk shape.
static int
test_rank3_non_pow2(void)
{
  uint32_t S[3] = { 3, 5, 7 };
  uint32_t N[3] = { 2, 2, 2 };
  int64_t aabb_lo_relative[3] = { 1, 2, 3 };
  int64_t aabb_extent[3] = { 5, 7, 10 };
  return run_scenario(3, S, N, aabb_lo_relative, aabb_extent, 2);
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
  RUN(test_rank3_bpe1);
  RUN(test_rank3_bpe4);
  RUN(test_rank3_boundary_one_voxel);
  RUN(test_rank3_non_pow2);
  RUN(test_rank4);
  RUN(test_rank5);

  cuDevicePrimaryCtxRelease(dev);
  log_info("all tests passed");
  return 0;
}
