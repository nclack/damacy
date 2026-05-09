// Unit tests for the host-side parts of src/batch_pool/.
//
//   test_compute_layout_simple — shape/strides/n_bytes for a 2-D AABB
//   test_compute_layout_idempotent — second call short-circuits
//   test_shape_match           — sample_shape_matches_pool accepts +
//                                rejects mismatches
//   test_state_predicates      — find_*_slot scan in batch_id order
//
// We don't exercise batch_slot_init / batch_pool_alloc_dev: those need
// a CUcontext and would belong in a CUDA-gated test.

#include "batch_pool/batch_pool.h"
#include "expect.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static struct damacy_aabb
mk_aabb_2d(int64_t y0, int64_t y1, int64_t x0, int64_t x1)
{
  struct damacy_aabb a = { 0 };
  a.rank = 2;
  a.dims[0].beg = y0;
  a.dims[0].end = y1;
  a.dims[1].beg = x0;
  a.dims[1].end = x1;
  return a;
}

static int
test_compute_layout_simple(void)
{
  struct damacy_batch_pool pool = { 0 };
  struct damacy_aabb a = mk_aabb_2d(0, 32, 0, 64);
  // batch_size=8, bpe=4 → 8 × 32 × 64 × 4 = 65536
  EXPECT(batch_pool_compute_layout(&pool, &a, 8, 4) == DAMACY_OK);
  EXPECT(pool.layout_set == 1);
  EXPECT(pool.allocated == 0);
  EXPECT(pool.rank == 3);
  EXPECT(pool.shape[0] == 8);
  EXPECT(pool.shape[1] == 32);
  EXPECT(pool.shape[2] == 64);
  EXPECT(pool.strides[2] == 1);
  EXPECT(pool.strides[1] == 64);
  EXPECT(pool.strides[0] == 32 * 64);
  EXPECT(pool.n_bytes == 8ull * 32 * 64 * 4);
  return 0;
}

static int
test_compute_layout_idempotent(void)
{
  struct damacy_batch_pool pool = { 0 };
  struct damacy_aabb a = mk_aabb_2d(0, 16, 0, 16);
  EXPECT(batch_pool_compute_layout(&pool, &a, 4, 2) == DAMACY_OK);
  uint64_t first_n = pool.n_bytes;
  // Second call with a different AABB must short-circuit (layout_set
  // pinning) so an in-flight pool's output extents stay stable.
  struct damacy_aabb a2 = mk_aabb_2d(0, 64, 0, 64);
  EXPECT(batch_pool_compute_layout(&pool, &a2, 4, 2) == DAMACY_OK);
  EXPECT(pool.n_bytes == first_n);
  return 0;
}

static int
test_compute_layout_rejects_zero_extent(void)
{
  struct damacy_batch_pool pool = { 0 };
  struct damacy_aabb a = mk_aabb_2d(8, 8, 0, 16); // y has zero extent
  EXPECT(batch_pool_compute_layout(&pool, &a, 4, 4) == DAMACY_INVAL);
  EXPECT(pool.layout_set == 0);
  return 0;
}

static int
test_shape_match(void)
{
  struct damacy_batch_pool pool = { 0 };
  struct damacy_aabb a = mk_aabb_2d(0, 32, 0, 64);
  EXPECT(batch_pool_compute_layout(&pool, &a, 8, 4) == DAMACY_OK);

  // Same extents, different origin → still matches (we compare extent only).
  struct damacy_aabb shifted = mk_aabb_2d(100, 132, 50, 114);
  EXPECT(sample_shape_matches_pool(&pool, &shifted) == 1);

  struct damacy_aabb wrong_size = mk_aabb_2d(0, 32, 0, 32);
  EXPECT(sample_shape_matches_pool(&pool, &wrong_size) == 0);

  struct damacy_aabb wrong_rank = { 0 };
  wrong_rank.rank = 1;
  wrong_rank.dims[0].beg = 0;
  wrong_rank.dims[0].end = 32;
  EXPECT(sample_shape_matches_pool(&pool, &wrong_rank) == 0);
  return 0;
}

static int
test_state_predicates(void)
{
  struct damacy_batch_pool pool = { 0 };
  // Both FREE
  EXPECT(find_free_batch_slot(&pool) == 0);
  EXPECT(find_oldest_ready_slot(&pool) == -1);
  EXPECT(find_oldest_filling_slot(&pool) == -1);
  EXPECT(find_filling_slot_with_work(&pool) == -1);
  EXPECT(any_batch_in_flight(&pool) == 0);

  // slot[0] FILLING with work, slot[1] READY (newer batch_id)
  pool.slots[0].state = BATCH_FILLING;
  pool.slots[0].batch_id = 10;
  pool.slots[0].n_chunks = 4;
  pool.slots[0].n_chunks_dispatched = 2;
  pool.slots[1].state = BATCH_READY;
  pool.slots[1].batch_id = 11;

  EXPECT(find_free_batch_slot(&pool) == -1);
  EXPECT(find_oldest_ready_slot(&pool) == 1);
  EXPECT(find_oldest_filling_slot(&pool) == 0);
  EXPECT(find_filling_slot_with_work(&pool) == 0);
  EXPECT(any_batch_in_flight(&pool) == 1);

  // FILLING but fully dispatched → not "with work"
  pool.slots[0].n_chunks_dispatched = 4;
  EXPECT(find_filling_slot_with_work(&pool) == -1);
  EXPECT(find_oldest_filling_slot(&pool) == 0); // still filling

  // Two READY: oldest by batch_id wins
  pool.slots[0].state = BATCH_READY;
  pool.slots[0].batch_id = 9;
  pool.slots[1].batch_id = 10;
  EXPECT(find_oldest_ready_slot(&pool) == 0);
  return 0;
}

int
main(void)
{
  RUN(test_compute_layout_simple);
  RUN(test_compute_layout_idempotent);
  RUN(test_compute_layout_rejects_zero_extent);
  RUN(test_shape_match);
  RUN(test_state_predicates);
  printf("all batch_pool tests passed\n");
  return 0;
}
