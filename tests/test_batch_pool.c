// Unit tests for the host-side parts of src/batch_pool/.
//
//   test_compute_layout_simple — shape/strides/n_bytes for a 2-D shape
//   test_compute_layout_idempotent — second call short-circuits
//   test_compute_layout_rejects_zero_extent — zero/negative dim rejected
//   test_state_predicates      — find_*_slot scan in batch_id order
//
// We don't exercise batch_slot_init / batch_pool_alloc_dev: those need
// a CUcontext and would belong in a CUDA-gated test.

#include "batch_pool/batch_pool.h"
#include "expect.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int
test_compute_layout_simple(void)
{
  struct damacy_batch_pool pool = { 0 };
  const int64_t shape[2] = { 32, 64 };
  // samples_per_batch=8, bpe=4 → 8 × 32 × 64 × 4 = 65536
  EXPECT(batch_pool_compute_layout(&pool, shape, 2, 8, 4) == DAMACY_OK);
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
  const int64_t shape[2] = { 16, 16 };
  EXPECT(batch_pool_compute_layout(&pool, shape, 2, 4, 2) == DAMACY_OK);
  uint64_t first_n = pool.n_bytes;
  // Second call with a different shape must short-circuit (layout_set
  // pinning) so an in-flight pool's output extents stay stable.
  const int64_t shape2[2] = { 64, 64 };
  EXPECT(batch_pool_compute_layout(&pool, shape2, 2, 4, 2) == DAMACY_OK);
  EXPECT(pool.n_bytes == first_n);
  return 0;
}

static int
test_compute_layout_rejects_zero_extent(void)
{
  struct damacy_batch_pool pool = { 0 };
  const int64_t shape[2] = { 0, 16 }; // y has zero extent
  EXPECT(batch_pool_compute_layout(&pool, shape, 2, 4, 4) == DAMACY_INVAL);
  EXPECT(pool.layout_set == 0);
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
  RUN(test_state_predicates);
  printf("all batch_pool tests passed\n");
  return 0;
}
