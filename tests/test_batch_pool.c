// Unit tests for the host-side parts of src/batch_pool/.
//
//   test_compute_layout_simple — shape/strides/n_bytes for a 2-D shape
//   test_compute_layout_idempotent — second call short-circuits
//   test_compute_layout_rejects_zero_extent — zero/negative dim rejected
//   test_state_predicates      — find_*_slot scan in batch_id order
//   test_completion_accounting — rendering → ready on completed chunks
//   test_reset_for_reuse       — reset clears caller-visible lifetime state
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
  EXPECT(find_oldest_rendering_slot(&pool) == -1);
  EXPECT(any_batch_in_flight(&pool) == 0);

  // slot[0] RENDERING, slot[1] READY (newer batch_id)
  pool.slots[0].state = BATCH_RENDERING;
  pool.slots[0].batch_id = 10;
  pool.slots[0].n_chunks = 4;
  pool.slots[1].state = BATCH_READY;
  pool.slots[1].batch_id = 11;

  EXPECT(find_free_batch_slot(&pool) == -1);
  EXPECT(find_oldest_ready_slot(&pool) == 1);
  EXPECT(find_oldest_rendering_slot(&pool) == 0);
  EXPECT(any_batch_in_flight(&pool) == 1);

  // Two READY: oldest by batch_id wins
  pool.slots[0].state = BATCH_READY;
  pool.slots[0].batch_id = 9;
  pool.slots[1].batch_id = 10;
  EXPECT(find_oldest_ready_slot(&pool) == 0);
  return 0;
}

static int
test_completion_accounting(void)
{
  struct damacy_batch_slot slot = {
    .state = BATCH_RENDERING,
    .n_chunks = 5,
    .chunks_remaining = 5,
  };
  batch_slot_consume_chunks(&slot, 2);
  EXPECT(slot.state == BATCH_RENDERING);
  EXPECT(slot.chunks_remaining == 3);
  batch_slot_consume_chunks(&slot, 3);
  EXPECT(slot.state == BATCH_READY);
  EXPECT(slot.chunks_remaining == 0);
  return 0;
}

static int
test_reset_for_reuse(void)
{
  struct damacy_batch_slot slot = {
    .state = BATCH_HELD,
    .batch_id = 42,
    .sample_seq_begin = 7,
    .n_samples = 3,
    .n_chunks = 4,
    .chunks_remaining = 1,
    .planning_close_batch = 1,
    .deferred_release_pending = 1,
  };
  batch_slot_reset_for_reuse(&slot);
  EXPECT(slot.state == BATCH_FREE);
  EXPECT(slot.batch_id == 0);
  EXPECT(slot.sample_seq_begin == 0);
  EXPECT(slot.n_samples == 0);
  EXPECT(slot.n_chunks == 0);
  EXPECT(slot.chunks_remaining == 0);
  EXPECT(slot.planning_close_batch == 0);
  EXPECT(slot.deferred_release_pending == 0);
  return 0;
}

int
main(void)
{
  RUN(test_compute_layout_simple);
  RUN(test_compute_layout_idempotent);
  RUN(test_compute_layout_rejects_zero_extent);
  RUN(test_state_predicates);
  RUN(test_completion_accounting);
  RUN(test_reset_for_reuse);
  printf("all batch_pool tests passed\n");
  return 0;
}
