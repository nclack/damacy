#include "batch_pool.h"

#include "log/log.h"
#include "planner/planner.h"
#include "util/cuda_check.h" // CU + CUDPTR
#include "util/prelude.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

int
batch_slot_init(struct damacy_batch_slot* slot, uint32_t samples_per_batch_cap)
{
  memset(slot, 0, sizeof(*slot));
  slot->stage_samples = (struct planner_sample*)calloc(
    samples_per_batch_cap, sizeof(struct planner_sample));
  CHECK(Error, slot->stage_samples);
  return 0;
Error:
  return 1;
}

void
batch_slot_destroy(struct damacy_batch_slot* slot, int cuda_skip)
{
  if (!slot)
    return;
  // dev_ptr is owned by the pool's lazy allocation (batch_pool_alloc_dev)
  // and freed in batch_pool_destroy — leave it alone here.
  if (slot->stage_samples) {
    for (uint32_t i = 0; i < slot->n_samples; ++i) {
      free(slot->stage_samples[i].h_shards);
      free((char*)slot->stage_samples[i].uri);
    }
  }
  free(slot->stage_samples);
  (void)cuda_skip;
  memset(slot, 0, sizeof(*slot));
}

void
batch_pool_destroy(struct damacy_batch_pool* pool, int cuda_skip)
{
  if (!pool)
    return;
  for (int s = 0; s < DAMACY_N_BATCH_SLOTS; ++s) {
    if (!cuda_skip && pool->slots[s].dev_ptr)
      cuMemFree(CUDPTR(pool->slots[s].dev_ptr));
    batch_slot_destroy(&pool->slots[s], cuda_skip);
  }
  memset(pool, 0, sizeof(*pool));
}

enum damacy_status
batch_pool_compute_layout(struct damacy_batch_pool* pool,
                          const int64_t* sample_shape,
                          uint8_t sample_rank,
                          uint32_t samples_per_batch,
                          uint32_t bpe)
{
  if (pool->layout_set)
    return DAMACY_OK;

  CHECK_SILENT(Invalid, sample_shape);
  CHECK_SILENT(Invalid, sample_rank > 0);
  uint8_t full_rank = (uint8_t)(sample_rank + 1);
  CHECK_SILENT(Rank, full_rank <= DAMACY_MAX_RANK + 1);

  int64_t spatial_volume = 1;
  pool->shape[0] = (int64_t)samples_per_batch;
  for (uint8_t d = 0; d < sample_rank; ++d) {
    CHECK_SILENT(Invalid, sample_shape[d] > 0);
    pool->shape[d + 1] = sample_shape[d];
    spatial_volume *= sample_shape[d];
  }
  pool->rank = full_rank;
  pool->strides[full_rank - 1] = 1;
  for (int d = (int)full_rank - 2; d >= 0; --d)
    pool->strides[d] = pool->strides[d + 1] * pool->shape[d + 1];
  pool->n_bytes =
    (uint64_t)samples_per_batch * (uint64_t)spatial_volume * (uint64_t)bpe;
  pool->layout_set = 1;
  return DAMACY_OK;
Rank:
  return DAMACY_RANK;
Invalid:
  return DAMACY_INVAL;
}

enum damacy_status
batch_pool_alloc_dev(struct damacy_batch_pool* pool)
{
  if (pool->allocated)
    return DAMACY_OK;
  if (!pool->layout_set)
    return DAMACY_INVAL;
  for (int s = 0; s < DAMACY_N_BATCH_SLOTS; ++s) {
    CUdeviceptr dptr = 0;
    if (cuMemAlloc(&dptr, pool->n_bytes) != CUDA_SUCCESS) {
      log_error("batch_pool: cuMemAlloc(%llu) failed",
                (unsigned long long)pool->n_bytes);
      return DAMACY_CUDA;
    }
    pool->slots[s].dev_ptr = (void*)(uintptr_t)dptr;
  }
  pool->allocated = 1;
  return DAMACY_OK;
}

int
find_free_batch_slot(const struct damacy_batch_pool* pool)
{
  for (int s = 0; s < DAMACY_N_BATCH_SLOTS; ++s)
    if (pool->slots[s].state == BATCH_FREE)
      return s;
  return -1;
}

int
find_accumulating_batch_slot(const struct damacy_batch_pool* pool)
{
  int best = -1;
  uint64_t best_id = UINT64_MAX;
  for (int s = 0; s < DAMACY_N_BATCH_SLOTS; ++s) {
    if (pool->slots[s].state == BATCH_ACCUMULATING &&
        pool->slots[s].batch_id < best_id) {
      best = s;
      best_id = pool->slots[s].batch_id;
    }
  }
  return best;
}

int
find_oldest_ready_slot(const struct damacy_batch_pool* pool)
{
  int best = -1;
  uint64_t best_id = UINT64_MAX;
  for (int s = 0; s < DAMACY_N_BATCH_SLOTS; ++s) {
    if (pool->slots[s].state == BATCH_READY &&
        pool->slots[s].batch_id < best_id) {
      best = s;
      best_id = pool->slots[s].batch_id;
    }
  }
  return best;
}

int
find_oldest_rendering_slot(const struct damacy_batch_pool* pool)
{
  int best = -1;
  uint64_t best_id = UINT64_MAX;
  for (int s = 0; s < DAMACY_N_BATCH_SLOTS; ++s) {
    if (pool->slots[s].state == BATCH_RENDERING &&
        pool->slots[s].batch_id < best_id) {
      best = s;
      best_id = pool->slots[s].batch_id;
    }
  }
  return best;
}

int
any_batch_in_flight(const struct damacy_batch_pool* pool)
{
  for (int s = 0; s < DAMACY_N_BATCH_SLOTS; ++s) {
    enum batch_slot_state st = pool->slots[s].state;
    if (st == BATCH_ACCUMULATING || st == BATCH_PLANNING ||
        st == BATCH_RENDERING || st == BATCH_READY || st == BATCH_HELD)
      return 1;
  }
  return 0;
}

int
any_batch_planning(const struct damacy_batch_pool* pool)
{
  for (int s = 0; s < DAMACY_N_BATCH_SLOTS; ++s)
    if (pool->slots[s].state == BATCH_PLANNING)
      return 1;
  return 0;
}

void
batch_slot_consume_chunks(struct damacy_batch_slot* slot, uint32_t n_consumed)
{
  if (!slot || slot->state != BATCH_RENDERING)
    return;
  slot->chunks_remaining -= (int32_t)n_consumed;
  if (slot->chunks_remaining <= 0) {
    slot->chunks_remaining = 0;
    slot->state = BATCH_READY;
  }
}

void
batch_slot_reset_for_reuse(struct damacy_batch_slot* slot)
{
  if (!slot)
    return;
  slot->state = BATCH_FREE;
  slot->batch_id = 0;
  slot->sample_seq_begin = 0;
  slot->n_samples = 0;
  slot->n_chunks = 0;
  slot->chunks_remaining = 0;
  slot->planning_close_batch = 0;
  slot->deferred_release_pending = 0;
}
