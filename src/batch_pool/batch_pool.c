#include "batch_pool.h"

#include "log/log.h"
#include "planner/planner.h"
#include "util/prelude.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

#define CUDPTR(p) ((CUdeviceptr)(uintptr_t)(p))

int
batch_slot_init(struct damacy_batch_slot* slot, uint32_t batch_size_cap)
{
  slot->read_ops = (struct read_op*)calloc(DAMACY_MAX_CHUNKS_PER_BATCH,
                                           sizeof(struct read_op));
  CHECK(Error, slot->read_ops);
  slot->chunk_plans = (struct chunk_plan*)calloc(DAMACY_MAX_CHUNKS_PER_BATCH,
                                                 sizeof(struct chunk_plan));
  CHECK(Error, slot->chunk_plans);
  slot->sample_plans =
    (struct sample_plan*)calloc(batch_size_cap, sizeof(struct sample_plan));
  CHECK(Error, slot->sample_plans);
  CUdeviceptr dptr = 0;
  if (cuMemAlloc(&dptr, (size_t)batch_size_cap * sizeof(struct sample_plan)) !=
      CUDA_SUCCESS)
    goto Error;
  slot->d_sample_plans = (void*)(uintptr_t)dptr;
  return 0;
Error:
  return 1;
}

void
batch_slot_destroy(struct damacy_batch_slot* slot, int cuda_skip)
{
  if (!slot)
    return;
  // dev_ptr is owned by the pool's lazy allocation; freed there.
  free(slot->read_ops);
  free(slot->chunk_plans);
  free(slot->sample_plans);
  if (!cuda_skip && slot->d_sample_plans)
    cuMemFree(CUDPTR(slot->d_sample_plans));
  memset(slot, 0, sizeof(*slot));
}

void
batch_pool_destroy(struct damacy_batch_pool* pool, int cuda_skip)
{
  if (!pool)
    return;
  for (int s = 0; s < 2; ++s) {
    if (!cuda_skip && pool->slots[s].dev_ptr)
      cuMemFree(CUDPTR(pool->slots[s].dev_ptr));
    batch_slot_destroy(&pool->slots[s], cuda_skip);
  }
  memset(pool, 0, sizeof(*pool));
}

enum damacy_status
batch_pool_compute_layout(struct damacy_batch_pool* pool,
                          const struct damacy_aabb* sample_aabb,
                          uint32_t batch_size,
                          uint32_t bpe)
{
  if (pool->allocated)
    return DAMACY_OK;

  uint8_t spatial_rank = sample_aabb->rank;
  uint8_t full_rank = (uint8_t)(spatial_rank + 1);
  CHECK_SILENT(Rank, full_rank <= DAMACY_MAX_RANK + 1);

  int64_t spatial_volume = 1;
  pool->shape[0] = (int64_t)batch_size;
  for (uint8_t d = 0; d < spatial_rank; ++d) {
    int64_t extent = sample_aabb->dims[d].end - sample_aabb->dims[d].beg;
    CHECK_SILENT(Invalid, extent > 0);
    pool->shape[d + 1] = extent;
    spatial_volume *= extent;
  }
  pool->rank = full_rank;
  pool->strides[full_rank - 1] = 1;
  for (int d = (int)full_rank - 2; d >= 0; --d)
    pool->strides[d] = pool->strides[d + 1] * pool->shape[d + 1];
  pool->n_bytes =
    (uint64_t)batch_size * (uint64_t)spatial_volume * (uint64_t)bpe;
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
  for (int s = 0; s < 2; ++s) {
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
sample_shape_matches_pool(const struct damacy_batch_pool* pool,
                          const struct damacy_aabb* aabb)
{
  uint8_t spatial_rank = (uint8_t)(pool->rank - 1);
  if (aabb->rank != spatial_rank)
    return 0;
  for (uint8_t d = 0; d < spatial_rank; ++d) {
    int64_t extent = aabb->dims[d].end - aabb->dims[d].beg;
    if (extent != pool->shape[d + 1])
      return 0;
  }
  return 1;
}

int
find_free_batch_slot(const struct damacy_batch_pool* pool)
{
  for (int s = 0; s < 2; ++s)
    if (pool->slots[s].state == BATCH_FREE)
      return s;
  return -1;
}

int
find_oldest_ready_slot(const struct damacy_batch_pool* pool)
{
  int best = -1;
  uint64_t best_id = UINT64_MAX;
  for (int s = 0; s < 2; ++s) {
    if (pool->slots[s].state == BATCH_READY &&
        pool->slots[s].batch_id < best_id) {
      best = s;
      best_id = pool->slots[s].batch_id;
    }
  }
  return best;
}

int
find_oldest_filling_slot(const struct damacy_batch_pool* pool)
{
  int best = -1;
  uint64_t best_id = UINT64_MAX;
  for (int s = 0; s < 2; ++s) {
    if (pool->slots[s].state == BATCH_FILLING &&
        pool->slots[s].batch_id < best_id) {
      best = s;
      best_id = pool->slots[s].batch_id;
    }
  }
  return best;
}

int
find_filling_slot_with_work(const struct damacy_batch_pool* pool)
{
  int best = -1;
  uint64_t best_id = UINT64_MAX;
  for (int s = 0; s < 2; ++s) {
    const struct damacy_batch_slot* slot = &pool->slots[s];
    if (slot->state == BATCH_FILLING &&
        slot->n_chunks_dispatched < slot->n_chunks &&
        slot->batch_id < best_id) {
      best = s;
      best_id = slot->batch_id;
    }
  }
  return best;
}

int
any_batch_in_flight(const struct damacy_batch_pool* pool)
{
  for (int s = 0; s < 2; ++s) {
    enum batch_slot_state st = pool->slots[s].state;
    if (st == BATCH_FILLING || st == BATCH_READY || st == BATCH_HELD)
      return 1;
  }
  return 0;
}
