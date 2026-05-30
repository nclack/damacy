#include "damacy.h"

#include "damacy_config.h"
#include "damacy_internal.h"
#include "damacy_stats.h"
#include "platform/platform.h"
#include "util/cuda_check.h"
#include "util/prelude.h"

#include <cuda.h>

// Lazy batch-output pool sizing + GPU-budget enforcement. Geometry is
// fixed by cfg->sample_shape at create-time; this only allocates the
// device buffers on first push. Idempotent.
static enum damacy_status
batch_pool_allocate(struct damacy* self)
{
  struct damacy_batch_pool* pool = &self->batch_pool;
  if (pool->allocated)
    return DAMACY_OK;

  enum damacy_status s =
    batch_pool_compute_layout(pool,
                              self->cfg.sample_shape,
                              self->cfg.sample_rank,
                              self->cfg.samples_per_batch,
                              damacy_dtype_bpe(self->cfg.dtype));
  if (s != DAMACY_OK)
    return s;

  const uint64_t need = 2ull * pool->n_bytes;
  s = gpu_budget_try_commit(self->budget, need, "batch-output pool");
  if (s != DAMACY_OK)
    return s;

  s = batch_pool_alloc_dev(pool);
  if (s != DAMACY_OK) {
    gpu_budget_release(self->budget, need);
    return s;
  }
  return DAMACY_OK;
}

// --- plan: reserve [locked] → run [unlocked] → commit [locked] -------------
// run does the planner CPU work + sample_plans H2D off the
// scheduler_lock. The slot sits in BATCH_PLANNING for the window so
// pop (any_batch_in_flight) and flush (any_batch_planning) wait.

enum damacy_status
plan_reserve(struct damacy* self,
             uint16_t slot_idx,
             struct prefetcher_wave_ticket ticket)
{
  if (ticket.n_samples == 0)
    return DAMACY_OK;
  if (ticket.batch_id != self->next_batch_id)
    return DAMACY_INVAL;
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  if (slot->state != BATCH_FREE)
    return DAMACY_INVAL;

  struct prefetcher_wave_ticket consumed = { 0 };
  if (!prefetcher_take_wave(self->prefetcher,
                            ticket.batch_id,
                            ticket.n_samples,
                            &consumed,
                            self->staging)) {
    return DAMACY_AGAIN;
  }
  if (consumed.n_samples != ticket.n_samples ||
      consumed.batch_id != ticket.batch_id) {
    for (uint32_t i = 0; i < consumed.n_samples; ++i)
      prefetcher_ready_free(&self->staging[i]);
    return DAMACY_INVAL;
  }

  for (uint32_t i = 0; i < ticket.n_samples; ++i) {
    // advance_from_shard absorbs NOTFOUND at the prefetcher level; a
    // slot-level ERROR here is always a real error.
    if (self->staging[i].result == PREFETCHER_RESULT_ERROR) {
      enum damacy_status es = self->staging[i].err_code
                                ? (enum damacy_status)self->staging[i].err_code
                                : DAMACY_INVAL;
      for (uint32_t j = 0; j < ticket.n_samples; ++j)
        prefetcher_ready_free(&self->staging[j]);
      self->failed_status = es;
      return es;
    }
  }

  enum damacy_status status = batch_pool_allocate(self);
  if (status != DAMACY_OK) {
    for (uint32_t i = 0; i < ticket.n_samples; ++i)
      prefetcher_ready_free(&self->staging[i]);
    self->failed_status = status;
    return status;
  }
  for (uint32_t i = 0; i < ticket.n_samples; ++i) {
    // Steal uri + h_shards into batch_stage; plan_commit frees the rest
    // of staging[i] (the bare handles are POD and safe to copy).
    self->batch_stage[i].uri = self->staging[i].uri;
    self->batch_stage[i].aabb = self->staging[i].aabb;
    self->batch_stage[i].h_meta = self->staging[i].h_meta;
    self->batch_stage[i].h_shards = self->staging[i].h_shards;
    self->batch_stage[i].n_shards = self->staging[i].n_shards;
    self->batch_stage[i].h_layout = self->staging[i].h_layout;
    self->staging[i].uri = NULL;
    self->staging[i].h_shards = NULL;
    self->staging[i].n_shards = 0;
  }
  slot->n_samples = ticket.n_samples;
  slot->state = BATCH_PLANNING;
  return DAMACY_OK;
}

// Returns elapsed ms via *out_elapsed_ms so the metric can be recorded
// in plan_commit, which runs under scheduler_lock (stats are read by
// damacy_stats_get under the same lock).
enum damacy_status
plan_run(struct damacy* self, uint16_t slot_idx, float* out_elapsed_ms)
{
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  CHECK(InvalidArg, slot->state == BATCH_PLANNING);
  struct planner_output plan_out = {
    .read_ops = slot->read_ops,
    .read_ops_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .chunk_plans = slot->chunk_plans,
    .chunk_plans_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .sample_plans = slot->sample_plans,
    .sample_plans_cap = self->cfg.samples_per_batch,
    .read_op_groups = slot->read_op_groups,
    .read_op_groups_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .paths = &slot->paths,
  };
  struct platform_clock plan_clock = { 0 };
  platform_toc(&plan_clock);
  enum damacy_status status = planner_plan(self->planner,
                                           self->batch_stage,
                                           slot->n_samples,
                                           slot_idx,
                                           self->batch_pool.strides,
                                           self->batch_pool.rank,
                                           &plan_out);
  *out_elapsed_ms = platform_toc(&plan_clock) * 1000.0f;
  if (status != DAMACY_OK)
    return status;
  slot->n_chunks = plan_out.n_chunk_plans;
  slot->n_chunks_to_load = plan_out.n_chunks_to_load;
  slot->n_loads_issued = plan_out.n_loads_issued;
  slot->n_sample_plans = plan_out.n_sample_plans;
  slot->n_read_op_groups = plan_out.n_read_op_groups;
  if (plan_out.n_sample_plans > 0) {
    if (cuMemcpyHtoD(CUDPTR(slot->d_sample_plans),
                     slot->sample_plans,
                     (size_t)plan_out.n_sample_plans *
                       sizeof(struct sample_plan)) != CUDA_SUCCESS)
      return DAMACY_CUDA;
  }
  return DAMACY_OK;
InvalidArg:
  return DAMACY_INVAL;
}

// *changed (nullable; flush passes NULL): OR-set on every BATCH transition.
enum damacy_status
plan_commit(struct damacy* self,
            uint16_t slot_idx,
            enum damacy_status run_status,
            float elapsed_ms,
            int* changed)
{
  metric_record(&self->stats.plan, elapsed_ms, 0, 0);
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  for (uint32_t i = 0; i < slot->n_samples; ++i) {
    prefetcher_ready_free(&self->staging[i]);
    // plan_reserve stole uri + h_shards from staging[i] into batch_stage[i];
    // planner is done with them by now, free here to close the leak.
    free(self->batch_stage[i].h_shards);
    free((char*)self->batch_stage[i].uri);
    self->batch_stage[i] = (struct planner_sample){ 0 };
  }
  if (run_status != DAMACY_OK) {
    slot->state = BATCH_FREE;
    slot->n_samples = 0;
    slot->deferred_release_pending = 0;
    self->failed_status = run_status;
    if (changed)
      *changed = 1;
    return run_status;
  }
  slot->n_chunks_dispatched = 0;
  slot->n_groups_dispatched = 0;
  slot->chunks_remaining = (int32_t)slot->n_chunks;
  slot->batch_id = self->next_batch_id++;
  prefetcher_advance_watermark(self->prefetcher, self->next_batch_id);
  slot->state = BATCH_FILLING;
  self->stats.chunks_planned += slot->n_chunks;
  self->stats.chunks_to_load += slot->n_chunks_to_load;
  self->stats.reads_issued += slot->n_loads_issued;
  if (changed)
    *changed = 1;

  if (slot->n_chunks == 0) {
    // Degenerate batch: zero the output and skip to READY. cuMemsetD8
    // runs on the legacy null stream; if a deferred release wait is
    // pending on stream_post, sync it first so the memset can't race
    // a still-in-flight consumer read.
    if (slot->deferred_release_pending) {
      cuStreamSynchronize(self->wave_pool.stream_post);
      slot->deferred_release_pending = 0;
    }
    if (cuMemsetD8(CUDPTR(slot->dev_ptr), 0, self->batch_pool.n_bytes) !=
        CUDA_SUCCESS) {
      slot->state = BATCH_FREE;
      self->failed_status = DAMACY_CUDA;
      return DAMACY_CUDA;
    }
    slot->state = BATCH_READY;
  }
  // Non-degenerate path doesn't clear the flag: assemble on FIFO stream_post
  // already inherits the cuStreamWaitEvent, so host-side consumption is moot.
  return DAMACY_OK;
}
