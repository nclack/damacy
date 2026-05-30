#include "damacy.h"

#include "damacy_config.h"
#include "damacy_internal.h"
#include "damacy_stats.h"
#include "platform/platform.h"
#include "util/cuda_check.h"
#include "util/prelude.h"

#include <cuda.h>
#include <stdlib.h>

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

static void
free_stage_sample(struct planner_sample* s)
{
  if (!s)
    return;
  free(s->h_shards);
  free((char*)s->uri);
  *s = (struct planner_sample){ 0 };
}

static void
free_slot_stage_samples(struct damacy_batch_slot* slot)
{
  if (!slot || !slot->stage_samples)
    return;
  for (uint32_t i = 0; i < slot->n_samples; ++i)
    free_stage_sample(&slot->stage_samples[i]);
}

enum damacy_status
plan_reserve(struct damacy* self,
             uint16_t slot_idx,
             struct prefetcher_ready* ready,
             uint32_t n_ready,
             int close_batch)
{
  if (n_ready == 0)
    return DAMACY_OK;
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  if (slot->state != BATCH_FREE && slot->state != BATCH_OPEN)
    return DAMACY_INVAL;
  uint32_t begin = slot->n_samples;
  if (slot->state == BATCH_FREE) {
    if (begin != 0)
      return DAMACY_INVAL;
    slot->batch_id = self->next_batch_id;
    slot->sample_seq_begin = ready[0].sample_seq;
  } else if (slot->batch_id != self->next_batch_id) {
    return DAMACY_INVAL;
  }
  if (begin + n_ready > self->cfg.samples_per_batch)
    return DAMACY_INVAL;

  for (uint32_t i = 0; i < n_ready; ++i) {
    // advance_from_shard absorbs NOTFOUND at the prefetcher level; a
    // slot-level ERROR here is always a real error.
    if (ready[i].result == PREFETCHER_RESULT_ERROR) {
      enum damacy_status es = ready[i].err_code
                                ? (enum damacy_status)ready[i].err_code
                                : DAMACY_INVAL;
      self->failed_status = es;
      return es;
    }
    if (ready[i].sample_seq != slot->sample_seq_begin + begin + i)
      return DAMACY_INVAL;
  }

  enum damacy_status status = batch_pool_allocate(self);
  if (status != DAMACY_OK) {
    self->failed_status = status;
    return status;
  }
  for (uint32_t i = 0; i < n_ready; ++i) {
    struct planner_sample* dst = &slot->stage_samples[begin + i];
    dst->uri = ready[i].uri;
    dst->aabb = ready[i].aabb;
    dst->h_meta = ready[i].h_meta;
    dst->h_shards = ready[i].h_shards;
    dst->n_shards = ready[i].n_shards;
    dst->h_layout = ready[i].h_layout;
    ready[i].uri = NULL;
    ready[i].h_shards = NULL;
    ready[i].n_shards = 0;
  }
  slot->n_samples = begin + n_ready;
  slot->planning_close_batch = close_batch;
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
                                           slot->stage_samples,
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
  if (run_status != DAMACY_OK) {
    free_slot_stage_samples(slot);
    slot->state = BATCH_FREE;
    slot->n_samples = 0;
    slot->sample_seq_begin = 0;
    slot->planning_close_batch = 0;
    slot->deferred_release_pending = 0;
    self->failed_status = run_status;
    if (changed)
      *changed = 1;
    return run_status;
  }
  slot->n_chunks_dispatched = 0;
  slot->n_groups_dispatched = 0;
  slot->chunks_remaining = (int32_t)slot->n_chunks;
  if (!slot->planning_close_batch) {
    slot->state = BATCH_OPEN;
    if (changed)
      *changed = 1;
    return DAMACY_OK;
  }

  free_slot_stage_samples(slot);
  prefetcher_advance_watermark(self->prefetcher,
                               slot->sample_seq_begin + slot->n_samples);
  self->next_batch_id++;
  slot->planning_close_batch = 0;
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
      slot->n_samples = 0;
      slot->sample_seq_begin = 0;
      self->failed_status = DAMACY_CUDA;
      return DAMACY_CUDA;
    }
    slot->state = BATCH_READY;
  }
  // Non-degenerate path doesn't clear the flag: assemble on FIFO stream_post
  // already inherits the cuStreamWaitEvent, so host-side consumption is moot.
  return DAMACY_OK;
}

static uint32_t
planning_capacity_locked(struct damacy* self)
{
  if (any_batch_planning(&self->batch_pool))
    return 0;
  uint32_t cap = 0;
  int open_slot = find_open_batch_slot(&self->batch_pool);
  if (open_slot >= 0) {
    struct damacy_batch_slot* slot = &self->batch_pool.slots[open_slot];
    if (slot->n_samples < self->cfg.samples_per_batch)
      cap += self->cfg.samples_per_batch - slot->n_samples;
  }
  for (int s = 0; s < 2; ++s) {
    if (self->batch_pool.slots[s].state == BATCH_FREE)
      cap += self->cfg.samples_per_batch;
  }
  uint32_t staging_cap = 2u * self->cfg.samples_per_batch;
  return cap < staging_cap ? cap : staging_cap;
}

static int
next_plan_slot_locked(struct damacy* self)
{
  int open_slot = find_open_batch_slot(&self->batch_pool);
  if (open_slot >= 0)
    return open_slot;
  return find_free_batch_slot(&self->batch_pool);
}

static void
free_ready_range(struct prefetcher_ready* ready, uint32_t n)
{
  for (uint32_t i = 0; i < n; ++i)
    prefetcher_ready_free(&ready[i]);
}

enum damacy_status
plan_ready_prefetch(struct damacy* self, int close_partial, int* changed)
{
  enum damacy_status status = DAMACY_OK;
  for (;;) {
    if (close_partial && prefetcher_ready_prefix_count(self->prefetcher) == 0) {
      int open_slot = find_open_batch_slot(&self->batch_pool);
      if (open_slot >= 0) {
        struct damacy_batch_slot* slot = &self->batch_pool.slots[open_slot];
        int truncated = slot->n_samples < self->cfg.samples_per_batch;
        slot->planning_close_batch = 1;
        slot->state = BATCH_PLANNING;
        scheduler_unlock(self->sched);
        float plan_ms = 0.f;
        enum damacy_status rs = plan_run(self, (uint16_t)open_slot, &plan_ms);
        scheduler_lock(self->sched);
        status = plan_commit(self, (uint16_t)open_slot, rs, plan_ms, changed);
        if (status == DAMACY_OK && truncated)
          self->stats.batches_truncated++;
        return status;
      }
    }

    uint32_t cap = planning_capacity_locked(self);
    if (cap == 0)
      return DAMACY_OK;

    struct prefetcher_wave_ticket ticket = { 0 };
    if (!prefetcher_take_ready_wave(
          self->prefetcher, cap, &ticket, self->staging)) {
      return DAMACY_OK;
    }

    uint32_t cursor = 0;
    while (cursor < ticket.n_samples) {
      int slot_idx = next_plan_slot_locked(self);
      if (slot_idx < 0) {
        free_ready_range(&self->staging[cursor], ticket.n_samples - cursor);
        return DAMACY_INVAL;
      }

      struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
      uint32_t begin = slot->n_samples;
      uint32_t room = self->cfg.samples_per_batch - begin;
      uint32_t remaining = ticket.n_samples - cursor;
      uint32_t n = remaining < room ? remaining : room;
      int close_batch = (begin + n == self->cfg.samples_per_batch) ||
                        (close_partial && ticket.n_samples < cap &&
                         cursor + n == ticket.n_samples);
      int truncated = close_batch && begin + n < self->cfg.samples_per_batch;

      status = plan_reserve(
        self, (uint16_t)slot_idx, &self->staging[cursor], n, close_batch);
      if (status != DAMACY_OK) {
        free_ready_range(&self->staging[cursor], ticket.n_samples - cursor);
        return status;
      }
      free_ready_range(&self->staging[cursor], n);

      scheduler_unlock(self->sched);
      float plan_ms = 0.f;
      enum damacy_status rs = plan_run(self, (uint16_t)slot_idx, &plan_ms);
      scheduler_lock(self->sched);
      status = plan_commit(self, (uint16_t)slot_idx, rs, plan_ms, changed);
      if (status == DAMACY_OK && truncated)
        self->stats.batches_truncated++;
      if (status != DAMACY_OK) {
        free_ready_range(&self->staging[cursor + n],
                         ticket.n_samples - cursor - n);
        return status;
      }
      cursor += n;
    }

    if (ticket.n_samples < cap)
      return DAMACY_OK;
  }
}
