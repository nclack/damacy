// damacy: streaming loader. Two batch slots and two wave slots in flight.
// A worker thread (src/scheduler) drives the pipeline; the user-thread
// API (push/pop/release/flush) coordinates via scheduler_lock + the
// scheduler's condition variable. Background I/O on the io_queue rounds
// out the threading model.

#include "damacy.h"

#include "damacy_config.h"
#include "damacy_internal.h"
#include "damacy_stats.h"
#include "log/log.h"
#include "nvtx/nvtx.h"
#include "util/cuda_check.h"
#include "util/prelude.h"
#include "wave/wave_budget.h"
#include "zarr/zarr_metadata.h"

#include <stdlib.h>
#include <string.h>

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
                              self->cfg.batch_size,
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

// 1 if `aabb` extents (assumed same rank as cfg) match cfg->sample_shape.
static int
sample_aabb_extents_match_cfg(const struct damacy_config* cfg,
                              const struct damacy_aabb* aabb)
{
  for (uint8_t d = 0; d < cfg->sample_rank; ++d) {
    int64_t extent = aabb->dims[d].end - aabb->dims[d].beg;
    if (extent != cfg->sample_shape[d])
      return 0;
  }
  return 1;
}

// --- planning -------------------------------------------------------------

static enum damacy_status
push_one(struct damacy* self, const struct damacy_sample* sample)
{
  if (!sample->uri)
    return DAMACY_INVAL;
  if (sample->aabb.rank == 0 || sample->aabb.rank > DAMACY_MAX_RANK)
    return DAMACY_RANK;

  struct zarr_metadata meta;
  enum damacy_status ms =
    zarr_meta_cache_get(self->meta_cache, sample->uri, &meta);
  if (ms != DAMACY_OK)
    return ms;

  if (!cast_path_supported(self->cfg.dtype, meta.dtype))
    return DAMACY_DTYPE;
  if (sample->aabb.rank != meta.rank)
    return DAMACY_RANK;
  if (sample->aabb.rank != self->cfg.sample_rank)
    return DAMACY_RANK;
  if (!sample_aabb_extents_match_cfg(&self->cfg, &sample->aabb))
    return DAMACY_INVAL;

  if (lookahead_push(&self->lookahead, sample))
    return DAMACY_OOM;
  return DAMACY_OK;
}

// --- plan: reserve [locked] → run [unlocked] → commit [locked] -------------
// run does the planner CPU work + sample_plans H2D off the
// scheduler_lock. The slot sits in BATCH_PLANNING for the window so
// pop (any_batch_in_flight) and flush (any_batch_planning) wait.

static enum damacy_status
plan_reserve(struct damacy* self, uint16_t slot_idx, uint32_t n_samples)
{
  if (n_samples == 0)
    return DAMACY_OK;
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  if (slot->state != BATCH_FREE)
    return DAMACY_INVAL;

  lookahead_drain(&self->lookahead, self->batch_samples, n_samples);

  enum damacy_status status = batch_pool_allocate(self);
  if (status != DAMACY_OK) {
    for (uint32_t i = 0; i < n_samples; ++i)
      sample_slot_clear(&self->batch_samples[i]);
    self->failed_status = status;
    return status;
  }
  for (uint32_t i = 0; i < n_samples; ++i) {
    self->batch_stage[i].uri = self->batch_samples[i].uri;
    self->batch_stage[i].aabb = self->batch_samples[i].aabb;
  }
  slot->n_samples = n_samples;
  slot->state = BATCH_PLANNING;
  return DAMACY_OK;
}

// Returns elapsed ms via *out_elapsed_ms so the metric can be recorded
// in plan_commit, which runs under scheduler_lock (stats are read by
// damacy_stats_get under the same lock).
static enum damacy_status
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
    .sample_plans_cap = self->cfg.batch_size,
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
static enum damacy_status
plan_commit(struct damacy* self,
            uint16_t slot_idx,
            enum damacy_status run_status,
            float elapsed_ms,
            int* changed)
{
  metric_record(&self->stats.plan, elapsed_ms, 0, 0);
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  for (uint32_t i = 0; i < slot->n_samples; ++i)
    sample_slot_clear(&self->batch_samples[i]);
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
  slot->deferred_release_pending = 0;
  return DAMACY_OK;
}

// Drains lookahead-planned batches into free host_slab_slots, planning
// a fresh batch when no FILLING batch has chunks left to peel. Stops
// when there are no free slots, no batches with work, and no room to
// plan more.
static enum damacy_status
kick_peel_into_free_slots(struct damacy* self, int* changed)
{
  for (;;) {
    int target_slot = find_filling_slot_with_work(&self->batch_pool);
    if (target_slot < 0) {
      int free_slot = find_free_batch_slot(&self->batch_pool);
      if (free_slot < 0)
        break;
      if (self->lookahead.size < self->cfg.batch_size)
        break;
      enum damacy_status s =
        plan_reserve(self, (uint16_t)free_slot, self->cfg.batch_size);
      if (s != DAMACY_OK)
        return s;
      scheduler_unlock(self->sched);
      float plan_ms = 0.f;
      enum damacy_status rs = plan_run(self, (uint16_t)free_slot, &plan_ms);
      scheduler_lock(self->sched);
      s = plan_commit(self, (uint16_t)free_slot, rs, plan_ms, changed);
      if (s != DAMACY_OK)
        return s;
      continue;
    }

    enum damacy_status err = DAMACY_OK;
    struct wave_pool_peel_ticket t =
      wave_pool_peel_reserve(&self->wave_pool, (uint16_t)target_slot, &err);
    if (err != DAMACY_OK)
      return err;
    if (t.slot_idx < 0)
      break;
    damacy_nvtx_range_pushf("peel/slot%d", t.slot_idx);
    scheduler_unlock(self->sched);
    struct store_event ev = wave_pool_peel_submit(&self->wave_pool, &t);
    scheduler_lock(self->sched);
    enum damacy_status s =
      wave_pool_peel_commit(&self->wave_pool, &t, ev, changed);
    damacy_nvtx_range_pop();
    if (s != DAMACY_OK)
      return s;
    if (!any_slot_free(&self->wave_pool))
      break;
  }
  return DAMACY_OK;
}

// --- scheduler ------------------------------------------------------------

// One scheduler tick, under scheduler_lock. Lazy ctx push on first call.
// *changed contract (authoritative): every transition site
// (wave_pool_advance, plan_commit, wave_pool_peel_commit) OR-sets it on
// a real state transition; the worker broadcasts iff non-zero.
int
damacy_scheduler_step(void* arg)
{
  struct damacy* self = (struct damacy*)arg;
  if (!self->worker_ctx_pushed) {
    if (self->worker_ctx)
      cuCtxPushCurrent(self->worker_ctx);
    self->worker_ctx_pushed = 1;
  }
  self->stats.worker_steps++;
  // Wake any pop waiter so it can observe the latched error.
  if (self->failed_status != DAMACY_OK)
    return 1;

  int changed = 0;
  enum damacy_status r = wave_pool_advance(&self->wave_pool, &changed);
  if (r == DAMACY_OK && self->failed_status == DAMACY_OK)
    r = kick_peel_into_free_slots(self, &changed);
  if (r != DAMACY_OK && self->failed_status == DAMACY_OK) {
    self->failed_status = r;
    return 1;
  }
  return changed;
}

// --- push -----------------------------------------------------------------

struct damacy_push_result
damacy_push(struct damacy* self, struct damacy_sample_slice samples)
{
  struct damacy_push_result r = { .unconsumed = samples, .status = DAMACY_OK };
  if (!self) {
    r.status = DAMACY_INVAL;
    return r;
  }
  if (samples.beg > samples.end) {
    r.status = DAMACY_INVAL;
    return r;
  }
  // No ctx_guard: push touches no CUDA. The lock guards lookahead +
  // meta/shape checks against the worker's plan_into_slot drain.
  scheduler_lock(self->sched);
  if (self->failed_status != DAMACY_OK) {
    r.status = DAMACY_SHUTDOWN;
    goto Done;
  }
  for (const struct damacy_sample* s = samples.beg; s != samples.end; ++s) {
    if (self->lookahead.size == self->lookahead.cap) {
      r.unconsumed.beg = s;
      r.status = DAMACY_AGAIN;
      goto Done;
    }
    enum damacy_status ps = push_one(self, s);
    if (ps != DAMACY_OK) {
      r.unconsumed.beg = s;
      r.status = ps;
      goto Done;
    }
  }
  r.unconsumed.beg = samples.end;
Done:
  scheduler_unlock(self->sched);
  return r;
}

// --- pop ------------------------------------------------------------------

enum damacy_status
damacy_pop(struct damacy* self, struct damacy_batch** out)
{
  CHECK_SILENT(InvalidArg, self);
  CHECK_SILENT(InvalidArg, out);
  *out = NULL;

  // No ctx_guard: pop only touches batch-slot state. CUDA stays on the worker.
  damacy_nvtx_range_push("damacy_pop");
  enum damacy_status r;
  scheduler_lock(self->sched);
  for (;;) {
    if (self->failed_status != DAMACY_OK) {
      r = self->failed_status;
      goto Done;
    }
    int slot_idx = find_oldest_ready_slot(&self->batch_pool);
    if (slot_idx >= 0) {
      struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
      slot->state = BATCH_HELD;
      self->handle.slot_idx = (uint16_t)slot_idx;
      self->handle.batch_id = slot->batch_id;
      self->stats.batches_emitted++;
      *out = &self->handle;
      r = DAMACY_OK;
      goto Done;
    }
    if (!any_wave_in_flight(&self->wave_pool) &&
        !any_slot_in_flight(&self->wave_pool) &&
        !any_batch_in_flight(&self->batch_pool) &&
        self->lookahead.size < self->cfg.batch_size) {
      r = DAMACY_AGAIN;
      goto Done;
    }
    struct platform_clock wait_clock = { 0 };
    platform_toc(&wait_clock);
    SCHEDULER_WAIT_DIAG(self->sched, 5000);
    metric_record(
      &self->stats.pop_wait, platform_toc(&wait_clock) * 1000.0f, 0, 0);
  }

Done:
  scheduler_unlock(self->sched);
  damacy_nvtx_range_pop();
  return r;

InvalidArg:
  return DAMACY_INVAL;
}

void
damacy_release(struct damacy* self, struct damacy_batch* b)
{
  if (!self || !b)
    return;
  if (b != &self->handle) {
    log_warn("damacy_release: foreign handle (not the active batch)");
    return;
  }
  uint16_t s = b->slot_idx;
  if (s >= 2) {
    log_warn("damacy_release: slot_idx=%u out of range", (unsigned)s);
    return;
  }
  scheduler_lock(self->sched);
  if (self->batch_pool.slots[s].state != BATCH_HELD) {
    log_warn("damacy_release: slot %u not HELD (state=%d); double release?",
             (unsigned)s,
             (int)self->batch_pool.slots[s].state);
    scheduler_unlock(self->sched);
    return;
  }
  self->batch_pool.slots[s].state = BATCH_FREE;
  self->batch_pool.slots[s].n_chunks = 0;
  self->batch_pool.slots[s].n_chunks_dispatched = 0;
  self->batch_pool.slots[s].n_groups_dispatched = 0;
  self->batch_pool.slots[s].deferred_release_pending = 0;
  scheduler_unlock(self->sched);
}

enum damacy_status
damacy_release_event(struct damacy* self, struct damacy_batch* b, void* event)
{
  // NULL event → degenerate to the immediate-release path.
  if (!event) {
    damacy_release(self, b);
    return DAMACY_OK;
  }
  if (!self || !b)
    return DAMACY_INVAL;
  if (b != &self->handle) {
    log_warn("damacy_release_event: foreign handle (not the active batch)");
    return DAMACY_INVAL;
  }
  uint16_t s = b->slot_idx;
  if (s >= 2) {
    log_warn("damacy_release_event: slot_idx=%u out of range", (unsigned)s);
    return DAMACY_INVAL;
  }

  // Push the retained-primary context so cuStreamWaitEvent / cuEventRecord
  // land on the right device when the caller is on another thread.
  struct ctx_guard cg = { 0 };
  enum damacy_status r = ctx_guard_enter(self, &cg);
  if (r != DAMACY_OK)
    return r;

  scheduler_lock(self->sched);
  struct damacy_batch_slot* slot = &self->batch_pool.slots[s];
  if (slot->state != BATCH_HELD) {
    log_warn(
      "damacy_release_event: slot %u not HELD (state=%d); double release?",
      (unsigned)s,
      (int)slot->state);
    r = DAMACY_INVAL;
    goto Done;
  }

  // Defer reuse on stream_post (where assemble writes the slot's
  // dev_ptr). stream_post is FIFO, so any subsequent kick_assemble — for
  // either slot — picks up this wait. The flag is read by plan_into_slot
  // to host-sync stream_post before its sync cuMemsetD8 (which targets the
  // legacy null stream and would otherwise race).
  if (cuStreamWaitEvent(self->wave_pool.stream_post, (CUevent)event, 0) !=
      CUDA_SUCCESS) {
    // Deferred wait couldn't be installed; fall back to immediate release
    // so the slot doesn't leak (caller would block forever in pop).
    slot->state = BATCH_FREE;
    slot->n_chunks = 0;
    slot->n_chunks_dispatched = 0;
    slot->n_groups_dispatched = 0;
    slot->deferred_release_pending = 0;
    r = DAMACY_CUDA;
    goto Done;
  }
  slot->deferred_release_pending = 1;

  slot->state = BATCH_FREE;
  slot->n_chunks = 0;
  slot->n_chunks_dispatched = 0;
  slot->n_groups_dispatched = 0;
  r = DAMACY_OK;

Done:
  scheduler_unlock(self->sched);
  ctx_guard_exit(&cg);
  return r;
}

// --- flush ----------------------------------------------------------------

enum damacy_status
damacy_flush(struct damacy* self)
{
  if (!self)
    return DAMACY_INVAL;

  // plan_into_slot below issues cuMemcpyHtoD on this thread; push the
  // retained primary so the call lands in the right context.
  struct ctx_guard cg = { 0 };
  enum damacy_status r = ctx_guard_enter(self, &cg);
  if (r != DAMACY_OK)
    return r;

  scheduler_lock(self->sched);
  if (self->failed_status != DAMACY_OK) {
    r = self->failed_status;
    goto Done;
  }

  // Worker only plans at full batch_size; flush emits the truncated tail.
  if (self->lookahead.size > 0 && self->lookahead.size < self->cfg.batch_size) {
    // Wait until a slot is FREE *and* no other plan is in progress.
    // plan_run releases the lock so a worker plan could be mid-flight;
    // both predicates are re-evaluated together under the lock on every
    // wake (cv_wait yields only inside the loop body), so we exit when
    // both hold simultaneously and the subsequent find_free_batch_slot
    // read is current.
    while ((find_free_batch_slot(&self->batch_pool) < 0 ||
            any_batch_planning(&self->batch_pool)) &&
           self->failed_status == DAMACY_OK)
      SCHEDULER_WAIT_DIAG(self->sched, 5000);
    if (self->failed_status != DAMACY_OK) {
      r = self->failed_status;
      goto Done;
    }
    int free_slot = find_free_batch_slot(&self->batch_pool);
    uint32_t n = self->lookahead.size;
    r = plan_reserve(self, (uint16_t)free_slot, n);
    if (r != DAMACY_OK)
      goto Done;
    scheduler_unlock(self->sched);
    float plan_ms = 0.f;
    enum damacy_status rs = plan_run(self, (uint16_t)free_slot, &plan_ms);
    scheduler_lock(self->sched);
    r = plan_commit(self, (uint16_t)free_slot, rs, plan_ms, NULL);
    if (r != DAMACY_OK)
      goto Done;
    self->stats.batches_truncated++;
  }

  // any_slot_in_flight catches the SLOT_PEELING window: peel_reserve has
  // already bumped n_chunks_dispatched (so find_oldest_filling_slot can
  // be -1) but peel_submit hasn't run yet, no wave exists yet — without
  // this check, flush would return while the worker still has unposted
  // IO to submit. damacy_pop's AGAIN gate keeps the same invariant.
  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  while ((any_wave_in_flight(&self->wave_pool) ||
          any_slot_in_flight(&self->wave_pool) ||
          find_oldest_filling_slot(&self->batch_pool) >= 0 ||
          any_batch_planning(&self->batch_pool)) &&
         self->failed_status == DAMACY_OK)
    SCHEDULER_WAIT_DIAG(self->sched, 5000);
  metric_record(
    &self->stats.flush_wait, platform_toc(&flush_clock) * 1000.0f, 0, 0);
  r = self->failed_status != DAMACY_OK ? self->failed_status : DAMACY_OK;

Done:
  scheduler_unlock(self->sched);
  ctx_guard_exit(&cg);
  return r;
}

// --- batch info / stats ---------------------------------------------------

void
damacy_batch_info(const struct damacy_batch* b, struct damacy_batch_info* out)
{
  if (!out)
    return;
  memset(out, 0, sizeof(*out));
  if (!b || !b->d || b->slot_idx >= 2)
    return;
  const struct damacy* self = b->d;
  const struct damacy_batch_slot* slot = &self->batch_pool.slots[b->slot_idx];
  if (slot->state != BATCH_HELD)
    return;
  out->device_ptr = slot->dev_ptr;
  out->rank = self->batch_pool.rank;
  out->dtype = self->cfg.dtype;
  out->ready_stream = (void*)self->wave_pool.stream_decode;
  out->batch_id = slot->batch_id;
  for (uint8_t d = 0; d < self->batch_pool.rank; ++d)
    out->shape[d] = self->batch_pool.shape[d];
  // shape[0] reflects actual sample count (< batch_size for flushed partials).
  out->shape[0] = (int64_t)slot->n_samples;
}

void
damacy_stats_get(const struct damacy* self, struct damacy_stats* out)
{
  if (!out)
    return;
  if (!self) {
    memset(out, 0, sizeof(*out));
    return;
  }
  // scheduler_lock guards every metric_record write; without it the
  // struct copy below races every plan/pop_wait/flush_wait update. The
  // mutex doesn't change observable state, so the const cast is safe.
  struct damacy* m = (struct damacy*)self;
  scheduler_lock(m->sched);
  *out = m->stats;
  out->gpu_bytes_committed = gpu_budget_committed(m->budget);
  scheduler_unlock(m->sched);
  // Cache stats have their own internal mutex; safe outside scheduler_lock.
  if (m->meta_cache) {
    struct zarr_meta_cache_stats ms;
    zarr_meta_cache_stats_get(m->meta_cache, &ms);
    out->zarr_meta_hits = ms.counters.hits;
    out->zarr_meta_misses = ms.counters.misses;
  }
  if (m->shard_cache) {
    struct zarr_shard_cache_stats ss;
    zarr_shard_cache_stats_get(m->shard_cache, &ss);
    out->shard_idx_hits = ss.counters.hits;
    out->shard_idx_misses = ss.counters.misses;
  }
}

void
damacy_stats_reset(struct damacy* self)
{
  if (!self)
    return;
  stats_init(&self->stats);
}
