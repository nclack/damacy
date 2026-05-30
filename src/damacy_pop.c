#include "damacy.h"

#include "damacy_internal.h"
#include "damacy_stats.h"
#include "log/log.h"
#include "nvtx/nvtx.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <cuda.h>
#include <string.h>

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
        lookahead_size(&self->lookahead) == 0 &&
        prefetcher_in_flight(self->prefetcher) == 0 &&
        !prefetcher_has_ready(self->prefetcher)) {
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
  if (s >= DAMACY_N_BATCH_SLOTS) {
    log_warn("damacy_release: slot_idx=%u out of range", (unsigned)s);
    return;
  }
  struct render_job* job =
    render_job_pool_for_batch_slot(&self->render_jobs, s);
  if (!job)
    return;
  scheduler_lock(self->sched);
  if (self->batch_pool.slots[s].state != BATCH_HELD) {
    log_warn("damacy_release: slot %u not HELD (state=%d); double release?",
             (unsigned)s,
             (int)self->batch_pool.slots[s].state);
    scheduler_unlock(self->sched);
    return;
  }
  batch_slot_reset_for_reuse(&self->batch_pool.slots[s]);
  render_job_reset(job);
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
  if (s >= DAMACY_N_BATCH_SLOTS) {
    log_warn("damacy_release_event: slot_idx=%u out of range", (unsigned)s);
    return DAMACY_INVAL;
  }
  struct render_job* job =
    render_job_pool_for_batch_slot(&self->render_jobs, s);
  if (!job)
    return DAMACY_INVAL;

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
  // either slot — picks up this wait. The flag is read by plan_commit
  // to host-sync stream_post before its sync cuMemsetD8 (which targets the
  // legacy null stream and would otherwise race).
  if (cuStreamWaitEvent(self->wave_pool.stream_post, (CUevent)event, 0) !=
      CUDA_SUCCESS) {
    // Deferred wait couldn't be installed; fall back to immediate release
    // so the slot doesn't leak (caller would block forever in pop).
    batch_slot_reset_for_reuse(slot);
    render_job_reset(job);
    r = DAMACY_CUDA;
    goto Done;
  }
  slot->deferred_release_pending = 1;

  batch_slot_reset_for_reuse(slot);
  slot->deferred_release_pending = 1;
  render_job_reset(job);
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

  // plan_run below issues cuMemcpyHtoD on this thread; push the
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

  // Drain the prefetcher first so the tail count is stable: in-flight samples
  // may still be admitted, fetched, and reach READY.
  while ((lookahead_size(&self->lookahead) > 0 ||
          prefetcher_in_flight(self->prefetcher) > 0) &&
         self->failed_status == DAMACY_OK)
    SCHEDULER_WAIT_DIAG(self->sched, 5000);
  if (self->failed_status != DAMACY_OK) {
    r = self->failed_status;
    goto Done;
  }
  while ((prefetcher_ready_prefix_count(self->prefetcher) > 0 ||
          find_accumulating_batch_slot(&self->batch_pool) >= 0) &&
         self->failed_status == DAMACY_OK) {
    while (((find_free_batch_slot(&self->batch_pool) < 0 &&
             find_accumulating_batch_slot(&self->batch_pool) < 0) ||
            any_batch_planning(&self->batch_pool)) &&
           self->failed_status == DAMACY_OK)
      SCHEDULER_WAIT_DIAG(self->sched, 5000);
    if (self->failed_status != DAMACY_OK) {
      r = self->failed_status;
      goto Done;
    }
    r = plan_ready_prefetch(self, 1, NULL);
    if (r != DAMACY_OK)
      goto Done;
  }

  // any_slot_in_flight catches the SLOT_PEELING window: peel_reserve has
  // already advanced the render-job cursor (so no job may look dispatchable)
  // but peel_submit hasn't run yet, no wave exists yet — without
  // this check, flush would return while the worker still has unposted
  // IO to submit. damacy_pop's AGAIN gate keeps the same invariant.
  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  while ((any_wave_in_flight(&self->wave_pool) ||
          any_slot_in_flight(&self->wave_pool) ||
          find_oldest_rendering_slot(&self->batch_pool) >= 0 ||
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
  if (!b || !b->d || b->slot_idx >= DAMACY_N_BATCH_SLOTS)
    return;
  const struct damacy* self = b->d;
  const struct damacy_batch_slot* slot = &self->batch_pool.slots[b->slot_idx];
  if (slot->state != BATCH_HELD)
    return;
  out->device_ptr = slot->dev_ptr;
  out->rank = self->batch_pool.rank;
  out->dtype = self->cfg.dtype;
  out->ready_stream = (void*)self->wave_pool.stream_post;
  out->batch_id = slot->batch_id;
  for (uint8_t d = 0; d < self->batch_pool.rank; ++d)
    out->shape[d] = self->batch_pool.shape[d];
  // shape[0] reflects actual sample count (< samples_per_batch for flushed
  // partials).
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
  if (m->array_meta_cache) {
    struct prefetch_cache_stats cs;
    prefetch_cache_stats_get(m->array_meta_cache, &cs);
    out->array_meta.hits = cs.counters.hits;
    out->array_meta.misses = cs.counters.misses;
  }
  if (m->shard_index_cache) {
    struct prefetch_cache_stats cs;
    prefetch_cache_stats_get(m->shard_index_cache, &cs);
    out->shard_index.hits = cs.counters.hits;
    out->shard_index.misses = cs.counters.misses;
  }
  if (m->chunk_layout_cache) {
    struct prefetch_cache_stats cs;
    prefetch_cache_stats_get(m->chunk_layout_cache, &cs);
    out->chunk_layout.hits = cs.counters.hits;
    out->chunk_layout.misses = cs.counters.misses;
  }
}

void
damacy_stats_reset(struct damacy* self)
{
  if (!self)
    return;
  stats_init(&self->stats);
}
