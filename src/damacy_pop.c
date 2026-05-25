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
  // TODO: misses tails already pulled by prefetcher into PENDING.
  uint32_t la_size = lookahead_size(&self->lookahead);
  if (la_size > 0 && la_size < self->cfg.batch_size) {
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
    r = plan_reserve(self, (uint16_t)free_slot, la_size);
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
  // PR-1 keeps the public stat names; PR-2 will rename + add chunk_layout.
  if (m->array_meta_cache) {
    struct prefetch_cache_stats cs;
    prefetch_cache_stats_get(m->array_meta_cache, &cs);
    out->zarr_meta_hits = cs.counters.hits;
    out->zarr_meta_misses = cs.counters.misses;
  }
  if (m->shard_index_cache) {
    struct prefetch_cache_stats cs;
    prefetch_cache_stats_get(m->shard_index_cache, &cs);
    out->shard_idx_hits = cs.counters.hits;
    out->shard_idx_misses = cs.counters.misses;
  }
}

void
damacy_stats_reset(struct damacy* self)
{
  if (!self)
    return;
  stats_init(&self->stats);
}
