#include "damacy.h"

#include "damacy_internal.h"
#include "nvtx/nvtx.h"

#include <cuda.h>

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
