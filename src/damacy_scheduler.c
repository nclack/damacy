#include "damacy.h"

#include "damacy_internal.h"
#include "nvtx/nvtx.h"
#include "wave/wave_input.h"

#include <cuda.h>

// Drains sealed render jobs into free input_slots, planning a fresh batch
// when no render job has work ready.
static enum damacy_status
kick_input_into_free_slots(struct damacy* self, int* changed)
{
  for (;;) {
    int target_job = find_render_job_with_work(&self->render_jobs);
    if (target_job < 0) {
      int plan_changed = 0;
      enum damacy_status s = plan_ready_prefetch(self, 0, &plan_changed);
      if (s != DAMACY_OK)
        return s;
      if (changed && plan_changed)
        *changed = 1;
      if (plan_changed)
        continue;
      break;
    }

    struct wave_input_reservation t = { 0 };
    enum damacy_status s =
      wave_input_reserve(&self->wave_pool, (uint16_t)target_job, &t);
    if (s != DAMACY_OK)
      return s;
    if (!t.active)
      break;
    damacy_nvtx_range_pushf("input/slot%d", t.input_slot_idx);
    scheduler_unlock(self->sched);
    struct store_submit_result submit = wave_input_submit(&self->wave_pool, &t);
    scheduler_lock(self->sched);
    s = wave_input_commit(&self->wave_pool, &t, submit, changed);
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
// (wave_pool_advance, plan_ready_prefetch/plan_commit,
// wave_input_commit) OR-sets it on a real state transition; the worker
// broadcasts iff non-zero.
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
    r = kick_input_into_free_slots(self, &changed);
  if (r != DAMACY_OK && self->failed_status == DAMACY_OK) {
    self->failed_status = r;
    return 1;
  }
  return changed;
}
