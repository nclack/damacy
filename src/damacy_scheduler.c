#include "damacy_internal.h"

void
damacy_scheduler_enter(void* arg)
{
  struct damacy* self = arg;
  if (self->executor->ops->enter_thread)
    self->failed_status = self->executor->ops->enter_thread(self->executor);
}

void
damacy_scheduler_leave(void* arg)
{
  struct damacy* self = arg;
  if (self->executor->ops->leave_thread)
    self->executor->ops->leave_thread(self->executor);
}

int
damacy_scheduler_step(void* arg)
{
  struct damacy* self = arg;
  if (self->stopping || self->failed_status != DAMACY_OK)
    return 0;
  ++self->stats.worker_steps;
  int changed = 0;
  enum damacy_status status =
    self->executor->ops->step(self->executor, &changed);
  if (status == DAMACY_AGAIN)
    status = DAMACY_OK;
  if (status == DAMACY_OK)
    status = pipeline_prepare(self, &changed);
  while (status == DAMACY_OK && self->plan_count) {
    struct prepared_plan* plan = self->plans[self->plan_head];
    status =
      self->executor->ops->submit(self->executor, plan, self->next_batch_id);
    if (status == DAMACY_AGAIN)
      return changed;
    if (status != DAMACY_OK)
      break;
    self->plans[self->plan_head] = NULL;
    self->plan_head = (self->plan_head + 1) % self->queues.prepared_batches;
    --self->plan_count;
    ++self->next_batch_id;
    changed = 1;
  }
  if (status != DAMACY_OK) {
    self->failed_status = status;
    return 1;
  }
  return changed;
}
