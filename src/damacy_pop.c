#include "damacy_internal.h"

#include "damacy_stats.h"
#include "log/log.h"

#include <string.h>

enum damacy_status
damacy_pop(struct damacy* self, struct damacy_batch** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!self)
    return DAMACY_INVAL;
  if (self->stopped)
    return DAMACY_SHUTDOWN;
  enum damacy_status status;
  scheduler_lock(self->sched);
  ++self->pop_calls;
  for (;;) {
    if (self->stopping) {
      status = DAMACY_SHUTDOWN;
      break;
    }
    if (self->failed_status != DAMACY_OK) {
      status = self->failed_status;
      break;
    }
    status = self->executor->ops->take(self->executor, out);
    if (status != DAMACY_AGAIN) {
      if (status == DAMACY_OK) {
        (*out)->owner = self;
        ++self->stats.batches_emitted;
      }
      break;
    }
    if (!self->plan_count && !self->planner->ops->pending(self->planner) &&
        !self->executor->ops->busy(self->executor))
      break;
    struct platform_clock clock = { 0 };
    platform_toc(&clock);
    SCHEDULER_WAIT_DIAG(self->sched, 5000);
    metric_record(&self->stats.pop_wait, platform_toc(&clock) * 1000, 0, 0);
  }
  --self->pop_calls;
  scheduler_broadcast(self->sched);
  scheduler_unlock(self->sched);
  return status;
}

void
damacy_release(struct damacy* self, struct damacy_batch* batch)
{
  if (!batch)
    return;
  if (batch->owner != self)
    log_warn("damacy_release: batch belongs to another pipeline");
  damacy_batch_release(batch);
}

enum damacy_status
damacy_release_event(struct damacy* self,
                     struct damacy_batch* batch,
                     void* event)
{
  if (!self || !batch || batch->owner != self)
    return DAMACY_INVAL;
  if (!event) {
    damacy_batch_release(batch);
    return DAMACY_OK;
  }
  scheduler_lock(self->sched);
  int stopping = self->stopping;
  enum damacy_status status = DAMACY_INVAL;
  if (!stopping)
    status = self->executor->ops->wait_event(self->executor, event);
  scheduler_unlock(self->sched);
  if (stopping && batch->buffer->wait_event)
    status = batch->buffer->wait_event(batch->buffer, event);
  damacy_batch_release(batch);
  return status;
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
  if (self->stopped) {
    *out = self->stats;
    return;
  }
  struct damacy* mutable = (struct damacy*)self;
  scheduler_lock(mutable->sched);
  *out = self->stats;
  if (!self->stopping) {
    mutable->planner->ops->stats(mutable->planner, out);
    mutable->executor->ops->stats(mutable->executor, out);
  }
  scheduler_unlock(mutable->sched);
}

void
damacy_stats_reset(struct damacy* self)
{
  if (!self || self->stopped)
    return;
  scheduler_lock(self->sched);
  if (!self->stopping) {
    stats_init(&self->stats);
    self->planner->ops->reset_stats(self->planner);
  }
  scheduler_unlock(self->sched);
}
