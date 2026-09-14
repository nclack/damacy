#include "damacy_internal.h"

struct damacy_push_result
damacy_push(struct damacy* self, struct damacy_sample_slice samples)
{
  struct damacy_push_result result = { .unconsumed = samples,
                                       .status = DAMACY_INVAL };
  if (!self || ((!samples.beg || !samples.end) && samples.beg != samples.end) ||
      samples.beg > samples.end)
    return result;
  if (self->stopped) {
    result.status = DAMACY_SHUTDOWN;
    return result;
  }
  scheduler_lock(self->sched);
  if (self->stopping || self->failed_status != DAMACY_OK)
    result.status = DAMACY_SHUTDOWN;
  else
    result = self->planner->ops->push(self->planner, samples);
  scheduler_broadcast(self->sched);
  scheduler_unlock(self->sched);
  return result;
}
