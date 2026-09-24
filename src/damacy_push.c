#include "damacy_internal.h"

#include "query/selection.h"

static struct damacy_push_result
push_to_planner(struct damacy* self, struct damacy_sample_slice samples)
{
  struct damacy_sample_slice accepted = samples;
  if (!self->executor->accepts_indexed_samples) {
    accepted.end = samples.beg;
    while (accepted.end != samples.end && !query_has_indices(accepted.end))
      ++accepted.end;
  }
  struct damacy_push_result result =
    self->planner->ops->push(self->planner, accepted);
  result.unconsumed.end = samples.end;
  if (result.status == DAMACY_OK && accepted.end != samples.end)
    result.status = DAMACY_BUDGET;
  return result;
}

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
    result = push_to_planner(self, samples);
  scheduler_broadcast(self->sched);
  scheduler_unlock(self->sched);
  return result;
}
