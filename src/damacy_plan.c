#include "damacy_internal.h"

#include "damacy_stats.h"

enum damacy_status
pipeline_prepare(struct damacy* self, int* changed)
{
  while (self->plan_count < self->queues.prepared_batches) {
    struct prepared_plan* plan = NULL;
    struct platform_clock clock = { 0 };
    platform_toc(&clock);
    enum damacy_status status = self->planner->ops->next(self->planner, &plan);
    if (status == DAMACY_AGAIN)
      return DAMACY_OK;
    if (status != DAMACY_OK)
      return status;
    metric_record(&self->stats.plan, platform_toc(&clock) * 1000, 0, 0);
    uint32_t index =
      (self->plan_head + self->plan_count) % self->queues.prepared_batches;
    self->plans[index] = plan;
    ++self->plan_count;
    self->stats.chunks_planned += plan->n_uses;
    for (uint32_t i = 0; i < plan->n_uses; ++i)
      self->stats.chunks_to_load += !plan->chunks[plan->uses[i].chunk].missing;
    *changed = 1;
  }
  return DAMACY_OK;
}
