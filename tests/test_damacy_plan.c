#include "damacy_internal.h"
#include "damacy_stats.h"
#include "expect.h"

#include <stdlib.h>

struct test_planner
{
  struct damacy_planner base;
  unsigned remaining;
  unsigned prepared;
  enum damacy_status failure;
};

struct test_executor
{
  struct damacy_executor base;
  enum damacy_status status;
  unsigned accepted;
  unsigned submissions;
  uint64_t ids[4];
  struct prepared_plan* last;
};

static enum damacy_status
prepare(struct damacy_planner* base, struct prepared_plan** out)
{
  struct test_planner* self = (void*)base;
  if (self->failure)
    return self->failure;
  if (!self->remaining)
    return DAMACY_AGAIN;
  *out = calloc(1, sizeof(**out));
  if (!*out)
    return DAMACY_OOM;
  --self->remaining;
  ++self->prepared;
  return DAMACY_OK;
}

static enum damacy_status
submit(struct damacy_executor* base, struct prepared_plan* plan, uint64_t id)
{
  struct test_executor* self = (void*)base;
  self->last = plan;
  ++self->submissions;
  if (self->status != DAMACY_OK)
    return self->status;
  self->ids[self->accepted++] = id;
  prepared_plan_destroy(plan);
  return DAMACY_OK;
}

static enum damacy_status
step(struct damacy_executor* base, int* changed)
{
  (void)base;
  (void)changed;
  return DAMACY_OK;
}

static const struct damacy_planner_ops planner_ops = { .next = prepare };
static const struct damacy_executor_ops executor_ops = { .submit = submit,
                                                         .step = step };

static int
test_bounded_preparation_and_retry(void)
{
  struct test_planner planner = { .base.ops = &planner_ops, .remaining = 4 };
  struct test_executor executor = { .base.ops = &executor_ops,
                                    .status = DAMACY_AGAIN };
  struct prepared_plan* plans[2] = { 0 };
  struct damacy pipeline = { .planner = &planner.base,
                             .executor = &executor.base,
                             .plans = plans,
                             .queues.prepared_batches = 2 };
  stats_init(&pipeline.stats);
  EXPECT(damacy_scheduler_step(&pipeline));
  EXPECT(planner.prepared == 2 && planner.remaining == 2);
  EXPECT(pipeline.plan_count == 2 && pipeline.next_batch_id == 0);
  struct prepared_plan* waiting = plans[0];
  EXPECT(executor.last == waiting);
  EXPECT(!damacy_scheduler_step(&pipeline));
  EXPECT(executor.last == waiting && planner.prepared == 2);
  executor.status = DAMACY_OK;
  EXPECT(damacy_scheduler_step(&pipeline));
  EXPECT(pipeline.plan_count == 0 && executor.accepted == 2);
  EXPECT(damacy_scheduler_step(&pipeline));
  EXPECT(pipeline.plan_count == 0 && executor.accepted == 4);
  for (unsigned i = 0; i < 4; ++i)
    EXPECT(executor.ids[i] == i);
  EXPECT(pipeline.stats.plan.count == 4);
  return 0;
}

static int
test_terminal_failures_preserve_ownership(void)
{
  struct test_planner planner = { .base.ops = &planner_ops,
                                  .failure = DAMACY_NOTFOUND };
  struct test_executor executor = { .base.ops = &executor_ops,
                                    .status = DAMACY_BUDGET };
  struct prepared_plan* plans[1] = { 0 };
  struct damacy pipeline = { .planner = &planner.base,
                             .executor = &executor.base,
                             .plans = plans,
                             .queues.prepared_batches = 1 };
  stats_init(&pipeline.stats);
  EXPECT(damacy_scheduler_step(&pipeline));
  EXPECT(pipeline.failed_status == DAMACY_NOTFOUND &&
         executor.submissions == 0);
  EXPECT(!damacy_scheduler_step(&pipeline));
  pipeline.failed_status = DAMACY_OK;
  planner.failure = DAMACY_OK;
  planner.remaining = 1;
  EXPECT(damacy_scheduler_step(&pipeline));
  EXPECT(pipeline.failed_status == DAMACY_BUDGET);
  EXPECT(pipeline.plan_count == 1 && plans[0] == executor.last);
  EXPECT(pipeline.next_batch_id == 0 && executor.accepted == 0);
  EXPECT(!damacy_scheduler_step(&pipeline));
  prepared_plan_destroy(plans[0]);
  return 0;
}

int
main(void)
{
  RUN(test_bounded_preparation_and_retry);
  RUN(test_terminal_failures_preserve_ownership);
  return 0;
}
