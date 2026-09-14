#include "planner/plan.h"

#include <stdlib.h>

void
prepared_plan_destroy(struct prepared_plan* plan)
{
  if (!plan)
    return;
  free(plan->storage);
  free(plan);
}
