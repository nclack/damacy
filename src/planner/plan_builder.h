#pragma once

#include "planner/plan.h"
#include "prefetch/prefetch_handle.h"

struct planner_sample
{
  const char* uri;
  struct damacy_aabb aabb;
  struct prefetch_handle h_meta;
  struct prefetch_handle* h_shards;
  uint32_t n_shards;
  struct prefetch_handle h_layout;
};

struct prefetch_cache;

enum damacy_status
prepared_plan_build(struct prefetch_cache* arrays,
                    struct prefetch_cache* shards,
                    const struct planner_sample* samples,
                    uint32_t n_samples,
                    const struct damacy_batch_spec* output,
                    const struct damacy_plan_limits* limits,
                    struct prepared_plan** out);
