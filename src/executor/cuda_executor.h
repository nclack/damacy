#pragma once

#include "damacy_pipeline.h"
#include "numa/numa.h"

void
cuda_resolve_numa(const struct damacy_config* config,
                  struct numa_resolved* out);
uint64_t
cuda_executor_set_budget(struct damacy_executor* executor, uint64_t value);
