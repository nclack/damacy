#pragma once

#include "damacy_pipeline.h"
#include "executor/dispatch.h"
#include "planner/plan.h"

enum
{
  CPU_READ_BUFFER_CHUNKS = 256
};

struct cpu_chunk_read
{
  uint32_t offset;
  uint32_t next;
};

struct cpu_read_plan
{
  struct read_op* reads;
  struct cpu_chunk_read* chunks;
  uint32_t* first_chunks;
  uint32_t count;
  uint64_t bytes;
};

void
cpu_read_plan_destroy(struct cpu_read_plan* plan);

enum damacy_status
cpu_read_plan_build(const struct prepared_plan* plan,
                    const struct damacy_cpu_config* config,
                    uint64_t available,
                    uint64_t active_plan_bytes,
                    struct cpu_read_plan* out);
