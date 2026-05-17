#include "planner/group_chunks.h"

#include <string.h>

enum damacy_status
group_chunks_by_read(struct planner_output* out,
                     uint32_t* u32_scratch,
                     struct chunk_plan* chunk_plan_scratch)
{
  if (!out || !out->chunk_plans)
    return DAMACY_INVAL;
  uint32_t n = out->n_chunk_plans;
  if (n < 2)
    return DAMACY_OK;
  if (!u32_scratch || !chunk_plan_scratch)
    return DAMACY_INVAL;

  uint32_t R = out->n_read_ops;
  if (R == 0)
    return DAMACY_OK;

  uint32_t* head = u32_scratch; // head[r] = next write position for read_op r

  for (uint32_t r = 0; r <= R; ++r)
    head[r] = 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t r = out->chunk_plans[i].read_op_idx;
    if (r >= R)
      return DAMACY_INVAL;
    head[r + 1]++;
  }
  for (uint32_t r = 1; r <= R; ++r)
    head[r] += head[r - 1];

  // head[r] now holds the start position for read_op r. Walk input in
  // order, placing each chunk at head[r] and incrementing — stable.
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t r = out->chunk_plans[i].read_op_idx;
    chunk_plan_scratch[head[r]++] = out->chunk_plans[i];
  }
  memcpy(out->chunk_plans, chunk_plan_scratch, (size_t)n * sizeof(*chunk_plan_scratch));
  return DAMACY_OK;
}
