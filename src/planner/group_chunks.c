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
  uint32_t R = out->n_read_ops;
  out->n_read_op_groups = 0;
  if (n == 0 || R == 0)
    return DAMACY_OK;
  if (!u32_scratch || !chunk_plan_scratch)
    return DAMACY_INVAL;
  if (!out->read_op_groups || out->read_op_groups_cap < R)
    return DAMACY_OOM;

  uint32_t* head = u32_scratch;

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

  uint32_t g_out = 0;
  for (uint32_t r = 0; r < R; ++r) {
    uint32_t first = head[r];
    uint32_t last = head[r + 1];
    if (first == last)
      continue;
    out->read_op_groups[g_out++] = (struct read_op_group){
      .read_op_idx = r,
      .first_chunk = first,
      .n_chunks = last - first,
      .total_decompressed = 0,
    };
  }
  out->n_read_op_groups = g_out;

  if (n >= 2) {
    for (uint32_t i = 0; i < n; ++i) {
      uint32_t r = out->chunk_plans[i].read_op_idx;
      chunk_plan_scratch[head[r]++] = out->chunk_plans[i];
    }
    memcpy(out->chunk_plans, chunk_plan_scratch, (size_t)n * sizeof(*chunk_plan_scratch));
  }

  for (uint32_t g = 0; g < g_out; ++g) {
    struct read_op_group* grp = &out->read_op_groups[g];
    uint64_t sum = 0;
    for (uint32_t i = 0; i < grp->n_chunks; ++i)
      sum += out->chunk_plans[grp->first_chunk + i].decompressed_nbytes;
    grp->total_decompressed = sum;
  }
  return DAMACY_OK;
}

void
read_op_group_iterator_init(struct read_op_group_iterator* it,
                            const struct read_op_group* groups,
                            uint32_t n_groups,
                            uint32_t start_group)
{
  it->groups = groups;
  it->n_groups = n_groups;
  it->cursor = start_group;
}

int
read_op_group_iterator_next(struct read_op_group_iterator* it,
                            struct read_op_group* out)
{
  if (it->cursor >= it->n_groups)
    return 0;
  *out = it->groups[it->cursor++];
  return 1;
}
