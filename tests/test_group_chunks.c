// Unit tests for planner/group_chunks.c: stable counting-sort of
// chunk_plans by read_op_idx. Synthetic inputs only.

#include "expect.h"
#include "planner/group_chunks.h"
#include "planner/planner.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
mk_cp(struct chunk_plan* cp,
      uint32_t read_op_idx,
      uint32_t offset_in_read,
      uint16_t sample_idx,
      uint32_t tag)
{
  memset(cp, 0, sizeof *cp);
  cp->read_op_idx = read_op_idx;
  cp->offset_in_read = offset_in_read;
  cp->sample_idx_in_batch = sample_idx;
  // Stash a unique tag in chunk_d[0] so we can verify relative order.
  cp->chunk_d[0] = tag;
}

static enum damacy_status
run_group(struct planner_output* out)
{
  uint32_t* u32 = (uint32_t*)calloc((size_t)out->n_read_ops + 1u,
                                    sizeof(uint32_t));
  struct chunk_plan* tmp = (struct chunk_plan*)calloc(
    out->n_chunk_plans, sizeof(struct chunk_plan));
  enum damacy_status s = group_chunks_by_read(out, u32, tmp);
  free(u32);
  free(tmp);
  return s;
}

// Empty input passes through.
static int
test_empty(void)
{
  struct planner_output out = { 0 };
  struct chunk_plan c = { 0 };
  out.chunk_plans = &c;
  out.chunk_plans_cap = 1;
  EXPECT(run_group(&out) == DAMACY_OK);
  EXPECT(out.n_chunk_plans == 0);
  return 0;
}

// Single chunk is a no-op.
static int
test_single_chunk(void)
{
  struct chunk_plan chunks[1];
  mk_cp(&chunks[0], 0, 100, 0, 42);
  struct planner_output out = {
    .chunk_plans = chunks,
    .chunk_plans_cap = 1,
    .n_chunk_plans = 1,
    .n_read_ops = 1,
  };
  EXPECT(run_group(&out) == DAMACY_OK);
  EXPECT(chunks[0].chunk_d[0] == 42);
  EXPECT(chunks[0].read_op_idx == 0);
  return 0;
}

// Already-sorted input stays put; relative order preserved.
static int
test_already_sorted(void)
{
  struct chunk_plan chunks[4];
  mk_cp(&chunks[0], 0, 0, 0, 100);
  mk_cp(&chunks[1], 0, 4096, 0, 101);
  mk_cp(&chunks[2], 1, 0, 0, 200);
  mk_cp(&chunks[3], 1, 4096, 0, 201);
  struct planner_output out = {
    .chunk_plans = chunks,
    .chunk_plans_cap = 4,
    .n_chunk_plans = 4,
    .n_read_ops = 2,
  };
  EXPECT(run_group(&out) == DAMACY_OK);
  EXPECT(chunks[0].chunk_d[0] == 100);
  EXPECT(chunks[1].chunk_d[0] == 101);
  EXPECT(chunks[2].chunk_d[0] == 200);
  EXPECT(chunks[3].chunk_d[0] == 201);
  return 0;
}

// Interleaved input gets grouped; chunks with same read_op_idx end up
// contiguous; relative order within each read_op preserved.
static int
test_interleaved_groups(void)
{
  struct chunk_plan chunks[6];
  // read_op:  0   1   0   2   1   0
  // tag:    100 200 101 300 201 102
  mk_cp(&chunks[0], 0, 0, 0, 100);
  mk_cp(&chunks[1], 1, 0, 1, 200);
  mk_cp(&chunks[2], 0, 100, 0, 101);
  mk_cp(&chunks[3], 2, 0, 2, 300);
  mk_cp(&chunks[4], 1, 100, 1, 201);
  mk_cp(&chunks[5], 0, 200, 0, 102);
  struct planner_output out = {
    .chunk_plans = chunks,
    .chunk_plans_cap = 6,
    .n_chunk_plans = 6,
    .n_read_ops = 3,
  };
  EXPECT(run_group(&out) == DAMACY_OK);
  // Expected: all read_op=0 first (tags 100,101,102), then read_op=1
  // (200,201), then read_op=2 (300).
  EXPECT(chunks[0].read_op_idx == 0 && chunks[0].chunk_d[0] == 100);
  EXPECT(chunks[1].read_op_idx == 0 && chunks[1].chunk_d[0] == 101);
  EXPECT(chunks[2].read_op_idx == 0 && chunks[2].chunk_d[0] == 102);
  EXPECT(chunks[3].read_op_idx == 1 && chunks[3].chunk_d[0] == 200);
  EXPECT(chunks[4].read_op_idx == 1 && chunks[4].chunk_d[0] == 201);
  EXPECT(chunks[5].read_op_idx == 2 && chunks[5].chunk_d[0] == 300);
  return 0;
}

// Chunks belonging to a read_op that has no other members still land
// in the correct slot (verifies count[] handling for sparse buckets).
static int
test_sparse_read_op(void)
{
  struct chunk_plan chunks[3];
  // Only read_op 2 used; read_ops 0 and 1 have no chunks pointing at them.
  mk_cp(&chunks[0], 2, 0, 0, 1);
  mk_cp(&chunks[1], 2, 100, 0, 2);
  mk_cp(&chunks[2], 2, 200, 0, 3);
  struct planner_output out = {
    .chunk_plans = chunks,
    .chunk_plans_cap = 3,
    .n_chunk_plans = 3,
    .n_read_ops = 3,
  };
  EXPECT(run_group(&out) == DAMACY_OK);
  EXPECT(chunks[0].chunk_d[0] == 1);
  EXPECT(chunks[1].chunk_d[0] == 2);
  EXPECT(chunks[2].chunk_d[0] == 3);
  return 0;
}

// Invalid input: read_op_idx >= n_read_ops returns DAMACY_INVAL.
static int
test_invalid_read_op_idx(void)
{
  struct chunk_plan chunks[2];
  mk_cp(&chunks[0], 0, 0, 0, 1);
  mk_cp(&chunks[1], 5, 0, 0, 2); // out of range
  struct planner_output out = {
    .chunk_plans = chunks,
    .chunk_plans_cap = 2,
    .n_chunk_plans = 2,
    .n_read_ops = 2,
  };
  EXPECT(run_group(&out) == DAMACY_INVAL);
  return 0;
}

// After grouping, chunks with same read_op_idx are contiguous AND
// sample_idx_in_batch distribution is preserved (counts unchanged).
static int
test_sample_distribution_preserved(void)
{
  struct chunk_plan chunks[5];
  mk_cp(&chunks[0], 1, 0, /*sample*/ 3, 1);
  mk_cp(&chunks[1], 0, 0, /*sample*/ 0, 2);
  mk_cp(&chunks[2], 1, 100, /*sample*/ 7, 3);
  mk_cp(&chunks[3], 0, 100, /*sample*/ 4, 4);
  mk_cp(&chunks[4], 0, 200, /*sample*/ 0, 5);
  struct planner_output out = {
    .chunk_plans = chunks,
    .chunk_plans_cap = 5,
    .n_chunk_plans = 5,
    .n_read_ops = 2,
  };
  // Count samples per id pre-group.
  uint32_t pre[8] = { 0 };
  for (uint32_t i = 0; i < 5; ++i)
    pre[chunks[i].sample_idx_in_batch]++;
  EXPECT(run_group(&out) == DAMACY_OK);
  uint32_t post[8] = { 0 };
  for (uint32_t i = 0; i < 5; ++i)
    post[chunks[i].sample_idx_in_batch]++;
  for (uint32_t s = 0; s < 8; ++s)
    EXPECT(pre[s] == post[s]);
  // Contiguity check.
  for (uint32_t i = 1; i < 5; ++i)
    EXPECT(chunks[i - 1].read_op_idx <= chunks[i].read_op_idx);
  return 0;
}

int
main(void)
{
  RUN(test_empty);
  RUN(test_single_chunk);
  RUN(test_already_sorted);
  RUN(test_interleaved_groups);
  RUN(test_sparse_read_op);
  RUN(test_invalid_read_op_idx);
  RUN(test_sample_distribution_preserved);
  return 0;
}
