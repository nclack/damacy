// Unit tests for planner/coalesce.c: synthetic planner_output → sort +
// fuse-with-cap + round-robin interleave. No zarr/store/cache plumbing —
// every input read_op and chunk_plan is constructed inline.

#include "damacy_limits.h"
#include "expect.h"
#include "planner/coalesce.h"
#include "planner/planner.h"
#include "util/path_intern.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAP_UNCAPPED UINT64_MAX

// Per-test intern table so equal path literals share a pointer — same
// invariant the production planner provides. Reset per test via free.
static struct path_intern g_paths;

static void
mk(struct read_op* r,
   struct chunk_plan* cp,
   const char* path,
   uint64_t file_offset,
   uint32_t nbytes,
   uint32_t read_op_idx,
   uint32_t offset_in_read)
{
  memset(r, 0, sizeof *r);
  r->shard_path = path_intern_acquire(&g_paths, path);
  r->file_offset = file_offset;
  r->nbytes = nbytes;
  memset(cp, 0, sizeof *cp);
  cp->read_op_idx = read_op_idx;
  cp->offset_in_read = offset_in_read;
  cp->compressed_nbytes = 32;
  cp->decompressed_nbytes = 64;
  cp->is_fill = 0;
}

static void
mk_fill(struct read_op* r, struct chunk_plan* cp, uint32_t read_op_idx)
{
  memset(r, 0, sizeof *r);
  r->shard_path = NULL;
  r->file_offset = 0;
  r->nbytes = 0;
  memset(cp, 0, sizeof *cp);
  cp->read_op_idx = read_op_idx;
  cp->offset_in_read = 0;
  cp->compressed_nbytes = 0;
  cp->decompressed_nbytes = 64;
  cp->is_fill = 1;
}

// Allocate scratch sized for n input read_ops and call coalesce.
static enum damacy_status
run_coalesce(struct planner_output* out, uint64_t cap, uint32_t n_in)
{
  uint32_t* u32 = (uint32_t*)calloc((size_t)n_in * 4u, sizeof(uint32_t));
  struct read_op* ops = (struct read_op*)calloc(n_in, sizeof(struct read_op));
  enum damacy_status s =
    coalesce_chunks(out, cap, DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE, u32, ops);
  free(u32);
  free(ops);
  return s;
}

// Empty input passes through.
static int
test_empty(void)
{
  struct planner_output out = { 0 };
  struct read_op r = { 0 };
  struct chunk_plan c = { 0 };
  out.read_ops = &r;
  out.read_ops_cap = 1;
  out.chunk_plans = &c;
  out.chunk_plans_cap = 1;
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, 1) == DAMACY_OK);
  EXPECT(out.n_read_ops == 0);
  EXPECT(out.n_chunk_plans == 0);
  EXPECT(out.n_chunks_to_load == 0);
  EXPECT(out.n_loads_issued == 0);
  return 0;
}

// One real chunk → one read_op, indices unchanged, counters = 1.
static int
test_single_chunk_passthrough(void)
{
  struct read_op reads[1] = { 0 };
  struct chunk_plan chunks[1] = { 0 };
  mk(&reads[0], &chunks[0], "shard/0", 4096, 4096, 0, 100);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 1,
    .n_read_ops = 1,
    .chunk_plans = chunks,
    .chunk_plans_cap = 1,
    .n_chunk_plans = 1,
  };
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, 1) == DAMACY_OK);
  EXPECT(out.n_read_ops == 1);
  EXPECT(reads[0].file_offset == 4096);
  EXPECT(reads[0].nbytes == 4096);
  EXPECT(chunks[0].read_op_idx == 0);
  EXPECT(chunks[0].offset_in_read == 100);
  EXPECT(out.n_chunks_to_load == 1);
  EXPECT(out.n_loads_issued == 1);
  return 0;
}

// Two touching reads in the same shard fuse to one.
static int
test_touching_fuses(void)
{
  struct read_op reads[2] = { 0 };
  struct chunk_plan chunks[2] = { 0 };
  mk(&reads[0], &chunks[0], "shard/0", 0, 4096, 0, 50);
  mk(&reads[1], &chunks[1], "shard/0", 4096, 4096, 1, 200);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 2,
    .n_read_ops = 2,
    .chunk_plans = chunks,
    .chunk_plans_cap = 2,
    .n_chunk_plans = 2,
  };
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, 2) == DAMACY_OK);
  EXPECT(out.n_read_ops == 1);
  EXPECT(reads[0].file_offset == 0);
  EXPECT(reads[0].nbytes == 8192);
  EXPECT(chunks[0].read_op_idx == 0);
  EXPECT(chunks[0].offset_in_read == 50);
  EXPECT(chunks[1].read_op_idx == 0);
  EXPECT(chunks[1].offset_in_read == 200 + 4096); // shifted into fused leader
  EXPECT(out.n_chunks_to_load == 2);
  EXPECT(out.n_loads_issued == 1);
  return 0;
}

// Page-overlapping windows fuse (curr_offset == leader_offset, same shard).
static int
test_overlapping_fuses(void)
{
  struct read_op reads[2] = { 0 };
  struct chunk_plan chunks[2] = { 0 };
  mk(&reads[0], &chunks[0], "shard/0", 0, 8192, 0, 100);
  mk(&reads[1], &chunks[1], "shard/0", 4096, 4096, 1, 50);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 2,
    .n_read_ops = 2,
    .chunk_plans = chunks,
    .chunk_plans_cap = 2,
    .n_chunk_plans = 2,
  };
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, 2) == DAMACY_OK);
  EXPECT(out.n_read_ops == 1);
  EXPECT(reads[0].file_offset == 0);
  EXPECT(reads[0].nbytes == 8192); // fused stays at 8192 (second was inside)
  EXPECT(chunks[0].offset_in_read == 100);
  EXPECT(chunks[1].offset_in_read == 50 + 4096);
  return 0;
}

// A gap between reads prevents fusion.
static int
test_gap_no_fusion(void)
{
  struct read_op reads[2] = { 0 };
  struct chunk_plan chunks[2] = { 0 };
  mk(&reads[0], &chunks[0], "shard/0", 0, 4096, 0, 0);
  mk(&reads[1], &chunks[1], "shard/0", 8192, 4096, 1, 0);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 2,
    .n_read_ops = 2,
    .chunk_plans = chunks,
    .chunk_plans_cap = 2,
    .n_chunk_plans = 2,
  };
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, 2) == DAMACY_OK);
  EXPECT(out.n_read_ops == 2);
  EXPECT(out.n_loads_issued == 2);
  return 0;
}

// Different shard paths don't fuse even at the same offset.
static int
test_different_paths_no_fusion(void)
{
  struct read_op reads[2] = { 0 };
  struct chunk_plan chunks[2] = { 0 };
  mk(&reads[0], &chunks[0], "shard/0", 0, 4096, 0, 0);
  mk(&reads[1], &chunks[1], "shard/1", 0, 4096, 1, 0);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 2,
    .n_read_ops = 2,
    .chunk_plans = chunks,
    .chunk_plans_cap = 2,
    .n_chunk_plans = 2,
  };
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, 2) == DAMACY_OK);
  EXPECT(out.n_read_ops == 2);
  return 0;
}

// Long run of touching reads with cap = 2×single → ceil(N/2) outputs.
static int
test_cap_splits(void)
{
  enum
  {
    N = 6
  };
  struct read_op reads[N] = { 0 };
  struct chunk_plan chunks[N] = { 0 };
  for (uint32_t i = 0; i < N; ++i)
    mk(&reads[i],
       &chunks[i],
       "shard/0",
       (uint64_t)i * 4096,
       4096,
       i,
       /*offset_in_read*/ 0);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = N,
    .n_read_ops = N,
    .chunk_plans = chunks,
    .chunk_plans_cap = N,
    .n_chunk_plans = N,
  };
  // cap = 2 * single chunk → at most 2 chunks per output read_op.
  EXPECT(run_coalesce(&out, 2u * 4096u, N) == DAMACY_OK);
  EXPECT(out.n_read_ops == 3);
  for (uint32_t i = 0; i < out.n_read_ops; ++i)
    EXPECT(reads[i].nbytes <= 2u * 4096u);
  // Pairs (0,1),(2,3),(4,5) share read_op_idx.
  EXPECT(chunks[0].read_op_idx == chunks[1].read_op_idx);
  EXPECT(chunks[2].read_op_idx == chunks[3].read_op_idx);
  EXPECT(chunks[4].read_op_idx == chunks[5].read_op_idx);
  EXPECT(chunks[0].read_op_idx != chunks[2].read_op_idx);
  EXPECT(chunks[2].read_op_idx != chunks[4].read_op_idx);
  return 0;
}

// A single read that already exceeds the cap is emitted as-is (best-effort).
static int
test_single_over_cap(void)
{
  struct read_op reads[1] = { 0 };
  struct chunk_plan chunks[1] = { 0 };
  mk(&reads[0], &chunks[0], "shard/0", 0, 1u << 20, 0, 0); // 1 MB
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 1,
    .n_read_ops = 1,
    .chunk_plans = chunks,
    .chunk_plans_cap = 1,
    .n_chunk_plans = 1,
  };
  // Cap < single read size.
  EXPECT(run_coalesce(&out, 4096u, 1) == DAMACY_OK);
  EXPECT(out.n_read_ops == 1);
  EXPECT(reads[0].nbytes == (1u << 20));
  return 0;
}

// Output read_ops are non-overlapping and per-shard sorted by offset.
static int
test_non_overlapping_output(void)
{
  enum
  {
    N = 5
  };
  // Mixed shards, deliberate out-of-order emission to force a sort.
  struct read_op reads[N] = { 0 };
  struct chunk_plan chunks[N] = { 0 };
  mk(&reads[0], &chunks[0], "shard/B", 4096, 4096, 0, 0);
  mk(&reads[1], &chunks[1], "shard/A", 8192, 4096, 1, 0);
  mk(&reads[2], &chunks[2], "shard/A", 0, 4096, 2, 0);
  mk(&reads[3], &chunks[3], "shard/B", 0, 4096, 3, 0);
  mk(&reads[4], &chunks[4], "shard/A", 4096, 4096, 4, 0);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = N,
    .n_read_ops = N,
    .chunk_plans = chunks,
    .chunk_plans_cap = N,
    .n_chunk_plans = N,
  };
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, N) == DAMACY_OK);
  // shard/A: 3 touching reads → 1 fused; shard/B: 2 touching reads → 1 fused.
  EXPECT(out.n_read_ops == 2);
  // Round-robin emit visits shard runs in sorted order: A precedes B;
  // within each, offsets are non-overlapping by construction (single
  // fused leader per shard).
  EXPECT(strcmp(reads[0].shard_path, "shard/A") == 0);
  EXPECT(strcmp(reads[1].shard_path, "shard/B") == 0);
  EXPECT(reads[0].file_offset == 0 && reads[0].nbytes == 12288);
  EXPECT(reads[1].file_offset == 0 && reads[1].nbytes == 8192);
  return 0;
}

// Fused ops alternate across shards (per-shard offset order kept);
// chunk_plans still point at the right op afterwards, including the
// offset shift picked up during fusion.
static int
test_round_robin_interleave(void)
{
  enum
  {
    N = 5
  };
  // shard/A: 0..4096 and 4096..8192 touch → fuse; 16384 is separate.
  // shard/B: two non-touching reads.
  struct read_op reads[N] = { 0 };
  struct chunk_plan chunks[N] = { 0 };
  mk(&reads[0], &chunks[0], "shard/A", 0, 4096, 0, 50);
  mk(&reads[1], &chunks[1], "shard/A", 4096, 4096, 1, 200);
  mk(&reads[2], &chunks[2], "shard/A", 16384, 4096, 2, 0);
  mk(&reads[3], &chunks[3], "shard/B", 0, 4096, 3, 0);
  mk(&reads[4], &chunks[4], "shard/B", 8192, 4096, 4, 0);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = N,
    .n_read_ops = N,
    .chunk_plans = chunks,
    .chunk_plans_cap = N,
    .n_chunk_plans = N,
  };
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, N) == DAMACY_OK);
  EXPECT(out.n_read_ops == 4);
  // Fuse order is A:[0..8192], A:[16384..20480], B:[0..4096],
  // B:[8192..12288]; round-robin emit = A0, B0, A1, B1.
  EXPECT(strcmp(reads[0].shard_path, "shard/A") == 0);
  EXPECT(reads[0].file_offset == 0 && reads[0].nbytes == 8192);
  EXPECT(strcmp(reads[1].shard_path, "shard/B") == 0);
  EXPECT(reads[1].file_offset == 0 && reads[1].nbytes == 4096);
  EXPECT(strcmp(reads[2].shard_path, "shard/A") == 0);
  EXPECT(reads[2].file_offset == 16384 && reads[2].nbytes == 4096);
  EXPECT(strcmp(reads[3].shard_path, "shard/B") == 0);
  EXPECT(reads[3].file_offset == 8192 && reads[3].nbytes == 4096);
  // chunk_plans follow the remap through fuse + interleave.
  EXPECT(chunks[0].read_op_idx == 0 && chunks[0].offset_in_read == 50);
  EXPECT(chunks[1].read_op_idx == 0 &&
         chunks[1].offset_in_read == 200 + 4096);
  EXPECT(chunks[2].read_op_idx == 2);
  EXPECT(chunks[3].read_op_idx == 1);
  EXPECT(chunks[4].read_op_idx == 3);
  EXPECT(out.n_chunks_to_load == 5);
  EXPECT(out.n_loads_issued == 4);
  return 0;
}

// Long contiguous touching stream past the default chunks-per-wave cap:
// no surviving read_op may end up referenced by more than the cap.
static int
test_chunk_count_cap(void)
{
  const uint32_t cap = DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE;
  const uint32_t n = cap + 5u;
  const uint32_t step = 4096u;
  struct read_op* reads = (struct read_op*)calloc(n, sizeof *reads);
  struct chunk_plan* chunks = (struct chunk_plan*)calloc(n, sizeof *chunks);
  EXPECT(reads && chunks);
  for (uint32_t i = 0; i < n; ++i)
    mk(&reads[i], &chunks[i], "shard/0", (uint64_t)i * step, step, i, 0);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = n,
    .n_read_ops = n,
    .chunk_plans = chunks,
    .chunk_plans_cap = n,
    .n_chunk_plans = n,
  };
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, n) == DAMACY_OK);
  // Exact split: one leader at the cap, one with the 5 stragglers —
  // proves fusion happened up to cap and reset cleanly on overflow.
  EXPECT(out.n_read_ops == 2);
  uint32_t refs[2] = { 0, 0 };
  for (uint32_t i = 0; i < out.n_chunk_plans; ++i)
    refs[chunks[i].read_op_idx]++;
  uint32_t lo = refs[0] < refs[1] ? refs[0] : refs[1];
  uint32_t hi = refs[0] < refs[1] ? refs[1] : refs[0];
  EXPECT(lo == 5u);
  EXPECT(hi == cap);
  free(reads);
  free(chunks);
  return 0;
}

// Fill placeholders survive coalesce as their own read_ops; counters
// exclude them; chunk_plans pointing at them get a valid read_op_idx.
static int
test_fills_passthrough(void)
{
  struct read_op reads[3] = { 0 };
  struct chunk_plan chunks[3] = { 0 };
  mk(&reads[0], &chunks[0], "shard/0", 0, 4096, 0, 0);
  mk_fill(&reads[1], &chunks[1], 1);
  mk(&reads[2], &chunks[2], "shard/0", 4096, 4096, 2, 0);
  struct planner_output out = {
    .read_ops = reads,
    .read_ops_cap = 3,
    .n_read_ops = 3,
    .chunk_plans = chunks,
    .chunk_plans_cap = 3,
    .n_chunk_plans = 3,
  };
  EXPECT(run_coalesce(&out, CAP_UNCAPPED, 3) == DAMACY_OK);
  // 2 real reads fuse → 1; fill stays separate.
  EXPECT(out.n_read_ops == 2);
  // chunk[1] is fill; its read_op_idx must point at a read_op with
  // nbytes == 0 (the fill placeholder).
  EXPECT(reads[chunks[1].read_op_idx].nbytes == 0);
  EXPECT(reads[chunks[0].read_op_idx].nbytes == 8192);
  EXPECT(chunks[0].read_op_idx == chunks[2].read_op_idx);
  EXPECT(out.n_chunks_to_load == 2);
  EXPECT(out.n_loads_issued == 1);
  return 0;
}

int
main(void)
{
  RUN(test_empty);
  RUN(test_single_chunk_passthrough);
  RUN(test_touching_fuses);
  RUN(test_overlapping_fuses);
  RUN(test_gap_no_fusion);
  RUN(test_different_paths_no_fusion);
  RUN(test_cap_splits);
  RUN(test_single_over_cap);
  RUN(test_non_overlapping_output);
  RUN(test_round_robin_interleave);
  RUN(test_chunk_count_cap);
  RUN(test_fills_passthrough);
  return 0;
}
