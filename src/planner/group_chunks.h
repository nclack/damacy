// Group step of the IO planning pipeline:
//   filter (emit) → sort → fuse-with-cap → group-by-read.
//
// Stable counting-sort of chunk_plans by read_op_idx so all chunks
// sharing a post-coalesce read_op are contiguous globally. The input dispatch
// in wave_pool relies on this to keep a read_op's chunks from
// straddling wave boundaries (which would otherwise cause cross-wave
// re-fires of the same IO).
//
// In-place from the caller's view: the function uses scratch the
// size of one chunk_plan[] to perform the permutation, then copies
// the result back.
#pragma once

#include "damacy.h"          // damacy_status
#include "planner/planner.h" // planner_output, chunk_plan

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Scratch: u32_scratch >= n_read_ops+1, chunk_plan_scratch >= n_chunk_plans.
  // out->read_op_groups must be sized >= out->n_read_ops.
  enum damacy_status group_chunks_by_read(
    struct planner_output* out,
    uint32_t* u32_scratch,
    struct chunk_plan* chunk_plan_scratch);

#ifdef __cplusplus
}
#endif
