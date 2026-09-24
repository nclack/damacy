// Coalesce step of the IO planning pipeline:
//   filter (emit) → sort → fuse-with-cap → interleave → group-by-read.
//
// Operates in place on dispatch_output: sorts per-chunk read windows
// by (shard_path, file_offset), then
// greedily fuses adjacent windows in the same shard into one
// read_op, bounded by read_op_max_bytes. Fused ops are emitted
// interleaved across shards (per-shard offset order kept) because
// simultaneous reads into one file serialize on network
// filesystems. chunk_plan.read_op_idx and
// chunk_plan.offset_in_read are rewritten to point at the surviving
// read_op. Equal shard paths need not share a pointer. Fill chunk_plans
// (path empty, nbytes == 0) keep their 1:1 placeholder read_ops untouched.
//
// Populates dispatch_output.n_chunks_to_load and n_loads_issued as
// part of the same pass.
//
// Cap policy: a single per-chunk read window that already exceeds
// read_op_max_bytes is emitted unfused as its own read_op (the cap
// is best-effort — fusion never grows past it, but it never
// shrinks a single chunk's read either).
#pragma once

#include "damacy.h"            // damacy_status
#include "executor/dispatch.h" // dispatch_output, read_op

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Merges reads in place and updates *n_reads. read_index[i] and
  // offset_in_read[i] locate original read i in the merged list.
  // Scratch requirements, with n the original *n_reads:
  //   read_index, offset_in_read: >= n uint32_t slots each
  //   u32_scratch:     >= 2 * n uint32_t slots
  //   read_op_scratch: >= n slots
  enum damacy_status coalesce_reads(struct read_op* reads,
                                    uint32_t* n_reads,
                                    uint64_t read_op_max_bytes,
                                    uint32_t max_chunks_per_wave,
                                    uint32_t* read_index,
                                    uint32_t* offset_in_read,
                                    uint32_t* u32_scratch,
                                    struct read_op* read_op_scratch);

  // Scratch requirements:
  //   u32_scratch:     >= 4 * out->n_read_ops uint32_t slots
  //   read_op_scratch: >= out->n_read_ops slots
  enum damacy_status coalesce_chunks(struct dispatch_output* out,
                                     uint64_t read_op_max_bytes,
                                     uint32_t max_chunks_per_wave,
                                     uint32_t* u32_scratch,
                                     struct read_op* read_op_scratch);

#ifdef __cplusplus
}
#endif
