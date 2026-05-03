// Planner: samples × cached zarr metadata → per-chunk read jobs and
// transform records ready for the wave scheduler / IO pool / decompress
// / assemble pipeline.
//
// Build-order step 3: page-aligned reads from day 1, one read_op per
// chunk (no coalescing, no waves). Wave-scheduler fields
// (dst_buf_offset, dev_decompressed_offset) are filled in later by the
// scheduler; the planner zeroes them.
#pragma once

#include "damacy.h" // damacy_status, damacy_sample, damacy_aabb
#include "limits.h" // DAMACY_MAX_RANK

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct zarr_meta_cache;
  struct zarr_shard_cache;

  // Page-aligned IO operation. Multiple chunk_plans may share one
  // read_op once coalescing lands; in step 3, 1:1 with chunk_plans.
  // shard_path lifetime is owned by the planner and remains valid until
  // the next planner_plan call.
  struct read_op
  {
    const char* shard_path; // null-terminated; planner-owned
    uint64_t file_offset;   // multiple of page_alignment
    uint32_t nbytes;        // multiple of page_alignment
    uint32_t _pad0;
    uint64_t dst_buf_offset; // wave-scheduler-assigned; planner sets 0
  };

  // One chunk's full plan: where on disk, where in the output batch.
  // src/dst use the rank-erased damacy_aabb so kernels can iterate
  // uniformly. src.rank == zarr.rank; dst.rank == zarr.rank + 1 with
  // dst.dims[0] = (sample_index_in_batch, +1).
  struct chunk_plan
  {
    uint32_t read_op_idx;
    uint32_t offset_in_read; // chunk start within the read
    uint32_t compressed_nbytes;
    uint32_t decompressed_nbytes;
    uint32_t dev_decompressed_offset; // scheduler-assigned
    uint16_t batch_pool_slot;
    uint8_t _pad0;
    uint8_t _pad1;
    struct damacy_aabb src;               // chunk-local
    struct damacy_aabb dst;               // [N, ...]
    int64_t src_strides[DAMACY_MAX_RANK]; // elements
  };

  struct planner_config
  {
    struct zarr_meta_cache* meta_cache;
    struct zarr_shard_cache* shard_cache;
    // Page alignment used for read_op.file_offset / nbytes. Typically
    // platform_page_alignment(), captured once at create.
    uint64_t page_alignment;
  };

  struct planner;

  enum damacy_status planner_create(const struct planner_config* cfg,
                                    struct planner** out);
  void planner_destroy(struct planner* p);

  // Output buffers for planner_plan. Caller owns the storage; planner
  // populates *_n on success. If either buffer fills before the plan
  // completes, planner_plan returns DAMACY_OOM.
  struct planner_output
  {
    struct read_op* read_ops;
    uint32_t read_ops_cap;
    uint32_t n_read_ops;
    struct chunk_plan* chunk_plans;
    uint32_t chunk_plans_cap;
    uint32_t n_chunk_plans;
  };

  // Plan a batch. samples are processed in order; sample i becomes
  // dst.dims[0] = (i, i+1). All chunk_plans are tagged with
  // batch_pool_slot. Empty chunks (offset == ZARR_SHARD_EMPTY_OFFSET)
  // are skipped — callers should treat the corresponding output region
  // as zeros.
  enum damacy_status planner_plan(struct planner* p,
                                  const struct damacy_sample* samples,
                                  uint32_t n_samples,
                                  uint16_t batch_pool_slot,
                                  struct planner_output* out);

#ifdef __cplusplus
}
#endif
