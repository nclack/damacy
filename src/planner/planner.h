// Planner: samples × cached zarr metadata → per-chunk read jobs and
// transform records ready for the wave scheduler / IO pool / decompress
// / assemble pipeline.
//
// Build-order step 3: page-aligned reads from day 1, one read_op per
// chunk (no coalescing, no waves). Wave-scheduler fields
// (dst_buf_offset, dev_decompressed_offset) are filled in later by the
// scheduler; the planner zeroes them.
#pragma once

#include "damacy.h"                 // damacy_status, damacy_sample, damacy_aabb
#include "damacy_limits.h"          // DAMACY_MAX_RANK
#include "zarr/zarr_chunk_layout.h" // struct chunk_layout
#include "zarr/zarr_metadata.h"     // DAMACY_MAX_DTYPE_BYTES

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct zarr_meta_cache;
  struct zarr_shard_cache;

  // Page-aligned IO operation. Multiple chunk_plans may share one
  // read_op once coalescing lands; pre-step-7 it's 1:1 with chunk_plans.
  // shard_path is inlined so a read_op survives across batches in the
  // wave scheduler's plan queue (step 5+).
  struct read_op
  {
    char shard_path[DAMACY_MAX_PATH]; // null-terminated; truncation = OOM
    uint64_t file_offset;             // multiple of page_alignment
    uint32_t nbytes;                  // multiple of page_alignment
    uint64_t dst_buf_offset;          // wave-scheduler-assigned; planner sets 0
  };

  // Per-dimension bundle for one sample. Co-locating all of dimension d's
  // parameters lets the assemble kernel's R-loop touch one cache line per
  // iteration instead of seven scattered loads.
  struct sample_dim
  {
    uint32_t chunk_shape;       // S[d] — uniform per source, no clipping
    uint32_t chunk_grid_extent; // N[d] — chunks per dimension within sample
    int64_t aabb_lo_relative;   // aabb_lo[d] − chunk_grid_origin[d], in [0,S)
    int64_t aabb_extent;        // sample AABB extent
    int64_t dst_stride;         // batch tensor stride (elements)
    int64_t src_stride;         // chunk-local row-major stride (elements)
  };

  // Per-sample header consumed by assemble. All chunks within the sample
  // share these constants (uniform shape, single source). Per-chunk
  // records are reduced to {dev_decompressed_offset, chunk_d}.
  struct sample_plan
  {
    uint16_t batch_pool_slot;
    uint16_t sample_idx_in_batch;
    uint8_t rank;                            // spatial rank
    uint8_t src_dtype;                       // enum dtype; source zarr type
    struct sample_dim dims[DAMACY_MAX_RANK]; // dims[0..rank)
    int64_t sample_dst_off_elems; // sample slot start (elements; * dst bpe at
                                  // runtime)
    uint32_t chunk_offset;        // first chunk in chunk_plans
    uint32_t chunk_count;         // ∏ dims[d].chunk_grid_extent
    // Array-level fill_value (zarr v3 metadata). Chunks tagged is_fill
    // broadcast these bytes; bytes are interpreted under src_dtype.
    uint8_t fill_value[DAMACY_MAX_DTYPE_BYTES];
    // Per-array blosc1 chunk layout, populated lazily from the meta
    // cache on first non-fill emit. layout_probed = 0 means downstream
    // should fall back to MAX_BLOCKS_PER_CHUNK in cap calculations.
    struct chunk_layout layout;
    uint8_t layout_probed;
  };

  // Per-chunk plan. Carries IO/decompress fields plus assemble-side
  // chunk_d (grid position within sample, 0..N[d]) and sample_idx so
  // the kernel can look up the sample_plan.
  //
  // is_fill marks chunks that are absent from the store (sparse zarr v3):
  // read_op_idx / offset_in_read / compressed_nbytes are unused, the
  // codec stage is skipped, and assemble broadcasts the sample's
  // fill_value across the chunk's region instead of reading the arena.
  // decompressed_nbytes is still set so wave accounting tracks the
  // conceptual chunk size.
  struct chunk_plan
  {
    uint32_t read_op_idx;
    uint32_t offset_in_read; // chunk start within the read
    uint32_t compressed_nbytes;
    uint32_t decompressed_nbytes;
    uint32_t dev_decompressed_offset; // scheduler-assigned (per wave)
    uint16_t batch_pool_slot;
    uint16_t sample_idx_in_batch; // index into planner_output.sample_plans
    uint8_t codec_id;
    uint8_t is_fill; // 1 = absent chunk; fill_value lives on the sample
    uint32_t chunk_d[DAMACY_MAX_RANK]; // grid position within sample (0..N)
  };

  struct planner_config
  {
    struct zarr_meta_cache* meta_cache;
    struct zarr_shard_cache* shard_cache;
    // Page alignment used for read_op.file_offset / nbytes. Typically
    // platform_page_alignment(), captured once at create.
    uint64_t page_alignment;
    // Runtime ceiling on per-chunk uncompressed bytes. Chunks exceeding
    // this fail planner_plan with DAMACY_INVAL — earlier than the parse
    // kernel's nblocks check, and surfaces sample.uri to the caller.
    // 0 means "no extra cap beyond DAMACY_MAX_CHUNK_BYTES".
    uint64_t max_chunk_uncompressed_bytes;
  };

  struct planner;

  enum damacy_status planner_create(const struct planner_config* cfg,
                                    struct planner** out);
  void planner_destroy(struct planner* p);

  // Output buffers for planner_plan. Caller owns the storage; planner
  // populates *_n on success. If any buffer fills before the plan
  // completes, planner_plan returns DAMACY_OOM.
  struct planner_output
  {
    struct read_op* read_ops;
    uint32_t read_ops_cap;
    uint32_t n_read_ops;
    struct chunk_plan* chunk_plans;
    uint32_t chunk_plans_cap;
    uint32_t n_chunk_plans;
    struct sample_plan* sample_plans;
    uint32_t sample_plans_cap;
    uint32_t n_sample_plans;
  };

  // Plan one training batch — i.e., the N samples that land in one
  // output tensor of shape [N, ...zarr_axes]. samples are processed in
  // order; sample i becomes sample_plans[i].sample_idx_in_batch == i.
  // All chunk_plans are tagged with batch_pool_slot so the scheduler
  // knows which slot in the batch pool receives the assembled output.
  //
  // This is independent of the scheduler's wave granularity: the wave
  // scheduler chunks the produced plan queue into wave-sized dispatch
  // units (one batch typically spans multiple waves). The planner is
  // batch-shaped; waves are a downstream concern.
  //
  // Empty inner chunks (offset == nbytes == 0xFFFF…) and shards that
  // don't exist in the store are emitted as fill-mode chunk_plans
  // carrying the array's fill_value; downstream skips IO/decompress and
  // assemble broadcasts the fill bytes over the chunk's region.
  //
  // shard_path strings are copied into each emitted read_op (inline
  // storage, capped at DAMACY_MAX_PATH); the planner no longer
  // retains string ownership across calls.
  enum damacy_status planner_plan(struct planner* p,
                                  const struct damacy_sample* samples,
                                  uint32_t n_samples,
                                  uint16_t batch_pool_slot,
                                  const int64_t* dst_strides, // [rank+1]
                                  uint8_t dst_full_rank,      // rank+1
                                  struct planner_output* out);

#ifdef __cplusplus
}
#endif
