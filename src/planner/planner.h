#pragma once

#include "executor/dispatch.h"
#include "planner/plan_builder.h"

#ifdef __cplusplus
extern "C"
{
#endif

  struct prefetch_cache;

  struct planner_placement
  {
    uint16_t batch_pool_slot;
    uint64_t batch_id;
    uint32_t sample_idx_begin_in_batch;
    uint32_t n_samples;
  };

  struct planner_config
  {
    struct prefetch_cache* array_meta_cache;
    struct prefetch_cache* chunk_layout_cache;
    struct prefetch_cache* shard_index_cache;
    // Source dtype lacking a cast path to this fails planner_plan with
    // DAMACY_DTYPE.
    enum damacy_dtype dst_dtype;
    // Page alignment used for read_op.file_offset / nbytes. Typically
    // platform_page_alignment(), captured once at create.
    uint64_t page_alignment;
    // Runtime ceiling on per-chunk uncompressed bytes. Chunks exceeding
    // this fail planner_plan with DAMACY_BUDGET — earlier than the parse
    // kernel's nblocks check, and surfaces sample.uri to the caller.
    // 0 means "no extra cap beyond DAMACY_MAX_CHUNK_BYTES".
    uint64_t max_chunk_uncompressed_bytes;
    // Cap on the size of any post-coalesce read_op (bytes). Tunes the
    // request-count vs queue-depth tradeoff: bigger = fewer IOs;
    // smaller = more in-flight requests.
    uint64_t read_op_max_bytes;
    // Coalesce caps each leader group at this many chunks (matches
    // wave_pool.max_chunks_per_wave).
    uint32_t max_chunks_per_wave;
    // Parser rejects blosc1 chunks with more sub-streams (DAMACY_DECODE).
    uint32_t max_substreams_per_chunk;
  };

  struct planner;

  enum damacy_status planner_create(const struct planner_config* cfg,
                                    struct planner** out);
  void planner_destroy(struct planner* p);

  enum damacy_status planner_plan(struct planner* p,
                                  const struct planner_sample* samples,
                                  uint32_t n_samples,
                                  uint16_t batch_pool_slot,
                                  const int64_t* dst_strides, // [rank+1]
                                  uint8_t dst_full_rank,      // rank+1
                                  struct dispatch_output* out);

  enum damacy_status planner_plan_segment(
    struct planner* p,
    const struct planner_sample* samples,
    const struct planner_placement* placement,
    const int64_t* dst_strides, // [rank+1]
    uint8_t dst_full_rank,      // rank+1
    struct dispatch_output* out);

#ifdef __cplusplus
}
#endif
