#pragma once

#include "planner/plan.h"
#include "zarr/zarr_chunk_layout.h"

struct path_intern;

#ifdef __cplusplus
extern "C"
{
#endif

  struct read_op
  {
    const char* shard_path;
    uint64_t file_offset;
    uint64_t host_buf_offset;
    uint32_t nbytes;
  };

  struct gather_index
  {
    uint32_t source;
    uint32_t output;
  };

  struct gather_dim
  {
    uint32_t begin;
    uint32_t count;
    uint64_t output_begin;
  };

  struct sample_dim
  {
    uint32_t chunk_shape;
    uint32_t chunk_grid_extent;
    uint32_t index_offset;
    uint32_t index_count;
    int64_t aabb_lo_relative;
    int64_t aabb_extent;
    int64_t dst_stride;
    int64_t src_stride;
  };

  struct sample_plan
  {
    uint16_t batch_pool_slot;
    uint16_t sample_idx_in_batch;
    uint8_t rank;
    uint8_t src_dtype;
    uint8_t indexed;
    const struct gather_index* indices;
    const struct gather_dim* gather_dims;
    struct sample_dim dims[DAMACY_MAX_RANK];
    int64_t sample_dst_off_elems;

    uint32_t chunk_count;

    uint8_t fill_value[DAMACY_MAX_DTYPE_BYTES];

    struct chunk_layout layout;
    uint8_t layout_probed;
  };

  struct chunk_plan
  {
    uint32_t read_op_idx;
    uint32_t offset_in_read;
    uint32_t compressed_nbytes;
    uint32_t decompressed_nbytes;
    uint64_t host_buf_offset;
    uint32_t dev_decompressed_offset;
    uint32_t gather_offset;
    uint16_t batch_pool_slot;
    uint16_t sample_idx_in_batch;
    uint8_t codec_id;
    uint8_t is_fill;
    uint32_t chunk_d[DAMACY_MAX_RANK];
  };

  struct read_op_group
  {
    uint32_t read_op_idx;
    uint32_t first_chunk;
    uint32_t n_chunks;
    uint64_t total_decompressed;
  };

  struct read_op_group_iterator
  {
    const struct read_op_group* groups;
    uint32_t n_groups;
    uint32_t cursor;
  };

  void read_op_group_iterator_init(struct read_op_group_iterator* it,
                                   const struct read_op_group* groups,
                                   uint32_t n_groups,
                                   uint32_t start_group);
  int read_op_group_iterator_next(struct read_op_group_iterator* it,
                                  struct read_op_group* out);

  struct dispatch_output
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
    struct read_op_group* read_op_groups;
    uint32_t read_op_groups_cap;
    uint32_t n_read_op_groups;
    struct gather_index* indices;
    uint32_t indices_cap;
    uint32_t n_indices;
    struct gather_dim* gather_dims;
    uint32_t gather_dims_cap;
    uint32_t n_gather_dims;
    struct path_intern* paths;
    uint32_t n_chunks_to_load;
    uint32_t n_loads_issued;
  };

  struct dispatch_scratch
  {
    uint32_t* indices;
    struct read_op* reads;
    struct chunk_plan* chunks;
    uint32_t capacity;
  };

  uint32_t dispatch_index_capacity(const struct damacy_config* config);
  uint32_t dispatch_gather_dim_capacity(const struct damacy_config* config);
  uint64_t dispatch_index_storage_bytes(const struct damacy_config* config);

  enum damacy_status dispatch_plan_build(const struct prepared_plan* plan,
                                         uint16_t slot,
                                         uint64_t alignment,
                                         uint64_t max_read_bytes,
                                         uint32_t max_chunks_per_wave,
                                         struct dispatch_output* out,
                                         struct dispatch_scratch* scratch);
  void dispatch_scratch_destroy(struct dispatch_scratch* scratch);

#ifdef __cplusplus
}
#endif
