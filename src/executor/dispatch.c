#include "executor/dispatch.h"

#include "executor/coalesce.h"
#include "executor/group_chunks.h"
#include "util/path_intern.h"

#include <stdlib.h>
#include <string.h>

void
dispatch_scratch_destroy(struct dispatch_scratch* scratch)
{
  free(scratch->indices);
  free(scratch->reads);
  free(scratch->chunks);
  *scratch = (struct dispatch_scratch){ 0 };
}

static enum damacy_status
scratch_reserve(struct dispatch_scratch* scratch, uint32_t count)
{
  if (scratch->capacity >= count)
    return DAMACY_OK;
  dispatch_scratch_destroy(scratch);
  scratch->indices = calloc((size_t)count * 4 + 1, sizeof(*scratch->indices));
  scratch->reads = calloc(count, sizeof(*scratch->reads));
  scratch->chunks = calloc(count, sizeof(*scratch->chunks));
  if (!scratch->indices || !scratch->reads || !scratch->chunks) {
    dispatch_scratch_destroy(scratch);
    return DAMACY_OOM;
  }
  scratch->capacity = count;
  return DAMACY_OK;
}

uint32_t
dispatch_index_capacity(const struct damacy_config* config)
{
  if (!config || !config->samples_per_batch || !config->sample_rank ||
      config->sample_rank > DAMACY_MAX_RANK)
    return 0;
  uint64_t capacity =
    config->tuning.max_index_bytes / sizeof(struct gather_index);
  if (capacity > UINT32_MAX)
    capacity = UINT32_MAX;
  uint64_t count = 0;
  for (uint8_t d = 0; d < config->sample_rank; ++d) {
    if (config->sample_shape[d] <= 0)
      return 0;
    uint64_t extent = (uint64_t)config->sample_shape[d];
    if (extent > capacity - count)
      return (uint32_t)capacity;
    count += extent;
  }
  if (config->samples_per_batch && count > capacity / config->samples_per_batch)
    return (uint32_t)capacity;
  return (uint32_t)(count * config->samples_per_batch);
}

uint32_t
dispatch_gather_dim_capacity(const struct damacy_config* config)
{
  return dispatch_index_capacity(config)
           ? DAMACY_MAX_CHUNKS_PER_BATCH * config->sample_rank
           : 0;
}

uint64_t
dispatch_index_storage_bytes(const struct damacy_config* config)
{
  return (uint64_t)dispatch_index_capacity(config) *
           sizeof(struct gather_index) +
         (uint64_t)dispatch_gather_dim_capacity(config) *
           sizeof(struct gather_dim);
}

enum damacy_status
dispatch_plan_build(const struct prepared_plan* plan,
                    uint16_t slot,
                    uint64_t alignment,
                    uint64_t max_read_bytes,
                    uint32_t max_chunks_per_wave,
                    struct dispatch_output* out,
                    struct dispatch_scratch* scratch)
{
  if (!plan || !out || !scratch || !alignment || !max_chunks_per_wave)
    return DAMACY_INVAL;
  if (plan->n_uses > out->chunk_plans_cap || plan->n_uses > out->read_ops_cap ||
      plan->n_regions > out->sample_plans_cap)
    return DAMACY_BUDGET;
  path_intern_reset(out->paths);
  out->n_chunk_plans = out->n_read_ops = out->n_read_op_groups = 0;
  out->n_sample_plans = plan->n_regions;
  out->n_indices = out->n_gather_dims = 0;
  int64_t strides[DAMACY_MAX_RANK + 1];
  strides[plan->output.sample_rank] = 1;
  for (int d = plan->output.sample_rank - 1; d >= 0; --d)
    strides[d] = strides[d + 1] * plan->output.sample_shape[d];
  for (uint32_t i = 0; i < plan->n_regions; ++i) {
    const struct plan_region* region = &plan->regions[i];
    const struct zarr_metadata* meta = &plan->arrays[region->array].metadata;
    struct sample_plan* sample = &out->sample_plans[region->sample];
    *sample = (struct sample_plan){
      .batch_pool_slot = slot,
      .sample_idx_in_batch = (uint16_t)region->sample,
      .rank = meta->rank,
      .src_dtype = (uint8_t)meta->dtype,
      .indexed = region->operation == PLAN_GATHER,
      .sample_dst_off_elems = (int64_t)region->sample * strides[0],
      .chunk_count = region->operation == PLAN_COPY ? 1 : 0
    };
    memcpy(sample->fill_value, meta->fill_value, sizeof(sample->fill_value));
    int64_t source_stride = 1;
    for (int d = meta->rank - 1; d >= 0; --d) {
      uint64_t chunk = meta->inner_chunk_shape[d];
      uint64_t begin = (uint64_t)region->source.dims[d].beg / chunk;
      uint64_t end = ((uint64_t)region->source.dims[d].end - 1) / chunk + 1;
      const struct query_axis* axis = &region->axes[d];
      if (axis->count > out->indices_cap - out->n_indices)
        return DAMACY_BUDGET;
      if (axis->count && !out->indices)
        return DAMACY_INVAL;
      uint32_t index_offset = out->n_indices;
      for (uint32_t j = 0; j < axis->count; ++j)
        out->indices[out->n_indices++] = (struct gather_index){
          .source = (uint32_t)((uint64_t)axis->indices[j].source % chunk),
          .output = axis->indices[j].output
        };
      sample->dims[d] =
        (struct sample_dim){ .chunk_shape = (uint32_t)chunk,
                             .chunk_grid_extent =
                               axis->count ? 0 : (uint32_t)(end - begin),
                             .index_offset = index_offset,
                             .index_count = axis->count,
                             .aabb_lo_relative = region->source.dims[d].beg -
                                                 (int64_t)(begin * chunk),
                             .aabb_extent = plan->output.sample_shape[d],
                             .dst_stride = strides[d + 1],
                             .src_stride = source_stride };
      if (!sample->indexed)
        sample->chunk_count *= (uint32_t)(end - begin);
      source_stride *= (int64_t)chunk;
    }
  }
  const char* source_path = NULL;
  const char* dispatch_path = NULL;
  for (uint32_t i = 0; i < plan->n_uses; ++i) {
    const struct plan_use* use = &plan->uses[i];
    const struct plan_chunk* chunk = &plan->chunks[use->chunk];
    const struct plan_region* region = &plan->regions[use->region];
    const struct zarr_metadata* meta = &plan->arrays[chunk->array].metadata;
    struct read_op* read = &out->read_ops[i];
    *read = (struct read_op){ 0 };
    struct chunk_plan* dispatch = &out->chunk_plans[i];
    *dispatch = (struct chunk_plan){
      .read_op_idx = i,
      .compressed_nbytes = chunk->encoded_bytes,
      .decompressed_nbytes = chunk->decoded_bytes,
      .batch_pool_slot = slot,
      .sample_idx_in_batch = (uint16_t)region->sample,
      .codec_id = chunk->missing ? CODEC_FILL : (uint8_t)meta->inner_codec.id,
      .is_fill = chunk->missing
    };
    struct sample_plan* sample = &out->sample_plans[region->sample];
    if (sample->indexed) {
      ++sample->chunk_count;
      if (meta->rank > out->gather_dims_cap - out->n_gather_dims)
        return DAMACY_BUDGET;
      dispatch->gather_offset = out->n_gather_dims;
      out->n_gather_dims += meta->rank;
    }
    for (uint8_t d = 0; d < meta->rank; ++d) {
      if (!sample->indexed) {
        dispatch->chunk_d[d] = (uint32_t)(chunk->coordinate[d] -
                                          (uint64_t)region->source.dims[d].beg /
                                            meta->inner_chunk_shape[d]);
        continue;
      }
      uint64_t origin = chunk->coordinate[d] * meta->inner_chunk_shape[d];
      struct selection_span span = query_chunk_span(&region->axes[d],
                                                    region->source.dims[d],
                                                    origin,
                                                    meta->inner_chunk_shape[d]);
      out->gather_dims[dispatch->gather_offset + d] = (struct gather_dim){
        .begin = region->axes[d].count
                   ? sample->dims[d].index_offset + (uint32_t)span.begin
                   : (uint32_t)(span.begin - origin),
        .count = (uint32_t)span.count,
        .output_begin = region->axes[d].count
                          ? 0
                          : span.begin - (uint64_t)region->source.dims[d].beg
      };
    }
    if (!chunk->missing) {
      uint64_t start = chunk->offset / alignment * alignment;
      uint64_t end = chunk->offset + chunk->encoded_bytes;
      if (end > UINT64_MAX - alignment + 1)
        return DAMACY_DECODE;
      end = (end + alignment - 1) / alignment * alignment;
      if (end - start > UINT32_MAX)
        return DAMACY_DECODE;
      if (source_path != chunk->path) {
        source_path = chunk->path;
        dispatch_path = path_intern_acquire(out->paths, source_path);
      }
      read->shard_path = dispatch_path;
      if (!read->shard_path)
        return DAMACY_OOM;
      read->file_offset = start;
      read->nbytes = (uint32_t)(end - start);
      dispatch->offset_in_read = (uint32_t)(chunk->offset - start);
    }
  }
  out->n_chunk_plans = out->n_read_ops = plan->n_uses;
  enum damacy_status status = scratch_reserve(scratch, plan->n_uses);
  if (status != DAMACY_OK)
    return status;
  status = coalesce_chunks(
    out, max_read_bytes, max_chunks_per_wave, scratch->indices, scratch->reads);
  if (status != DAMACY_OK)
    return status;
  return group_chunks_by_read(out, scratch->indices, scratch->chunks);
}
