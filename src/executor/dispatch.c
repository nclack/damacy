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
  int64_t strides[DAMACY_MAX_RANK + 1];
  strides[plan->output.sample_rank] = 1;
  for (int d = plan->output.sample_rank - 1; d >= 0; --d)
    strides[d] = strides[d + 1] * plan->output.sample_shape[d];
  for (uint32_t i = 0; i < plan->n_regions; ++i) {
    const struct plan_region* region = &plan->regions[i];
    const struct zarr_metadata* meta = &plan->arrays[region->array].metadata;
    struct sample_plan* sample = &out->sample_plans[region->sample];
    *sample =
      (struct sample_plan){ .batch_pool_slot = slot,
                            .sample_idx_in_batch = (uint16_t)region->sample,
                            .rank = meta->rank,
                            .src_dtype = (uint8_t)meta->dtype,
                            .sample_dst_off_elems =
                              (int64_t)region->sample * strides[0],
                            .chunk_count = 1 };
    memcpy(sample->fill_value, meta->fill_value, sizeof(sample->fill_value));
    int64_t source_stride = 1;
    for (int d = meta->rank - 1; d >= 0; --d) {
      uint64_t chunk = meta->inner_chunk_shape[d];
      uint64_t begin = (uint64_t)region->source.dims[d].beg / chunk;
      uint64_t end = ((uint64_t)region->source.dims[d].end - 1) / chunk + 1;
      sample->dims[d] = (struct sample_dim){
        .chunk_shape = (uint32_t)chunk,
        .chunk_grid_extent = (uint32_t)(end - begin),
        .aabb_lo_relative =
          region->source.dims[d].beg - (int64_t)(begin * chunk),
        .aabb_extent = region->source.dims[d].end - region->source.dims[d].beg,
        .dst_stride = strides[d + 1],
        .src_stride = source_stride
      };
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
    for (uint8_t d = 0; d < meta->rank; ++d)
      dispatch->chunk_d[d] =
        (uint32_t)(chunk->coordinate[d] - (uint64_t)region->source.dims[d].beg /
                                            meta->inner_chunk_shape[d]);
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
