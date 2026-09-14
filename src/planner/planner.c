#include "planner/planner.h"

#include "prefetch/prefetch_cache.h"

#include <stdlib.h>
#include <string.h>

struct planner
{
  struct planner_config config;
  struct dispatch_scratch scratch;
};

enum damacy_status
planner_create(const struct planner_config* config, struct planner** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!config || !config->array_meta_cache || !config->shard_index_cache ||
      !config->chunk_layout_cache || !config->page_alignment ||
      !config->max_chunks_per_wave || !config->max_substreams_per_chunk)
    return DAMACY_INVAL;
  struct planner* self = calloc(1, sizeof(*self));
  if (!self)
    return DAMACY_OOM;
  self->config = *config;
  *out = self;
  return DAMACY_OK;
}

void
planner_destroy(struct planner* self)
{
  if (self) {
    dispatch_scratch_destroy(&self->scratch);
    free(self);
  }
}

enum damacy_status
planner_plan_segment(struct planner* self,
                     const struct planner_sample* samples,
                     const struct planner_placement* placement,
                     const int64_t* strides,
                     uint8_t rank,
                     struct dispatch_output* out)
{
  if (!self || !samples || !placement || !strides || !out || !out->paths ||
      !out->read_ops || !out->chunk_plans || !out->sample_plans ||
      !placement->n_samples || placement->n_samples > UINT16_MAX || rank < 2 ||
      rank > DAMACY_MAX_RANK + 1)
    return DAMACY_INVAL;
  uint32_t count = placement->n_samples;
  uint32_t begin = placement->sample_idx_begin_in_batch;
  if (count > out->sample_plans_cap || begin > out->sample_plans_cap - count ||
      begin > UINT16_MAX - count)
    return DAMACY_BUDGET;
  struct damacy_batch_spec output = { .dtype = self->config.dst_dtype,
                                      .sample_rank = rank - 1,
                                      .samples_per_batch = count };
  for (uint8_t d = 0; d < output.sample_rank; ++d) {
    int64_t lo = samples[0].aabb.dims[d].beg;
    int64_t hi = samples[0].aabb.dims[d].end;
    if (lo < 0 || hi <= lo)
      return DAMACY_INVAL;
    output.sample_shape[d] = hi - lo;
  }
  struct damacy_plan_limits limits = {
    .max_chunks = out->chunk_plans_cap < out->read_ops_cap
                    ? out->chunk_plans_cap
                    : out->read_ops_cap,
    .max_chunk_bytes =
      self->config.max_chunk_uncompressed_bytes &&
          self->config.max_chunk_uncompressed_bytes < UINT32_MAX
        ? (uint32_t)self->config.max_chunk_uncompressed_bytes
        : UINT32_MAX,
    .max_shards_per_sample = UINT32_MAX,
    .max_plan_bytes = 64ull << 20
  };
  struct prepared_plan* plan = NULL;
  enum damacy_status status =
    prepared_plan_build(self->config.array_meta_cache,
                        self->config.shard_index_cache,
                        samples,
                        count,
                        &output,
                        &limits,
                        &plan);
  if (status != DAMACY_OK)
    return status;
  for (uint32_t i = 0; i < plan->n_chunks; ++i) {
    const struct plan_chunk* chunk = &plan->chunks[i];
    enum compression_codec codec =
      plan->arrays[chunk->array].metadata.inner_codec.id;
    if (!chunk->missing && codec != CODEC_NONE && codec != CODEC_ZSTD &&
        codec != CODEC_BLOSC_ZSTD) {
      status = DAMACY_DECODE;
      goto Done;
    }
  }
  status = dispatch_plan_build(plan,
                               placement->batch_pool_slot,
                               self->config.page_alignment,
                               self->config.read_op_max_bytes,
                               self->config.max_chunks_per_wave,
                               out,
                               &self->scratch);
  if (status != DAMACY_OK)
    goto Done;
  memmove(out->sample_plans + begin,
          out->sample_plans,
          count * sizeof(*out->sample_plans));
  out->n_sample_plans = begin + count;
  for (uint32_t i = 0; i < count; ++i) {
    struct sample_plan* sample = &out->sample_plans[begin + i];
    sample->sample_idx_in_batch = (uint16_t)(begin + i);
    sample->sample_dst_off_elems = (int64_t)(begin + i) * strides[0];
    for (uint8_t d = 0; d < output.sample_rank; ++d)
      sample->dims[d].dst_stride = strides[d + 1];
    const void* layout = NULL;
    int error = 0;
    enum prefetch_state state = prefetch_cache_query(
      self->config.chunk_layout_cache, samples[i].h_layout, &layout, &error);
    if (state != PREFETCH_STATE_READY) {
      status = error ? (enum damacy_status)error : DAMACY_INVAL;
      goto Done;
    }
    if (layout) {
      sample->layout = *(const struct chunk_layout*)layout;
      sample->layout_probed = 1;
    }
  }
  for (uint32_t i = 0; i < out->n_chunk_plans; ++i)
    out->chunk_plans[i].sample_idx_in_batch += (uint16_t)begin;
Done:
  prepared_plan_destroy(plan);
  return status;
}

enum damacy_status
planner_plan(struct planner* self,
             const struct planner_sample* samples,
             uint32_t n_samples,
             uint16_t slot,
             const int64_t* strides,
             uint8_t rank,
             struct dispatch_output* out)
{
  return planner_plan_segment(
    self,
    samples,
    &(struct planner_placement){ .batch_pool_slot = slot,
                                 .n_samples = n_samples },
    strides,
    rank,
    out);
}
