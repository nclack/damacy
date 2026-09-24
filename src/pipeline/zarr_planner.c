#include "pipeline/components.h"

#include "log/log.h"
#include "lookahead/lookahead.h"
#include "planner/plan_builder.h"
#include "prefetch/array_meta.h"
#include "prefetch/prefetcher.h"
#include "prefetch/shard_index.h"
#include "store/metadata_store_async.h"

#include <stdlib.h>
#include <string.h>

struct zarr_planner
{
  struct damacy_planner base;
  struct damacy_metadata metadata;
  struct damacy_plan_limits limits;
  struct damacy_batch_spec output;
  struct damacy_queue_limits queues;
  struct metadata_store_async* reader;
  struct array_meta_async_fetcher array_fetcher;
  struct shard_index_async_fetcher shard_fetcher;
  struct prefetch_cache* arrays;
  struct prefetch_cache* shards;
  struct damacy_lookahead lookahead;
  struct prefetcher* prefetcher;
  struct planner_sample* samples;
  uint32_t staged;
  uint64_t pushed;
  uint64_t planned;
  uint64_t watermark;
};

static void
clear_samples(struct zarr_planner* self)
{
  for (uint32_t i = 0; i < self->staged; ++i) {
    free((char*)self->samples[i].uri);
    free(self->samples[i].h_shards);
    self->samples[i] = (struct planner_sample){ 0 };
  }
  self->staged = 0;
}

static void
zarr_stop(struct damacy_planner* base)
{
  struct zarr_planner* self = (void*)base;
  prefetcher_stop(self->prefetcher);
  metadata_store_async_destroy(self->reader);
  self->reader = NULL;
  prefetcher_destroy(self->prefetcher);
  self->prefetcher = NULL;
  prefetch_cache_destroy(self->shards);
  self->shards = NULL;
  prefetch_cache_destroy(self->arrays);
  self->arrays = NULL;
  lookahead_destroy(&self->lookahead);
  clear_samples(self);
  free(self->samples);
  self->samples = NULL;
}

static enum damacy_status
zarr_start(struct damacy_planner* base,
           const struct damacy_batch_spec* output,
           const struct damacy_queue_limits* queues)
{
  struct zarr_planner* self = (void*)base;
  const struct damacy_metadata* metadata = &self->metadata;
  uint64_t floor =
    (uint64_t)queues->lookahead_samples + output->samples_per_batch;
  if (metadata->cache.array_entries < floor ||
      metadata->cache.shard_entries <
        floor * self->limits.max_shards_per_sample) {
    log_error(
      "metadata cache requires array_entries >= %llu and shard_entries >= %llu",
      (unsigned long long)floor,
      (unsigned long long)(floor * self->limits.max_shards_per_sample));
    return DAMACY_INVAL;
  }
  self->output = *output;
  self->queues = *queues;
  self->pushed = self->planned = self->watermark = 0;
  self->reader = metadata_store_async_create(
    (int)metadata->reader.concurrency, NULL, &metadata->reader.latency);
  if (!self->reader)
    goto Fail;
  array_meta_async_fetcher_init(&self->array_fetcher, self->reader);
  self->arrays = prefetch_cache_create(&(struct prefetch_cache_config){
    .capacity = metadata->cache.array_entries,
    .max_probe = 16,
    .knob_name = "array_entries",
    .ops = &array_meta_ops,
    .async_fetcher = &self->array_fetcher.base });
  if (!self->arrays)
    goto Fail;
  shard_index_async_fetcher_init(
    &self->shard_fetcher, self->reader, self->arrays);
  self->shards = prefetch_cache_create(&(struct prefetch_cache_config){
    .capacity = metadata->cache.shard_entries,
    .max_probe = 16,
    .knob_name = "shard_entries",
    .ops = &shard_index_ops,
    .async_fetcher = &self->shard_fetcher.base });
  if (!self->shards ||
      lookahead_init(&self->lookahead, queues->lookahead_samples))
    goto Fail;
  self->samples = calloc(output->samples_per_batch, sizeof(*self->samples));
  if (!self->samples)
    goto Fail;
  self->prefetcher = prefetcher_create(&(struct prefetcher_config){
    .lookahead = &self->lookahead,
    .array_meta_cache = self->arrays,
    .shard_index_cache = self->shards,
    .capacity = queues->lookahead_samples,
    .owner_capacity = queues->lookahead_samples + 4,
    .max_shards_per_sample = self->limits.max_shards_per_sample });
  if (!self->prefetcher || prefetcher_start(self->prefetcher))
    goto Fail;
  return DAMACY_OK;
Fail:
  zarr_stop(base);
  return DAMACY_OOM;
}

static struct damacy_push_result
zarr_push(struct damacy_planner* base, struct damacy_sample_slice samples)
{
  struct zarr_planner* self = (void*)base;
  struct damacy_push_result result = { .unconsumed = samples,
                                       .status = DAMACY_OK };
  for (const struct damacy_sample* sample = samples.beg; sample != samples.end;
       ++sample) {
    result.unconsumed.beg = sample;
    if (prefetcher_unconsumed_count(self->prefetcher, self->pushed) >=
        self->queues.lookahead_samples) {
      result.status = DAMACY_AGAIN;
      return result;
    }
    result.status =
      query_validate(sample, &self->output, self->limits.max_plan_bytes);
    if (result.status != DAMACY_OK)
      return result;
    if (lookahead_push_with_sample_seq(
          &self->lookahead, sample, self->pushed)) {
      result.status = DAMACY_OOM;
      return result;
    }
    ++self->pushed;
  }
  result.unconsumed.beg = samples.end;
  return result;
}

static enum damacy_status
zarr_next(struct damacy_planner* base, struct prepared_plan** out)
{
  struct zarr_planner* self = (void*)base;
  *out = NULL;
  while (self->staged < self->output.samples_per_batch) {
    struct prefetcher_ready ready = { 0 };
    if (!prefetcher_pop_ready(self->prefetcher, &ready))
      return DAMACY_AGAIN;
    if (ready.result == PREFETCHER_RESULT_ERROR) {
      enum damacy_status status =
        ready.err_code ? (enum damacy_status)ready.err_code : DAMACY_INVAL;
      prefetcher_ready_free(&ready);
      return status;
    }
    self->samples[self->staged++] =
      (struct planner_sample){ .uri = ready.uri,
                               .aabb = ready.aabb,
                               .h_meta = ready.h_meta,
                               .h_shards = ready.h_shards,
                               .n_shards = ready.n_shards };
    memcpy(
      self->samples[self->staged - 1].axes, ready.axes, sizeof(ready.axes));
    self->watermark = ready.sample_seq + 1;
  }
  enum damacy_status status = prepared_plan_build(self->arrays,
                                                  self->shards,
                                                  self->samples,
                                                  self->staged,
                                                  &self->output,
                                                  &self->limits,
                                                  out);
  if (status == DAMACY_AGAIN)
    return status;
  self->planned += self->staged;
  clear_samples(self);
  prefetcher_advance_watermark(self->prefetcher, self->watermark);
  return status;
}

static uint64_t
zarr_pending(const struct damacy_planner* base)
{
  const struct zarr_planner* self = (const void*)base;
  return self->pushed - self->planned;
}

static void
zarr_stats(struct damacy_planner* base, struct damacy_stats* out)
{
  struct zarr_planner* self = (void*)base;
  struct prefetch_cache_stats cache;
  if (self->arrays) {
    prefetch_cache_stats_get(self->arrays, &cache);
    out->array_meta.hits = cache.counters.hits;
    out->array_meta.misses = cache.counters.misses;
  }
  if (self->shards) {
    prefetch_cache_stats_get(self->shards, &cache);
    out->shard_index.hits = cache.counters.hits;
    out->shard_index.misses = cache.counters.misses;
  }
  if (!self->reader)
    return;
  struct metadata_store_async_latency_stats latency;
  metadata_store_async_latency_stats_get(self->reader, &latency);
  out->metadata_latency.ops = latency.ops;
  out->metadata_latency.stat_ops = latency.stat_ops;
  out->metadata_latency.submit_ops = latency.submit_ops;
  out->metadata_latency.active = latency.active;
  out->metadata_latency.max_active = latency.max_active;
  out->metadata_latency.total_sleep_ns = latency.total_sleep_ns;
  out->metadata_latency.max_sleep_ns = latency.max_sleep_ns;
  struct metadata_store_async_backend_stats backend;
  metadata_store_async_backend_stats_get(self->reader, &backend);
  out->metadata_backend.read_jobs = backend.read_jobs;
  out->metadata_backend.read_active = backend.read_active;
  out->metadata_backend.read_max_active = backend.read_max_active;
  struct metadata_store_async_op_latency_stats operations;
  metadata_store_async_op_latency_stats_get(self->reader, &operations);
  for (unsigned i = 0; i < DAMACY_METADATA_OP_LATENCY_NKINDS; ++i) {
    out->metadata_op_latency[i].count = operations.kinds[i].count;
    out->metadata_op_latency[i].sum_ns = operations.kinds[i].sum_ns;
    out->metadata_op_latency[i].max_ns = operations.kinds[i].max_ns;
    for (unsigned j = 0; j < DAMACY_METADATA_OP_LATENCY_NBUCKETS; ++j)
      out->metadata_op_latency[i].buckets[j] = operations.kinds[i].buckets[j];
  }
}

static void
zarr_reset_stats(struct damacy_planner* base)
{
  struct zarr_planner* self = (void*)base;
  metadata_store_async_latency_stats_reset(self->reader);
  metadata_store_async_backend_stats_reset(self->reader);
  metadata_store_async_op_latency_stats_reset(self->reader);
}

static void
zarr_destroy(struct damacy_planner* base)
{
  zarr_stop(base);
  free(base);
}

static const struct damacy_planner_ops zarr_ops = { .start = zarr_start,
                                                    .push = zarr_push,
                                                    .next = zarr_next,
                                                    .pending = zarr_pending,
                                                    .stats = zarr_stats,
                                                    .reset_stats =
                                                      zarr_reset_stats,
                                                    .stop = zarr_stop,
                                                    .destroy = zarr_destroy };

enum damacy_status
damacy_chunk_planner_create(struct damacy_metadata* metadata,
                            const struct damacy_plan_limits* limits,
                            struct damacy_planner** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!metadata || !limits || !limits->max_chunks ||
      limits->max_chunks > DAMACY_MAX_CHUNKS_PER_BATCH ||
      !limits->max_chunk_bytes || !limits->max_shards_per_sample ||
      !limits->max_plan_bytes)
    return DAMACY_INVAL;
  struct zarr_planner* self = calloc(1, sizeof(*self));
  if (!self)
    return DAMACY_OOM;
  self->base.ops = &zarr_ops;
  self->metadata = *metadata;
  self->limits = *limits;
  *out = &self->base;
  return DAMACY_OK;
}
