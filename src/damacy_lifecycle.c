#include "damacy_internal.h"

#include "damacy_config.h"
#include "damacy_stats.h"

#ifdef DAMACY_HAS_CUDA
#include "executor/cuda_executor.h"
#endif

#include <stdlib.h>
#include <string.h>

enum damacy_status
damacy_pipeline_create(struct damacy_planner* planner,
                       struct damacy_executor* executor,
                       const struct damacy_batch_spec* output,
                       const struct damacy_queue_limits* queues,
                       struct damacy** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!planner || !executor || !queues || !output || planner->active ||
      executor->active || !queues->prepared_batches ||
      queues->prepared_batches > 1024 ||
      queues->lookahead_samples < output->samples_per_batch ||
      queues->lookahead_samples > UINT32_MAX - 4)
    return DAMACY_INVAL;
  int64_t shape[DAMACY_MAX_RANK + 1], strides[DAMACY_MAX_RANK + 1];
  uint64_t bytes;
  enum damacy_status status = batch_spec_layout(output, shape, strides, &bytes);
  if (status != DAMACY_OK)
    return status;
  struct damacy* self = calloc(1, sizeof(*self));
  if (!self)
    return DAMACY_OOM;
  int planner_claimed = 0, executor_claimed = 0;
  int planner_started = 0, executor_started = 0;
  self->output = *output;
  self->queues = *queues;
  self->planner = planner;
  self->executor = executor;
  stats_init(&self->stats);
  self->plans = calloc(queues->prepared_batches, sizeof(*self->plans));
  if (!self->plans) {
    status = DAMACY_OOM;
    goto Fail;
  }
  int expected = 0;
  if (!atomic_compare_exchange_strong(&planner->active, &expected, 1)) {
    status = DAMACY_INVAL;
    goto Fail;
  }
  planner_claimed = 1;
  expected = 0;
  if (!atomic_compare_exchange_strong(&executor->active, &expected, 1)) {
    status = DAMACY_INVAL;
    goto Fail;
  }
  executor_claimed = 1;
  status = planner->ops->start(planner, output, queues);
  if (status != DAMACY_OK)
    goto Fail;
  planner_started = 1;
  status = executor->ops->start(executor, output, &self->stats);
  if (status != DAMACY_OK)
    goto Fail;
  executor_started = 1;
  self->device =
    executor->device_type == DAMACY_DEVICE_CUDA ? executor->device_id : -1;
  self->sched =
    scheduler_create(damacy_scheduler_step,
                     self,
                     10000,
                     NULL,
                     &(struct scheduler_hooks){ damacy_scheduler_enter,
                                                damacy_scheduler_leave });
  if (!self->sched) {
    status = DAMACY_OOM;
    goto Fail;
  }
  *out = self;
  return DAMACY_OK;
Fail:
  if (executor_started)
    executor->ops->stop(executor);
  if (planner_started)
    planner->ops->stop(planner);
  if (executor_claimed)
    executor->active = 0;
  if (planner_claimed)
    planner->active = 0;
  free(self->plans);
  free(self);
  return status;
}

void
damacy_shutdown(struct damacy* self)
{
  if (!self || self->stopped)
    return;
  scheduler_lock(self->sched);
  if (self->stopping) {
    while (!self->stopped)
      scheduler_wait(self->sched);
    scheduler_unlock(self->sched);
    return;
  }
  self->planner->ops->stats(self->planner, &self->stats);
  self->executor->ops->stats(self->executor, &self->stats);
  self->stopping = 1;
  scheduler_broadcast(self->sched);
  while (self->pop_calls)
    scheduler_wait(self->sched);
  scheduler_unlock(self->sched);
  scheduler_stop(self->sched);
  self->planner->ops->stop(self->planner);
  self->executor->ops->stop(self->executor);
  self->planner->active = self->executor->active = 0;
  for (uint32_t i = 0; i < self->queues.prepared_batches; ++i) {
    prepared_plan_destroy(self->plans[i]);
    self->plans[i] = NULL;
  }
  scheduler_lock(self->sched);
  self->plan_count = 0;
  self->stopped = 1;
  scheduler_broadcast(self->sched);
  scheduler_unlock(self->sched);
}

void
damacy_destroy(struct damacy* self)
{
  if (!self)
    return;
  damacy_shutdown(self);
  scheduler_destroy(self->sched);
  if (self->owns_components) {
    damacy_executor_destroy(self->executor);
    damacy_planner_destroy(self->planner);
    damacy_metadata_destroy(self->owned_metadata);
    damacy_metadata_reader_destroy(self->owned_metadata_reader);
    damacy_reader_destroy(self->owned_reader);
  }
  free(self->plans);
  free(self);
}

int
damacy_get_device(const struct damacy* self)
{
  return self ? self->device : -1;
}

enum damacy_status
damacy_create(const struct damacy_config* config, struct damacy** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  enum damacy_status status = validate_config(config);
  if (status != DAMACY_OK)
    return status;
  struct damacy_reader* reader = NULL;
  struct damacy_metadata_reader* metadata_reader = NULL;
  struct damacy_metadata* metadata = NULL;
  struct damacy_planner* planner = NULL;
  struct damacy_executor* executor = NULL;
  const struct damacy_tuning* tuning = &config->tuning;
#ifdef DAMACY_HAS_CUDA
  struct numa_resolved affinity;
  struct platform_cpu_mask saved;
  cuda_resolve_numa(config, &affinity);
  numa_scope_enter(&affinity, &saved);
#endif
  status = damacy_file_reader_create(tuning->n_io_threads,
                                     (uint32_t)tuning->host_buffer_waves *
                                         tuning->max_chunks_per_wave +
                                       DAMACY_READ_JOB_SLACK,
                                     &reader);
  if (status != DAMACY_OK)
    goto Done;
  status = damacy_file_metadata_reader_create(tuning->metadata_io_concurrency,
                                              &config->debug.metadata_latency,
                                              &metadata_reader);
  if (status != DAMACY_OK)
    goto Done;
  status = damacy_zarr_metadata_create(
    metadata_reader,
    &(struct damacy_metadata_cache_config){
      .array_entries = tuning->n_array_meta_cache,
      .shard_entries = tuning->n_shard_index_cache },
    &metadata);
  if (status != DAMACY_OK)
    goto Done;
  status = damacy_chunk_planner_create(
    metadata,
    &(struct damacy_plan_limits){
      .max_chunks = DAMACY_MAX_CHUNKS_PER_BATCH,
      .max_chunk_bytes = tuning->max_chunk_uncompressed_bytes,
      .max_shards_per_sample = tuning->max_shards_per_sample,
      .max_plan_bytes = 64ull << 20 },
    &planner);
  if (status != DAMACY_OK)
    goto Done;
  status = damacy_cuda_executor_create(
    reader,
    &(struct damacy_cuda_config){
      .device = config->device,
      .max_gpu_memory_bytes = tuning->max_gpu_memory_bytes,
      .max_chunk_bytes = tuning->max_chunk_uncompressed_bytes,
      .max_read_bytes = tuning->max_read_op_bytes,
      .max_chunks_per_wave = tuning->max_chunks_per_wave,
      .max_substreams_per_chunk = tuning->max_substreams_per_chunk,
      .host_buffer_waves = tuning->host_buffer_waves,
      .chunk_layout_entries = tuning->n_chunk_layout_cache,
      .numa_strategy = tuning->numa_strategy,
      .numa_node = tuning->numa_node,
      .enable_gds = tuning->enable_gds,
      .bypass_decode = config->debug.bypass_decode },
    &executor);
  if (status != DAMACY_OK)
    goto Done;
  struct damacy_batch_spec output = { .dtype = config->dtype,
                                      .sample_rank = config->sample_rank,
                                      .samples_per_batch =
                                        config->samples_per_batch };
  memcpy(
    output.sample_shape, config->sample_shape, sizeof(output.sample_shape));
  status = damacy_pipeline_create(
    planner,
    executor,
    &output,
    &(struct damacy_queue_limits){
      .lookahead_samples = config->lookahead_samples, .prepared_batches = 2 },
    out);
  if (status == DAMACY_OK) {
    (*out)->owns_components = 1;
    (*out)->owned_reader = reader;
    (*out)->owned_metadata_reader = metadata_reader;
    (*out)->owned_metadata = metadata;
#ifdef DAMACY_HAS_CUDA
    numa_scope_exit(&saved);
#endif
    return DAMACY_OK;
  }
Done:
#ifdef DAMACY_HAS_CUDA
  numa_scope_exit(&saved);
#endif
  damacy_executor_destroy(executor);
  damacy_planner_destroy(planner);
  damacy_metadata_destroy(metadata);
  damacy_metadata_reader_destroy(metadata_reader);
  damacy_reader_destroy(reader);
  return status;
}

#ifdef DAMACY_HAS_CUDA
uint64_t
damacy_set_gpu_bytes_committed_for_test(struct damacy* self, uint64_t value)
{
  return self ? cuda_executor_set_budget(self->executor, value) : 0;
}
#endif
