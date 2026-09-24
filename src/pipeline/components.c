#include "pipeline/components.h"

#include "damacy_config.h"
#include "log/log.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum damacy_status
batch_spec_layout(const struct damacy_batch_spec* spec,
                  int64_t* shape,
                  int64_t* strides,
                  uint64_t* bytes)
{
  if (!spec || !shape || !strides || !bytes || !spec->samples_per_batch ||
      spec->samples_per_batch > UINT16_MAX || !spec->sample_rank ||
      spec->sample_rank > DAMACY_MAX_RANK || !damacy_dtype_bpe(spec->dtype))
    return DAMACY_INVAL;
  shape[0] = spec->samples_per_batch;
  uint64_t elements = spec->samples_per_batch;
  for (uint8_t d = 0; d < spec->sample_rank; ++d) {
    int64_t extent = spec->sample_shape[d];
    if (extent <= 0 || elements > INT64_MAX / (uint64_t)extent)
      return DAMACY_INVAL;
    elements *= (uint64_t)extent;
    shape[d + 1] = extent;
  }
  if (elements > SIZE_MAX / damacy_dtype_bpe(spec->dtype))
    return DAMACY_BUDGET;
  strides[spec->sample_rank] = 1;
  for (int d = spec->sample_rank - 1; d >= 0; --d)
    strides[d] = strides[d + 1] * shape[d + 1];
  *bytes = elements * damacy_dtype_bpe(spec->dtype);
  return DAMACY_OK;
}

enum damacy_status
damacy_file_reader_create(uint32_t workers,
                          uint32_t max_inflight_reads,
                          struct damacy_reader** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!workers || workers > DAMACY_MAX_IO_THREADS || !max_inflight_reads)
    return DAMACY_INVAL;
  int cpus = platform_default_thread_count();
  if (workers > (uint32_t)cpus) {
    log_error("file reader workers (n_io_threads)=%u exceeds the %d online "
              "CPUs; set it to at most %d",
              workers,
              cpus,
              cpus);
    return DAMACY_INVAL;
  }
  struct damacy_reader* reader = calloc(1, sizeof(*reader));
  if (!reader)
    return DAMACY_OOM;
  reader->max_inflight_reads = max_inflight_reads;
  reader->store = store_fs_create(
    &(struct store_fs_config){ .root = "",
                               .nthreads = (int)workers,
                               .max_inflight_reads = max_inflight_reads });
  if (!reader->store) {
    free(reader);
    return DAMACY_OOM;
  }
  *out = reader;
  return DAMACY_OK;
}

void
damacy_reader_destroy(struct damacy_reader* reader)
{
  if (reader) {
    store_destroy(reader->store);
    free(reader);
  }
}

enum damacy_status
damacy_file_metadata_reader_create(uint32_t concurrency,
                                   const struct damacy_latency_model* latency,
                                   struct damacy_metadata_reader** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!concurrency || concurrency > DAMACY_MAX_METADATA_IO_CONCURRENCY ||
      (latency && (!isfinite(latency->lognormal_mu_ln_ns) ||
                   !isfinite(latency->lognormal_sigma_ln_ns) ||
                   latency->lognormal_sigma_ln_ns < 0)))
    return DAMACY_INVAL;
  struct damacy_metadata_reader* reader = calloc(1, sizeof(*reader));
  if (!reader)
    return DAMACY_OOM;
  reader->concurrency = concurrency;
  if (latency)
    reader->latency = *latency;
  *out = reader;
  return DAMACY_OK;
}

void
damacy_metadata_reader_destroy(struct damacy_metadata_reader* reader)
{
  free(reader);
}

enum damacy_status
damacy_zarr_metadata_create(struct damacy_metadata_reader* reader,
                            const struct damacy_metadata_cache_config* cache,
                            struct damacy_metadata** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!reader || !cache || !cache->array_entries || !cache->shard_entries)
    return DAMACY_INVAL;
  struct damacy_metadata* metadata = calloc(1, sizeof(*metadata));
  if (!metadata)
    return DAMACY_OOM;
  metadata->reader = *reader;
  metadata->cache = *cache;
  *out = metadata;
  return DAMACY_OK;
}

void
damacy_metadata_destroy(struct damacy_metadata* metadata)
{
  free(metadata);
}

void
damacy_planner_destroy(struct damacy_planner* planner)
{
  if (!planner)
    return;
  if (planner->active) {
    log_error("planner is still in use");
    return;
  }
  planner->ops->destroy(planner);
}

void
damacy_executor_destroy(struct damacy_executor* executor)
{
  if (!executor)
    return;
  if (executor->active) {
    log_error("executor is still in use");
    return;
  }
  executor->ops->destroy(executor);
}

void
buffer_retain(struct damacy_buffer* buffer)
{
  atomic_fetch_add_explicit(&buffer->references, 1, memory_order_relaxed);
}

void
buffer_release(struct damacy_buffer* buffer)
{
  if (buffer && atomic_fetch_sub_explicit(
                  &buffer->references, 1, memory_order_acq_rel) == 1)
    buffer->destroy(buffer);
}

int
buffer_available(const struct damacy_buffer* buffer)
{
  return buffer &&
         atomic_load_explicit(&buffer->references, memory_order_acquire) == 1;
}

struct damacy_batch*
batch_create(struct damacy_buffer* buffer,
             const struct damacy_batch_spec* output,
             uint64_t batch_id)
{
  struct damacy_batch* batch = calloc(1, sizeof(*batch));
  if (!batch)
    return NULL;
  atomic_init(&batch->references, 1);
  batch->buffer = buffer;
  buffer_retain(buffer);
  batch->info.rank = output->sample_rank + 1;
  batch->info.dtype = output->dtype;
  batch->info.batch_id = batch_id;
  batch->info.shape[0] = output->samples_per_batch;
  memcpy(batch->info.shape + 1,
         output->sample_shape,
         output->sample_rank * sizeof(int64_t));
  return batch;
}

void
damacy_batch_retain(struct damacy_batch* batch)
{
  if (batch)
    atomic_fetch_add_explicit(&batch->references, 1, memory_order_relaxed);
}

void
damacy_batch_release(struct damacy_batch* batch)
{
  if (batch && atomic_fetch_sub_explicit(
                 &batch->references, 1, memory_order_acq_rel) == 1) {
    buffer_release(batch->buffer);
    free(batch);
  }
}

void
damacy_batch_info(const struct damacy_batch* batch,
                  struct damacy_batch_info* out)
{
  if (!out)
    return;
  *out = batch ? batch->info : (struct damacy_batch_info){ 0 };
  if (batch) {
    out->data = batch->buffer->data;
    out->device_ptr = batch->buffer->data;
    out->ready_stream = batch->buffer->ready_stream;
    out->device_type = batch->buffer->device_type;
    out->device_id = batch->buffer->device_id;
  }
}
