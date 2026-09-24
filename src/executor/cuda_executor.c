#include "executor/cuda_executor.h"
#include "pipeline/components.h"

#include "batch_pool/batch_pool.h"
#include "damacy_config.h"
#include "executor/dispatch.h"
#include "gpu_budget/gpu_budget.h"
#include "log/log.h"
#include "numa/numa_cuda.h"
#include "render_job/render_job.h"
#include "store/store_fs_gds.h"
#include "util/cuda_check.h"
#include "wave/wave_budget.h"
#include "wave/wave_input.h"
#include "wave/wave_pool.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_geometry.inc"

struct cuda_context
{
  _Atomic uint32_t references;
  CUcontext handle;
  CUstream retained_stream;
  int retained_device;
};

struct cuda_layout_entry
{
  char* uri;
  struct chunk_layout layout;
};

struct cuda_executor
{
  struct damacy_executor base;
  struct damacy_reader* reader;
  struct damacy_cuda_config config;
  struct damacy_batch_spec output;
  struct damacy_config cfg;
  struct cuda_context* context;
  struct numa_resolved numa;
  struct gpu_budget* budget;
  struct store* gds;
  struct damacy_stats* stats;
  struct damacy_batch_pool batches;
  struct render_job_pool jobs;
  struct wave_pool waves;
  struct dispatch_scratch scratch;
  struct prepared_plan* plans[DAMACY_N_BATCH_SLOTS];
  struct damacy_buffer* buffers[DAMACY_N_BATCH_SLOTS];
  struct cuda_layout_entry* layouts;
  uint32_t next_layout;
  uint64_t max_read_bytes;
  int worker_context;
};

static void
context_release(struct cuda_context* context)
{
  if (context && atomic_fetch_sub_explicit(
                   &context->references, 1, memory_order_acq_rel) == 1) {
    if (context->retained_stream &&
        cuCtxPushCurrent(context->handle) == CUDA_SUCCESS) {
      cuStreamDestroy(context->retained_stream);
      cuCtxPopCurrent(NULL);
    }
    if (context->retained_device >= 0)
      cuDevicePrimaryCtxRelease((CUdevice)context->retained_device);
    free(context);
  }
}

static void
cuda_buffer_destroy(struct damacy_buffer* buffer)
{
  struct cuda_context* context = buffer->context;
  if (cuCtxPushCurrent(context->handle) == CUDA_SUCCESS) {
    cuMemFree(CUDPTR(buffer->data));
    cuCtxPopCurrent(NULL);
  }
  context_release(context);
  free(buffer);
}

static enum damacy_status
cuda_buffer_wait_event(struct damacy_buffer* buffer, void* event)
{
  struct cuda_context* context = buffer->context;
  if (cuCtxPushCurrent(context->handle) != CUDA_SUCCESS)
    return DAMACY_CUDA;
  CUresult result = cuEventSynchronize(event);
  cuCtxPopCurrent(NULL);
  return result == CUDA_SUCCESS ? DAMACY_OK : DAMACY_CUDA;
}

static void
cuda_stop(struct damacy_executor* base)
{
  struct cuda_executor* self = (void*)base;
  int pushed =
    self->context && cuCtxPushCurrent(self->context->handle) == CUDA_SUCCESS;
  if (pushed && self->waves.stream_post) {
    self->context->retained_stream = self->waves.stream_post;
    cuStreamSynchronize(self->waves.stream_post);
    self->waves.stream_post = NULL;
  }
  wave_pool_destroy(&self->waves, !pushed);
  render_job_pool_destroy(&self->jobs, !pushed);
  for (unsigned i = 0; i < DAMACY_N_BATCH_SLOTS; ++i) {
    prepared_plan_destroy(self->plans[i]);
    self->plans[i] = NULL;
    if (self->buffers[i]) {
      self->batches.slots[i].dev_ptr = NULL;
      buffer_release(self->buffers[i]);
      self->buffers[i] = NULL;
    }
  }
  batch_pool_destroy(&self->batches, !pushed);
  gpu_budget_destroy(self->budget);
  self->budget = NULL;
  store_destroy(self->gds);
  self->gds = NULL;
  dispatch_scratch_destroy(&self->scratch);
  if (self->layouts) {
    for (unsigned i = 0; i < self->config.chunk_layout_entries; ++i)
      free(self->layouts[i].uri);
    free(self->layouts);
    self->layouts = NULL;
  }
  if (pushed)
    cuCtxPopCurrent(NULL);
  context_release(self->context);
  self->context = NULL;
}

static enum damacy_status
cuda_start(struct damacy_executor* base,
           const struct damacy_batch_spec* output,
           struct damacy_stats* stats)
{
  struct cuda_executor* self = (void*)base;
  self->output = *output;
  self->stats = stats;
  self->cfg = (struct damacy_config){
    .dtype = output->dtype,
    .sample_rank = output->sample_rank,
    .samples_per_batch = output->samples_per_batch,
    .device = self->config.device,
    .tuning = { .max_gpu_memory_bytes = self->config.max_gpu_memory_bytes,
                .max_chunk_uncompressed_bytes = self->config.max_chunk_bytes,
                .max_read_op_bytes = self->config.max_read_bytes,
                .host_buffer_waves = self->config.host_buffer_waves,
                .max_chunks_per_wave = self->config.max_chunks_per_wave,
                .max_substreams_per_chunk =
                  self->config.max_substreams_per_chunk,
                .numa_strategy = self->config.numa_strategy,
                .numa_node = self->config.numa_node,
                .enable_gds = self->config.enable_gds },
    .debug = { .bypass_decode = self->config.bypass_decode }
  };
  if (self->cfg.tuning.max_chunks_per_wave > self->reader->max_inflight_reads)
    self->cfg.tuning.max_chunks_per_wave = self->reader->max_inflight_reads;
  memcpy(self->cfg.sample_shape,
         output->sample_shape,
         sizeof(self->cfg.sample_shape));
  enum damacy_status status = DAMACY_CUDA;
  int pushed = 0;
  if (cuInit(0) != CUDA_SUCCESS)
    return status;
  CUcontext caller = NULL;
  if (cuCtxGetCurrent(&caller) != CUDA_SUCCESS)
    return status;
  self->context = calloc(1, sizeof(*self->context));
  if (!self->context)
    return DAMACY_OOM;
  atomic_init(&self->context->references, 1);
  self->context->retained_device = -1;
  CUdevice device;
  if (self->config.device >= 0) {
    if (caller) {
      if (cuCtxGetDevice(&device) != CUDA_SUCCESS)
        goto Fail;
      if (device != self->config.device) {
        status = DAMACY_INVAL;
        goto Fail;
      }
    }
    if (cuDeviceGet(&device, self->config.device) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&self->context->handle, device) !=
          CUDA_SUCCESS)
      goto Fail;
    self->context->retained_device = self->config.device;
  } else {
    if (!caller) {
      status = DAMACY_INVAL;
      goto Fail;
    }
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS)
      goto Fail;
    self->context->handle = caller;
  }
  self->base.device_id = device;
  if (cuCtxPushCurrent(self->context->handle) != CUDA_SUCCESS)
    goto Fail;
  pushed = 1;
  numa_init(
    self->config.numa_strategy, self->config.numa_node, device, &self->numa);
  status = batch_pool_compute_layout(&self->batches,
                                     output->sample_shape,
                                     output->sample_rank,
                                     output->samples_per_batch,
                                     damacy_dtype_bpe(output->dtype));
  if (status != DAMACY_OK)
    goto Fail;
  if (self->batches.n_bytes >= self->config.max_gpu_memory_bytes / 2) {
    status = DAMACY_BUDGET;
    goto Fail;
  }
  uint64_t resolver_budget =
    self->config.max_gpu_memory_bytes - 2 * self->batches.n_bytes;
  struct resolved_wave_geometry geometry;
  status = resolve_wave_geometry(
    &self->cfg, resolver_budget, self->config.max_chunk_bytes, &geometry, NULL);
  if (status != DAMACY_OK)
    goto Fail;
  self->max_read_bytes = self->config.max_read_bytes;
  if (self->max_read_bytes > geometry.sizing.input_staging_per_wave)
    self->max_read_bytes = geometry.sizing.input_staging_per_wave;
  status = DAMACY_OOM;
  self->budget = gpu_budget_new(self->config.max_gpu_memory_bytes);
  if (!self->budget)
    goto Fail;
  gpu_budget_commit(self->budget, geometry.predicted.total);
  if (geometry.want_gds) {
    self->gds =
      store_fs_gds_create(&(struct store_fs_gds_config){ .root = "" });
    if (!self->gds) {
      status = DAMACY_INVAL;
      goto Fail;
    }
  }
  struct platform_cpu_mask saved;
  numa_scope_enter(&self->numa, &saved);
  int failed = 0;
  for (unsigned i = 0; i < DAMACY_N_BATCH_SLOTS && !failed; ++i)
    failed = render_job_init(&self->jobs.jobs[i], output->samples_per_batch);
  if (!failed)
    failed = wave_pool_init(&self->waves,
                            &self->batches,
                            &self->jobs,
                            geometry.want_gds ? self->gds : self->reader->store,
                            stats,
                            output->dtype,
                            geometry.host_buffer_waves,
                            geometry.max_chunks_per_wave,
                            geometry.max_substreams_per_chunk,
                            geometry.sizing.input_staging_per_wave,
                            geometry.sizing.dev_decompressed_per_wave,
                            self->config.max_chunk_bytes,
                            geometry.input,
                            self->config.bypass_decode,
                            self->budget);
  numa_scope_exit(&saved);
  if (failed)
    goto Fail;
  self->layouts =
    calloc(self->config.chunk_layout_entries, sizeof(*self->layouts));
  if (!self->layouts)
    goto Fail;
  cuCtxPopCurrent(NULL);
  return DAMACY_OK;
Fail:
  if (pushed)
    cuCtxPopCurrent(NULL);
  cuda_stop(base);
  return status;
}

static enum damacy_status
allocate_outputs(struct cuda_executor* self)
{
  if (self->batches.allocated)
    return DAMACY_OK;
  uint64_t bytes = 2 * self->batches.n_bytes;
  enum damacy_status status =
    gpu_budget_try_commit(self->budget, bytes, "batch-output pool");
  if (status != DAMACY_OK)
    return status;
  status = batch_pool_alloc_dev(&self->batches);
  if (status != DAMACY_OK) {
    gpu_budget_release(self->budget, bytes);
    return status;
  }
  for (unsigned i = 0; i < DAMACY_N_BATCH_SLOTS; ++i) {
    struct damacy_buffer* buffer = calloc(1, sizeof(*buffer));
    if (!buffer)
      return DAMACY_OOM;
    atomic_init(&buffer->references, 1);
    buffer->data = self->batches.slots[i].dev_ptr;
    buffer->nbytes = self->batches.n_bytes;
    buffer->ready_stream = self->waves.stream_post;
    buffer->device_type = DAMACY_DEVICE_CUDA;
    buffer->device_id = self->base.device_id;
    buffer->destroy = cuda_buffer_destroy;
    buffer->wait_event = cuda_buffer_wait_event;
    buffer->context = self->context;
    atomic_fetch_add_explicit(
      &self->context->references, 1, memory_order_relaxed);
    self->buffers[i] = buffer;
  }
  return DAMACY_OK;
}

static struct cuda_layout_entry*
find_layout(struct cuda_executor* self, const char* uri)
{
  for (unsigned i = 0; i < self->config.chunk_layout_entries; ++i)
    if (self->layouts[i].uri && !strcmp(uri, self->layouts[i].uri))
      return &self->layouts[i];
  return NULL;
}

static enum damacy_status
prepare_layouts(struct cuda_executor* self,
                const struct prepared_plan* plan,
                struct dispatch_output* output)
{
  for (uint32_t a = 0; a < plan->n_arrays; ++a) {
    const struct plan_array* array = &plan->arrays[a];
    enum compression_codec codec = array->metadata.inner_codec.id;
    if (codec == CODEC_NONE || codec == CODEC_ZSTD)
      continue;
    if (codec != CODEC_BLOSC_ZSTD)
      return DAMACY_DECODE;
    const struct plan_chunk* chunk = NULL;
    for (uint32_t i = 0; i < plan->n_chunks; ++i)
      if (plan->chunks[i].array == a && !plan->chunks[i].missing) {
        chunk = &plan->chunks[i];
        break;
      }
    if (!chunk)
      continue;
    struct cuda_layout_entry* entry = find_layout(self, array->uri);
    if (!entry) {
      ++self->stats->chunk_layout.misses;
      if (chunk->encoded_bytes < 16)
        return DAMACY_DECODE;
      unsigned char header[16];
      struct store_read read = { .key = chunk->path,
                                 .offset = chunk->offset,
                                 .dst = header,
                                 .len = sizeof(header) };
      struct store_submit_result result =
        store_read_submit(self->reader->store, &read, 1);
      if (result.status != DAMACY_OK)
        return result.status;
      enum damacy_status status =
        store_event_wait(self->reader->store, result.event);
      if (status != DAMACY_OK)
        return status;
      struct chunk_layout layout;
      if (zarr_chunk_layout_parse_header(header,
                                         chunk->encoded_bytes,
                                         (uint8_t)codec,
                                         self->config.max_substreams_per_chunk,
                                         &layout) ||
          layout.nbytes != chunk->decoded_bytes ||
          layout.typesize != dtype_bpe(array->metadata.dtype))
        return DAMACY_DECODE;
      char* uri = strdup(array->uri);
      if (!uri)
        return DAMACY_OOM;
      entry =
        &self->layouts[self->next_layout++ % self->config.chunk_layout_entries];
      free(entry->uri);
      *entry = (struct cuda_layout_entry){ .uri = uri, .layout = layout };
    } else {
      ++self->stats->chunk_layout.hits;
    }
    for (uint32_t i = 0; i < plan->n_regions; ++i)
      if (plan->regions[i].array == a) {
        struct sample_plan* sample =
          &output->sample_plans[plan->regions[i].sample];
        sample->layout = entry->layout;
        sample->layout_probed = 1;
      }
  }
  return DAMACY_OK;
}

static enum damacy_status
cuda_submit(struct damacy_executor* base,
            struct prepared_plan* plan,
            uint64_t batch_id)
{
  struct cuda_executor* self = (void*)base;
  int slot_index = find_free_batch_slot(&self->batches);
  if (slot_index < 0 || find_render_job_with_work(&self->jobs) >= 0)
    return DAMACY_AGAIN;
  if (cuCtxPushCurrent(self->context->handle) != CUDA_SUCCESS)
    return DAMACY_CUDA;
  enum damacy_status status = allocate_outputs(self);
  if (status != DAMACY_OK)
    goto Done;
  struct render_job* job = &self->jobs.jobs[slot_index];
  struct dispatch_output output =
    render_job_dispatch_output(job, self->output.samples_per_batch);
  status = dispatch_plan_build(plan,
                               (uint16_t)slot_index,
                               platform_page_alignment(),
                               self->max_read_bytes,
                               self->cfg.tuning.max_chunks_per_wave,
                               &output,
                               &self->scratch);
  if (status != DAMACY_OK)
    goto Done;
  status = prepare_layouts(self, plan, &output);
  if (status != DAMACY_OK)
    goto Done;
  render_job_commit_plan(job, (uint16_t)slot_index, batch_id, &output);
  status = render_job_upload_sample_plans(job, self->waves.stream_input);
  if (status != DAMACY_OK)
    goto Done;
  struct damacy_batch_slot* slot = &self->batches.slots[slot_index];
  slot->batch_id = batch_id;
  slot->n_samples = self->output.samples_per_batch;
  slot->n_chunks = job->n_chunks;
  slot->chunks_remaining = (int32_t)job->n_chunks;
  slot->state = BATCH_RENDERING;
  self->plans[slot_index] = plan;
  self->stats->reads_issued += job->n_loads_issued;
Done:
  cuCtxPopCurrent(NULL);
  return status;
}

static enum damacy_status
cuda_enter_thread(struct damacy_executor* base)
{
  struct cuda_executor* self = (void*)base;
  if (cuCtxPushCurrent(self->context->handle) != CUDA_SUCCESS)
    return DAMACY_CUDA;
  self->worker_context = 1;
  numa_apply_thread_affinity(&self->numa, "cuda_executor");
  return DAMACY_OK;
}

static void
cuda_leave_thread(struct damacy_executor* base)
{
  struct cuda_executor* self = (void*)base;
  if (self->worker_context) {
    cuCtxPopCurrent(NULL);
    self->worker_context = 0;
  }
}

static enum damacy_status
cuda_step(struct damacy_executor* base, int* changed)
{
  struct cuda_executor* self = (void*)base;
  for (unsigned i = 0; i < DAMACY_N_BATCH_SLOTS; ++i) {
    struct damacy_batch_slot* slot = &self->batches.slots[i];
    if (slot->state == BATCH_HELD && buffer_available(self->buffers[i])) {
      batch_slot_reset_for_reuse(slot);
      render_job_reset(&self->jobs.jobs[i]);
      *changed = 1;
    }
  }
  enum damacy_status status = wave_pool_advance(&self->waves, changed);
  while (status == DAMACY_OK) {
    int index = find_render_job_with_work(&self->jobs);
    if (index < 0)
      break;
    struct wave_input_reservation reservation = { 0 };
    status = wave_input_reserve(&self->waves, (uint16_t)index, &reservation);
    if (status != DAMACY_OK || !wave_input_reservation_has_slot(&reservation))
      break;
    struct store_submit_result result =
      wave_input_submit(&self->waves, &reservation);
    status = wave_input_commit(&self->waves, &reservation, result, changed);
    if (!any_slot_free(&self->waves))
      break;
  }
  for (unsigned i = 0; i < DAMACY_N_BATCH_SLOTS; ++i)
    if (self->batches.slots[i].state == BATCH_READY && self->plans[i]) {
      prepared_plan_destroy(self->plans[i]);
      self->plans[i] = NULL;
    }
  return status;
}

static enum damacy_status
cuda_take(struct damacy_executor* base, struct damacy_batch** out)
{
  struct cuda_executor* self = (void*)base;
  int oldest = -1;
  for (unsigned i = 0; i < DAMACY_N_BATCH_SLOTS; ++i) {
    const struct damacy_batch_slot* slot = &self->batches.slots[i];
    if ((slot->state == BATCH_RENDERING || slot->state == BATCH_READY) &&
        (oldest < 0 || slot->batch_id < self->batches.slots[oldest].batch_id))
      oldest = (int)i;
  }
  if (oldest < 0 || self->batches.slots[oldest].state != BATCH_READY)
    return DAMACY_AGAIN;
  *out = batch_create(
    self->buffers[oldest], &self->output, self->batches.slots[oldest].batch_id);
  if (!*out)
    return DAMACY_OOM;
  self->batches.slots[oldest].state = BATCH_HELD;
  return DAMACY_OK;
}

static enum damacy_status
cuda_wait_event(struct damacy_executor* base, void* event)
{
  struct cuda_executor* self = (void*)base;
  if (cuCtxPushCurrent(self->context->handle) != CUDA_SUCCESS)
    return DAMACY_CUDA;
  CUresult result = cuStreamWaitEvent(self->waves.stream_post, event, 0);
  cuCtxPopCurrent(NULL);
  return result == CUDA_SUCCESS ? DAMACY_OK : DAMACY_CUDA;
}

static int
cuda_busy(const struct damacy_executor* base)
{
  const struct cuda_executor* self = (const void*)base;
  return any_batch_in_flight(&self->batches);
}

static void
cuda_stats(struct damacy_executor* base, struct damacy_stats* out)
{
  out->gpu_bytes_committed =
    gpu_budget_committed(((struct cuda_executor*)base)->budget);
}

static void
cuda_destroy(struct damacy_executor* base)
{
  cuda_stop(base);
  free(base);
}

static const struct damacy_executor_ops cuda_ops = {
  .enter_thread = cuda_enter_thread,
  .leave_thread = cuda_leave_thread,
  .start = cuda_start,
  .submit = cuda_submit,
  .step = cuda_step,
  .take = cuda_take,
  .wait_event = cuda_wait_event,
  .busy = cuda_busy,
  .stats = cuda_stats,
  .stop = cuda_stop,
  .destroy = cuda_destroy
};

uint64_t
cuda_executor_set_budget(struct damacy_executor* base, uint64_t value)
{
  return base->ops == &cuda_ops
           ? gpu_budget_set_committed_for_test(
               ((struct cuda_executor*)base)->budget, value)
           : 0;
}

enum damacy_status
damacy_cuda_executor_create(struct damacy_reader* reader,
                            const struct damacy_cuda_config* config,
                            struct damacy_executor** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!reader || !config || config->device < -1 ||
      !config->max_gpu_memory_bytes || !config->chunk_layout_entries ||
      !config->max_chunk_bytes || !config->max_read_bytes ||
      config->max_read_bytes > UINT32_MAX || !config->max_chunks_per_wave ||
      config->max_chunks_per_wave > DAMACY_HARD_MAX_CHUNKS_PER_WAVE ||
      !config->max_substreams_per_chunk ||
      config->max_substreams_per_chunk > DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK ||
      config->host_buffer_waves < DAMACY_N_WAVES ||
      config->host_buffer_waves > DAMACY_MAX_HOST_BUFFER_WAVES ||
      config->numa_strategy < DAMACY_NUMA_AUTO ||
      config->numa_strategy > DAMACY_NUMA_PIN_TO ||
      (config->numa_strategy == DAMACY_NUMA_PIN_TO && config->numa_node < 0) ||
      config->enable_gds < DAMACY_GDS_AUTO ||
      config->enable_gds > DAMACY_GDS_OFF)
    return DAMACY_INVAL;
  struct cuda_executor* self = calloc(1, sizeof(*self));
  if (!self)
    return DAMACY_OOM;
  self->base.ops = &cuda_ops;
  self->base.device_type = DAMACY_DEVICE_CUDA;
  self->reader = reader;
  self->config = *config;
  *out = &self->base;
  return DAMACY_OK;
}

void
cuda_resolve_numa(const struct damacy_config* config, struct numa_resolved* out)
{
  *out = (struct numa_resolved){ .node = -1 };
  if (config->tuning.numa_strategy == DAMACY_NUMA_DISABLED ||
      cuInit(0) != CUDA_SUCCESS)
    return;
  CUdevice device;
  if (config->device >= 0) {
    if (cuDeviceGet(&device, config->device) != CUDA_SUCCESS)
      return;
  } else if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
    return;
  }
  numa_init(
    config->tuning.numa_strategy, config->tuning.numa_node, device, out);
}
