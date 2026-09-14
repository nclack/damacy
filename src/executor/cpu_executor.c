#define ZSTD_STATIC_LINKING_ONLY
#include <blosc.h>
#include <zstd.h>

#include "pipeline/components.h"

#include "damacy_config.h"
#include "damacy_stats.h"
#include "threadpool/threadpool.h"

#include <stdlib.h>
#include <string.h>

enum cpu_slot_state
{
  CPU_FREE,
  CPU_RENDERING,
  CPU_READY,
  CPU_HELD
};

struct cpu_slot
{
  enum cpu_slot_state state;
  struct damacy_buffer* buffer;
  struct prepared_plan* plan;
  uint64_t batch_id;
  uint32_t dispatched;
  uint32_t remaining;
};

struct cpu_worker
{
  void* decoded;
  void* workspace;
  ZSTD_DCtx* zstd;
};

struct cpu_wave
{
  struct store_event event;
  struct store_read* reads;
  void* input;
  enum damacy_status* results;
  float* decode_ms;
  float* assemble_ms;
  uint64_t* output_bytes;
  struct platform_clock clock;
  uint32_t first_chunk;
  uint32_t count;
  uint32_t slot;
  uint64_t input_bytes;
  int active;
};

struct cpu_executor
{
  struct damacy_executor base;
  struct damacy_reader* reader;
  struct damacy_cpu_config config;
  struct damacy_batch_spec output;
  int64_t shape[DAMACY_MAX_RANK + 1];
  int64_t strides[DAMACY_MAX_RANK + 1];
  struct damacy_stats* stats;
  struct threadpool* pool;
  struct cpu_worker* workers;
  struct cpu_slot slots[2];
  struct cpu_wave waves[2];
  struct cpu_wave* decoding;
  uint64_t committed;
};

static float
half_to_float(uint16_t half)
{
  uint32_t sign = (uint32_t)(half & 0x8000) << 16;
  uint32_t exponent = (half >> 10) & 31;
  uint32_t mantissa = half & 1023;
  uint32_t bits;
  if (!exponent) {
    if (!mantissa)
      bits = sign;
    else {
      int shift = 0;
      while (!(mantissa & 1024)) {
        mantissa <<= 1;
        ++shift;
      }
      bits = sign | (uint32_t)(113 - shift) << 23 | (mantissa & 1023) << 13;
    }
  } else {
    bits =
      sign | (exponent == 31 ? 255 : exponent + 112) << 23 | mantissa << 13;
  }
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint16_t
float_to_bfloat(float value)
{
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffff) > 0x7f800000)
    return 0x7fff;
  return (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

static float
source_value(const void* source, enum dtype dtype, uint64_t index)
{
  const unsigned char* bytes = source;
  switch (dtype) {
    case dtype_u8:
      return bytes[index];
    case dtype_u16: {
      uint16_t value;
      memcpy(&value, bytes + index * sizeof(value), sizeof(value));
      return value;
    }
    case dtype_i16: {
      int16_t value;
      memcpy(&value, bytes + index * sizeof(value), sizeof(value));
      return value;
    }
    case dtype_u32: {
      uint32_t value;
      memcpy(&value, bytes + index * sizeof(value), sizeof(value));
      return (float)value;
    }
    case dtype_i32: {
      int32_t value;
      memcpy(&value, bytes + index * sizeof(value), sizeof(value));
      return (float)value;
    }
    case dtype_f16: {
      uint16_t value;
      memcpy(&value, bytes + index * sizeof(value), sizeof(value));
      return half_to_float(value);
    }
    case dtype_f32: {
      float value;
      memcpy(&value, bytes + index * sizeof(value), sizeof(value));
      return value;
    }
    default:
      return 0;
  }
}

static uint64_t
gather_chunk(struct cpu_executor* self,
             struct cpu_slot* slot,
             const struct plan_chunk* chunk,
             const struct plan_region* region,
             const struct zarr_metadata* meta,
             const void* decoded,
             float fill)
{
  struct selection_span spans[DAMACY_MAX_RANK];
  uint64_t origin[DAMACY_MAX_RANK];
  uint64_t position[DAMACY_MAX_RANK] = { 0 };
  uint64_t elements = 1;
  for (uint8_t d = 0; d < meta->rank; ++d) {
    origin[d] = chunk->coordinate[d] * meta->inner_chunk_shape[d];
    spans[d] = query_chunk_span(&region->axes[d],
                                region->source.dims[d],
                                origin[d],
                                meta->inner_chunk_shape[d]);
    elements *= spans[d].count;
  }
  if (!elements)
    return 0;
  uint8_t last = meta->rank - 1;
  uint64_t width = region->axes[last].count ? 1 : spans[last].count;
  for (;;) {
    uint64_t source = 0;
    uint64_t destination = (uint64_t)region->sample * self->strides[0];
    for (uint8_t d = 0; d < meta->rank; ++d) {
      const struct query_axis* axis = &region->axes[d];
      uint64_t p = spans[d].begin + position[d];
      uint64_t coordinate = axis->count ? (uint64_t)axis->indices[p].source : p;
      uint64_t output = axis->count
                          ? axis->indices[p].output
                          : coordinate - (uint64_t)region->source.dims[d].beg;
      source = source * meta->inner_chunk_shape[d] + coordinate - origin[d];
      destination += output * (uint64_t)self->strides[d + 1];
    }
    if (!chunk->missing && meta->dtype == dtype_f32 &&
        self->output.dtype == DAMACY_F32)
      memcpy((float*)slot->buffer->data + destination,
             (const float*)decoded + source,
             width * sizeof(float));
    else {
      for (uint64_t j = 0; j < width; ++j) {
        float value = chunk->missing
                        ? fill
                        : source_value(decoded, meta->dtype, source + j);
        if (self->output.dtype == DAMACY_F32)
          ((float*)slot->buffer->data)[destination + j] = value;
        else
          ((uint16_t*)slot->buffer->data)[destination + j] =
            float_to_bfloat(value);
      }
    }
    int d = region->axes[last].count ? last : (int)last - 1;
    for (; d >= 0; --d) {
      if (++position[d] < spans[d].count)
        break;
      position[d] = 0;
    }
    if (d < 0)
      break;
  }
  return elements * damacy_dtype_bpe(self->output.dtype);
}

static uint64_t
assemble_chunk(struct cpu_executor* self,
               struct cpu_slot* slot,
               const struct plan_chunk* chunk,
               const void* decoded)
{
  const struct prepared_plan* plan = slot->plan;
  const struct zarr_metadata* meta = &plan->arrays[chunk->array].metadata;
  float fill = source_value(meta->fill_value, meta->dtype, 0);
  uint64_t output_bytes = 0;
  for (uint32_t u = chunk->first_use; u != UINT32_MAX; u = plan->uses[u].next) {
    const struct plan_region* region = &plan->regions[plan->uses[u].region];
    if (region->operation == PLAN_GATHER) {
      output_bytes +=
        gather_chunk(self, slot, chunk, region, meta, decoded, fill);
      continue;
    }
    uint64_t lo[DAMACY_MAX_RANK], hi[DAMACY_MAX_RANK];
    uint64_t origin[DAMACY_MAX_RANK], coordinate[DAMACY_MAX_RANK];
    uint64_t region_bytes = damacy_dtype_bpe(self->output.dtype);
    for (uint8_t d = 0; d < meta->rank; ++d) {
      origin[d] = chunk->coordinate[d] * meta->inner_chunk_shape[d];
      uint64_t begin = (uint64_t)region->source.dims[d].beg;
      uint64_t end = (uint64_t)region->source.dims[d].end;
      lo[d] = origin[d] > begin ? origin[d] : begin;
      uint64_t chunk_end = origin[d] + meta->inner_chunk_shape[d];
      hi[d] = chunk_end < end ? chunk_end : end;
      coordinate[d] = lo[d];
      region_bytes *= hi[d] - lo[d];
    }
    output_bytes += region_bytes;
    uint8_t last = meta->rank - 1;
    uint64_t width = hi[last] - lo[last];
    for (;;) {
      uint64_t source = 0;
      uint64_t destination = (uint64_t)region->sample * self->strides[0];
      for (uint8_t d = 0; d < meta->rank; ++d) {
        source =
          source * meta->inner_chunk_shape[d] + coordinate[d] - origin[d];
        destination += (coordinate[d] - (uint64_t)region->source.dims[d].beg) *
                       (uint64_t)self->strides[d + 1];
      }
      if (!chunk->missing && meta->dtype == dtype_f32 &&
          self->output.dtype == DAMACY_F32)
        memcpy((float*)slot->buffer->data + destination,
               (const char*)decoded + source * sizeof(float),
               width * sizeof(float));
      else if (self->output.dtype == DAMACY_F32) {
        float* dst = (float*)slot->buffer->data + destination;
        for (uint64_t j = 0; j < width; ++j)
          dst[j] = chunk->missing
                     ? fill
                     : source_value(decoded, meta->dtype, source + j);
      } else {
        uint16_t* dst = (uint16_t*)slot->buffer->data + destination;
        for (uint64_t j = 0; j < width; ++j)
          dst[j] = float_to_bfloat(
            chunk->missing ? fill
                           : source_value(decoded, meta->dtype, source + j));
      }
      int finished = 1;
      for (int d = (int)last - 1; d >= 0; --d) {
        if (++coordinate[d] < hi[d]) {
          finished = 0;
          break;
        }
        coordinate[d] = lo[d];
      }
      if (finished)
        break;
    }
  }
  return output_bytes;
}

static enum damacy_status
decode_chunk(struct cpu_worker* worker,
             const struct plan_chunk* chunk,
             const struct zarr_metadata* metadata,
             const void* input,
             const void** decoded)
{
  *decoded = worker->decoded;
  if (chunk->missing)
    return DAMACY_OK;
  switch (metadata->inner_codec.id) {
    case CODEC_NONE:
      if (chunk->encoded_bytes != chunk->decoded_bytes)
        return DAMACY_DECODE;
      *decoded = input;
      return DAMACY_OK;
    case CODEC_ZSTD: {
      size_t size = ZSTD_decompressDCtx(worker->zstd,
                                        worker->decoded,
                                        chunk->decoded_bytes,
                                        input,
                                        chunk->encoded_bytes);
      return !ZSTD_isError(size) && size == chunk->decoded_bytes
               ? DAMACY_OK
               : DAMACY_DECODE;
    }
    case CODEC_BLOSC_ZSTD: {
      size_t size = 0;
      if (blosc_cbuffer_validate(input, chunk->encoded_bytes, &size) ||
          size != chunk->decoded_bytes ||
          chunk->encoded_bytes < BLOSC_MIN_HEADER_LENGTH)
        return DAMACY_DECODE;
      size_t nbytes, cbytes, blocksize;
      blosc_cbuffer_sizes(input, &nbytes, &cbytes, &blocksize);
      size_t typesize;
      int flags;
      blosc_cbuffer_metainfo(input, &typesize, &flags);
      const char* compressor = blosc_cbuffer_complib(input);
      if (!blocksize || blocksize > size ||
          typesize != dtype_bpe(metadata->dtype) ||
          cbytes != chunk->encoded_bytes || size > BLOSC_MAX_BUFFERSIZE ||
          (!(flags & BLOSC_MEMCPYED) &&
           (!compressor || strcmp(compressor, BLOSC_ZSTD_LIBNAME))))
        return DAMACY_DECODE;
      int result = blosc_decompress_ctx(input, worker->decoded, size, 1);
      return result > 0 && (size_t)result == size ? DAMACY_OK : DAMACY_DECODE;
    }
    default:
      return DAMACY_DECODE;
  }
}

static void
decode_one(size_t index, int tid, void* arg)
{
  struct cpu_executor* self = arg;
  struct cpu_wave* wave = self->decoding;
  struct cpu_slot* slot = &self->slots[wave->slot];
  const struct plan_chunk* chunk =
    &slot->plan->chunks[wave->first_chunk + index];
  const struct zarr_metadata* meta = &slot->plan->arrays[chunk->array].metadata;
  const void* input =
    (const char*)wave->input + index * self->config.max_encoded_chunk_bytes;
  const void* decoded;
  struct platform_clock clock = { 0 };
  platform_toc(&clock);
  wave->results[index] =
    decode_chunk(&self->workers[tid], chunk, meta, input, &decoded);
  wave->decode_ms[index] = platform_toc(&clock) * 1000;
  wave->assemble_ms[index] = 0;
  wave->output_bytes[index] = 0;
  if (wave->results[index] == DAMACY_OK) {
    wave->output_bytes[index] = assemble_chunk(self, slot, chunk, decoded);
    wave->assemble_ms[index] = platform_toc(&clock) * 1000;
  }
}

static void
cpu_buffer_destroy(struct damacy_buffer* buffer)
{
  free(buffer->data);
  free(buffer);
}

static void
cpu_stop(struct damacy_executor* base)
{
  struct cpu_executor* self = (void*)base;
  for (unsigned i = 0; i < 2; ++i) {
    struct cpu_wave* wave = &self->waves[i];
    if (wave->active && wave->event.impl)
      store_event_wait(self->reader->store, wave->event);
    free(wave->input);
    free(wave->reads);
    free(wave->results);
    free(wave->decode_ms);
    free(wave->assemble_ms);
    free(wave->output_bytes);
    *wave = (struct cpu_wave){ 0 };
    prepared_plan_destroy(self->slots[i].plan);
    buffer_release(self->slots[i].buffer);
    self->slots[i] = (struct cpu_slot){ 0 };
  }
  threadpool_free(self->pool);
  self->pool = NULL;
  if (self->workers) {
    for (uint32_t i = 0; i < self->config.decode_workers; ++i) {
      free(self->workers[i].decoded);
      free(self->workers[i].workspace);
    }
    free(self->workers);
    self->workers = NULL;
  }
  self->committed = 0;
}

static enum damacy_status
cpu_start(struct damacy_executor* base,
          const struct damacy_batch_spec* output,
          struct damacy_stats* stats)
{
  struct cpu_executor* self = (void*)base;
  self->output = *output;
  self->stats = stats;
  uint64_t bytes;
  enum damacy_status status =
    batch_spec_layout(output, self->shape, self->strides, &bytes);
  if (status != DAMACY_OK)
    return status;
  uint64_t workers = self->config.decode_workers;
  uint64_t workspace = ZSTD_estimateDCtxSize();
  uint64_t codec_reserve =
    workspace + 3ull * self->config.max_decoded_chunk_bytes + (256u << 10);
  uint64_t per_worker =
    2ull * self->config.max_encoded_chunk_bytes +
    self->config.max_decoded_chunk_bytes + codec_reserve +
    sizeof(struct cpu_worker) +
    2 * (sizeof(struct store_read) + sizeof(enum damacy_status) +
         2 * sizeof(float) + sizeof(uint64_t));
  uint64_t fixed = sizeof(*self) + 2 * sizeof(struct damacy_buffer);
  if (per_worker > self->config.max_memory_bytes / workers ||
      bytes > self->config.max_memory_bytes / 2)
    return DAMACY_BUDGET;
  uint64_t need = fixed + per_worker * workers + 2 * bytes;
  if (need < bytes || need > self->config.max_memory_bytes)
    return DAMACY_BUDGET;
  self->committed = need;
  for (unsigned i = 0; i < 2; ++i) {
    struct damacy_buffer* buffer = calloc(1, sizeof(*buffer));
    if (!buffer)
      goto Fail;
    atomic_init(&buffer->references, 1);
    buffer->destroy = cpu_buffer_destroy;
    buffer->device_type = DAMACY_DEVICE_CPU;
    buffer->nbytes = bytes;
    self->slots[i].buffer = buffer;
    buffer->data = malloc((size_t)bytes);
    if (!buffer->data)
      goto Fail;
    struct cpu_wave* wave = &self->waves[i];
    wave->input =
      malloc((size_t)workers * self->config.max_encoded_chunk_bytes);
    wave->reads = calloc((size_t)workers, sizeof(*wave->reads));
    wave->results = calloc((size_t)workers, sizeof(*wave->results));
    wave->decode_ms = calloc((size_t)workers, sizeof(*wave->decode_ms));
    wave->assemble_ms = calloc((size_t)workers, sizeof(*wave->assemble_ms));
    wave->output_bytes = calloc((size_t)workers, sizeof(*wave->output_bytes));
    if (!wave->input || !wave->reads || !wave->results || !wave->decode_ms ||
        !wave->assemble_ms || !wave->output_bytes)
      goto Fail;
  }
  self->workers = calloc((size_t)workers, sizeof(*self->workers));
  if (!self->workers)
    goto Fail;
  for (uint32_t i = 0; i < workers; ++i) {
    struct cpu_worker* worker = &self->workers[i];
    worker->decoded = malloc(self->config.max_decoded_chunk_bytes);
    worker->workspace = malloc((size_t)workspace);
    if (!worker->decoded || !worker->workspace)
      goto Fail;
    worker->zstd = ZSTD_initStaticDCtx(worker->workspace, (size_t)workspace);
    if (!worker->zstd)
      goto Fail;
  }
  self->pool = threadpool_new((int)workers - 1);
  if (!self->pool)
    goto Fail;
  return DAMACY_OK;
Fail:
  cpu_stop(base);
  return DAMACY_OOM;
}

static enum damacy_status
cpu_submit(struct damacy_executor* base,
           struct prepared_plan* plan,
           uint64_t batch_id)
{
  struct cpu_executor* self = (void*)base;
  int slot = -1;
  for (unsigned i = 0; i < 2; ++i)
    if (self->slots[i].state == CPU_FREE) {
      slot = (int)i;
      break;
    }
  if (slot < 0)
    return DAMACY_AGAIN;
  for (uint32_t i = 0; i < plan->n_chunks; ++i) {
    const struct plan_chunk* chunk = &plan->chunks[i];
    if (chunk->encoded_bytes > self->config.max_encoded_chunk_bytes ||
        chunk->decoded_bytes > self->config.max_decoded_chunk_bytes)
      return DAMACY_BUDGET;
    enum compression_codec codec =
      plan->arrays[chunk->array].metadata.inner_codec.id;
    if (!chunk->missing && codec != CODEC_NONE && codec != CODEC_ZSTD &&
        codec != CODEC_BLOSC_ZSTD)
      return DAMACY_DECODE;
  }
  struct cpu_slot* target = &self->slots[slot];
  target->plan = plan;
  target->batch_id = batch_id;
  target->dispatched = 0;
  target->remaining = plan->n_chunks;
  target->state = CPU_RENDERING;
  return DAMACY_OK;
}

static enum damacy_status
cpu_dispatch(struct cpu_executor* self, struct cpu_wave* wave, int* changed)
{
  int slot_index = -1;
  for (unsigned i = 0; i < 2; ++i) {
    const struct cpu_slot* slot = &self->slots[i];
    if (slot->state == CPU_RENDERING &&
        slot->dispatched < slot->plan->n_chunks &&
        (slot_index < 0 || slot->batch_id < self->slots[slot_index].batch_id))
      slot_index = (int)i;
  }
  if (slot_index < 0)
    return DAMACY_OK;
  struct cpu_slot* slot = &self->slots[slot_index];
  uint32_t count = slot->plan->n_chunks - slot->dispatched;
  if (count > self->config.decode_workers)
    count = self->config.decode_workers;
  if (count > self->reader->max_inflight_reads)
    count = self->reader->max_inflight_reads;
  uint32_t n_reads = 0;
  uint64_t bytes = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const struct plan_chunk* chunk = &slot->plan->chunks[slot->dispatched + i];
    if (chunk->missing)
      continue;
    wave->reads[n_reads++] =
      (struct store_read){ .key = chunk->path,
                           .offset = chunk->offset,
                           .len = chunk->encoded_bytes,
                           .dst =
                             (char*)wave->input +
                             (size_t)i * self->config.max_encoded_chunk_bytes };
    bytes += chunk->encoded_bytes;
  }
  platform_toc(&wave->clock);
  struct store_submit_result result =
    store_read_submit(self->reader->store, wave->reads, n_reads);
  if (result.status != DAMACY_OK)
    return result.status;
  wave->event = result.event;
  wave->active = 1;
  wave->slot = (uint32_t)slot_index;
  wave->first_chunk = slot->dispatched;
  wave->count = count;
  wave->input_bytes = bytes;
  slot->dispatched += count;
  self->stats->reads_issued += n_reads;
  self->stats->chunks_dispatched += count;
  ++self->stats->waves_emitted;
  *changed = 1;
  return DAMACY_OK;
}

static enum damacy_status
cpu_step(struct damacy_executor* base, int* changed)
{
  struct cpu_executor* self = (void*)base;
  for (unsigned i = 0; i < 2; ++i) {
    struct cpu_slot* slot = &self->slots[i];
    if (slot->state == CPU_HELD && buffer_available(slot->buffer)) {
      slot->state = CPU_FREE;
      *changed = 1;
    }
  }
  for (unsigned i = 0; i < 2; ++i) {
    if (!self->waves[i].active) {
      enum damacy_status status = cpu_dispatch(self, &self->waves[i], changed);
      if (status != DAMACY_OK && status != DAMACY_AGAIN)
        return status;
    }
  }
  for (unsigned i = 0; i < 2; ++i) {
    struct cpu_wave* wave = &self->waves[i];
    if (!wave->active)
      continue;
    struct store_event_poll poll =
      store_event_query(self->reader->store, wave->event);
    if (!poll.ready)
      continue;
    wave->event = (struct store_event){ 0 };
    if (poll.status != DAMACY_OK)
      return poll.status;
    metric_record(&self->stats->io,
                  platform_toc(&wave->clock) * 1000,
                  (double)wave->input_bytes,
                  (double)wave->input_bytes);
    self->decoding = wave;
    threadpool_for_n_dynamic(self->pool, wave->count, decode_one, self);
    struct cpu_slot* slot = &self->slots[wave->slot];
    for (uint32_t j = 0; j < wave->count; ++j) {
      if (wave->results[j] != DAMACY_OK)
        return wave->results[j];
      const struct plan_chunk* chunk =
        &slot->plan->chunks[wave->first_chunk + j];
      metric_record(&self->stats->decode,
                    wave->decode_ms[j],
                    chunk->encoded_bytes,
                    chunk->missing ? 0 : chunk->decoded_bytes);
      metric_record(&self->stats->assemble,
                    wave->assemble_ms[j],
                    chunk->decoded_bytes,
                    wave->output_bytes[j]);
    }
    slot->remaining -= wave->count;
    wave->active = 0;
    if (!slot->remaining) {
      slot->state = CPU_READY;
      prepared_plan_destroy(slot->plan);
      slot->plan = NULL;
    }
    *changed = 1;
  }
  return DAMACY_OK;
}

static enum damacy_status
cpu_take(struct damacy_executor* base, struct damacy_batch** out)
{
  struct cpu_executor* self = (void*)base;
  int oldest = -1;
  for (unsigned i = 0; i < 2; ++i) {
    struct cpu_slot* slot = &self->slots[i];
    if ((slot->state == CPU_RENDERING || slot->state == CPU_READY) &&
        (oldest < 0 || slot->batch_id < self->slots[oldest].batch_id))
      oldest = (int)i;
  }
  if (oldest < 0 || self->slots[oldest].state != CPU_READY)
    return DAMACY_AGAIN;
  struct cpu_slot* slot = &self->slots[oldest];
  *out = batch_create(slot->buffer, &self->output, slot->batch_id);
  if (!*out)
    return DAMACY_OOM;
  slot->state = CPU_HELD;
  return DAMACY_OK;
}

static enum damacy_status
cpu_wait_event(struct damacy_executor* base, void* event)
{
  (void)base;
  (void)event;
  return DAMACY_INVAL;
}

static int
cpu_busy(const struct damacy_executor* base)
{
  const struct cpu_executor* self = (const void*)base;
  return self->slots[0].state != CPU_FREE || self->slots[1].state != CPU_FREE;
}

static void
cpu_stats(struct damacy_executor* base, struct damacy_stats* out)
{
  out->host_bytes_committed = ((struct cpu_executor*)base)->committed;
}

static void
cpu_destroy(struct damacy_executor* base)
{
  cpu_stop(base);
  free(base);
}

static const struct damacy_executor_ops cpu_ops = { .start = cpu_start,
                                                    .submit = cpu_submit,
                                                    .step = cpu_step,
                                                    .take = cpu_take,
                                                    .wait_event =
                                                      cpu_wait_event,
                                                    .busy = cpu_busy,
                                                    .stats = cpu_stats,
                                                    .stop = cpu_stop,
                                                    .destroy = cpu_destroy };

enum damacy_status
damacy_cpu_executor_create(struct damacy_reader* reader,
                           const struct damacy_cpu_config* config,
                           struct damacy_executor** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  if (!reader || !config || !config->decode_workers ||
      config->decode_workers > DAMACY_MAX_IO_THREADS ||
      !config->max_encoded_chunk_bytes || !config->max_decoded_chunk_bytes ||
      !config->max_memory_bytes)
    return DAMACY_INVAL;
  struct cpu_executor* self = calloc(1, sizeof(*self));
  if (!self)
    return DAMACY_OOM;
  self->base.ops = &cpu_ops;
  self->base.device_type = DAMACY_DEVICE_CPU;
  self->reader = reader;
  self->config = *config;
  *out = &self->base;
  return DAMACY_OK;
}
