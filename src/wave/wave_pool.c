#include "wave_pool.h"

#include "batch_pool/batch_pool.h"
#include "damacy_config.h"
#include "damacy_stats.h"
#include "decoder/blosc1_parse.h"
#include "decoder/decoder_zstd.h"
#include "decoder/status_reduce.h"
#include "fanout.h"
#include "gpu_budget/gpu_budget.h"
#include "log/log.h"
#include "nvtx/nvtx.h"
#include "planner/planner.h"
#include "render_job/render_job.h"
#include "store/store.h"
#include "util/cuda_check.h"
#include "util/prelude.h"
#include "wave_budget.h"
#include "zarr/zarr_metadata.h" // CODEC_*

#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum damacy_status
decoder_scratch_grow(struct decoder_zstd* decoder,
                     CUstream stream_decode,
                     uint32_t max_chunks_per_wave,
                     uint32_t max_substreams_per_wave,
                     uint64_t dev_per_wave,
                     uint64_t max_chunk_uncompressed_bytes,
                     struct gpu_budget* budget,
                     size_t need);

// Wave-index helper for NVTX range labels.
static inline ptrdiff_t
wave_index_of(const struct wave_pool* wp, const struct damacy_wave* wave)
{
  return wave - wp->waves;
}

static inline struct render_job*
wave_job(const struct wave_pool* wp, const struct damacy_wave* wave)
{
  return render_job_pool_get(wp->render_jobs, wave->render_job_idx);
}

int
wave_pool_init(struct wave_pool* wp,
               struct damacy_batch_pool* pool,
               struct render_job_pool* render_jobs,
               struct store* store,
               struct damacy_stats* stats,
               enum damacy_dtype dtype,
               uint8_t host_buffer_waves,
               uint32_t max_chunks_per_wave,
               uint32_t max_substreams_per_chunk,
               uint64_t input_staging_per_wave,
               uint64_t dev_decompressed_per_wave,
               uint64_t max_chunk_uncompressed_bytes,
               const struct input_transfer_ops* input,
               int bypass_decode,
               struct gpu_budget* budget)
{
  memset(wp, 0, sizeof(*wp));
  wp->pool = pool;
  wp->render_jobs = render_jobs;
  wp->store = store;
  wp->stats = stats;
  wp->dtype = dtype;
  wp->budget = budget;
  wp->n_slots = host_buffer_waves;
  wp->input = input;
  wp->bypass_decode = (uint8_t)(bypass_decode != 0);
  wp->max_chunks_per_wave = max_chunks_per_wave;
  wp->max_substreams_per_wave = DAMACY_MAX_SUBSTREAMS_PER_WAVE(
    max_chunks_per_wave, max_substreams_per_chunk);

  // NON_BLOCKING so we don't serialize against the legacy default stream.
  CU(Fail, cuStreamCreate(&wp->stream_input, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&wp->stream_decode, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&wp->stream_post, CU_STREAM_NON_BLOCKING));
  wp->input->bind_stream(store, wp->stream_input);
  for (size_t i = 0; i < countof(wp->decode_done_ring); ++i)
    CU(Fail, cuEventCreate(&wp->decode_done_ring[i], CU_EVENT_DEFAULT));
  damacy_nvtx_stream_name(wp->stream_input, "damacy:input");
  damacy_nvtx_stream_name(wp->stream_decode, "damacy:decode");
  damacy_nvtx_stream_name(wp->stream_post, "damacy:post");

  const uint64_t dev_per_wave = dev_decompressed_per_wave;
  wp->dev_per_wave = dev_per_wave;
  wp->max_chunk_uncompressed_bytes = max_chunk_uncompressed_bytes;

  size_t substreams = 0, zstd_per = 0, total_uncompressed = 0;
  decoder_initial_caps(max_chunks_per_wave,
                       dev_per_wave,
                       max_chunk_uncompressed_bytes,
                       &substreams,
                       &zstd_per,
                       &total_uncompressed);
  wp->zstd_decoder =
    decoder_zstd_create(substreams, zstd_per, total_uncompressed);
  CHECK(Fail, wp->zstd_decoder);

  const struct input_transfer_resources input_resources =
    input_transfer_resources(input, host_buffer_waves, input_staging_per_wave);
  for (uint8_t s = 0; s < host_buffer_waves; ++s)
    if (input_slot_init(&wp->slots[s],
                        max_chunks_per_wave,
                        input_resources.slot_host_bytes,
                        input_resources.slot_device_bytes) != 0)
      goto Fail;
  for (int w = 0; w < DAMACY_N_WAVES; ++w) {
    if (wave_init(&wp->waves[w],
                  max_chunks_per_wave,
                  wp->max_substreams_per_wave,
                  input_resources.wave_device_bytes,
                  dev_per_wave) != 0)
      goto Fail;
    wp->waves[w].bound_slot = -1;
  }
  return 0;
Fail:
  wave_pool_destroy(wp, 0);
  return 1;
}

void
wave_pool_destroy(struct wave_pool* wp, int cuda_skip)
{
  if (!wp)
    return;
  CUstream* const streams[] = { &wp->stream_post,
                                &wp->stream_decode,
                                &wp->stream_input };
  if (!cuda_skip) {
    // Sync streams (but don't destroy yet) so any pending GPU work
    // touching wave + shared-decoder buffers has retired before we
    // free those buffers.
    for (size_t i = 0; i < countof(streams); ++i)
      if (*streams[i])
        cuStreamSynchronize(*streams[i]);
  }
  for (int w = 0; w < DAMACY_N_WAVES; ++w)
    wave_destroy(&wp->waves[w], cuda_skip);
  for (uint8_t s = 0; s < wp->n_slots; ++s) {
    struct input_slot* slot = &wp->slots[s];
    // is_fill_wave never attaches a backend ref; relies on impl=NULL
    // guard inside store_event_discard.
    if (slot->state == SLOT_IO && wp->store)
      store_event_discard(wp->store, slot->io_event);
    input_slot_destroy(slot, cuda_skip);
  }
  decoder_zstd_destroy(wp->zstd_decoder, cuda_skip);
  wp->zstd_decoder = NULL;
  if (!cuda_skip) {
    for (size_t i = 0; i < countof(wp->decode_done_ring); ++i)
      if (wp->decode_done_ring[i]) {
        cuEventDestroy_v2(wp->decode_done_ring[i]);
        wp->decode_done_ring[i] = NULL;
      }
    for (size_t i = 0; i < countof(streams); ++i)
      if (*streams[i]) {
        cuStreamDestroy(*streams[i]);
        *streams[i] = NULL;
      }
  }
  // Borrowed pointers — owner frees them.
  wp->pool = NULL;
  wp->render_jobs = NULL;
  wp->store = NULL;
  wp->stats = NULL;
}

int
any_wave_in_flight(const struct wave_pool* wp)
{
  for (int w = 0; w < DAMACY_N_WAVES; ++w)
    if (wp->waves[w].state != WAVE_FREE)
      return 1;
  return 0;
}

int
any_slot_in_flight(const struct wave_pool* wp)
{
  return input_slot_any_in_flight(wp->slots, wp->n_slots);
}

int
any_slot_free(const struct wave_pool* wp)
{
  return input_slot_any_free(wp->slots, wp->n_slots);
}

static int
wave_chunks_eligible(const struct wave_pool* wp, const struct damacy_wave* wave)
{
  if (wp->bypass_decode)
    return 1;
  struct render_job* job = wave_job(wp, wave);
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    const struct chunk_plan* c =
      &job->chunk_plans[wave->batch_chunk_offset + i];
    if (c->is_fill)
      continue;
    if (c->codec_id != (uint8_t)CODEC_BLOSC_ZSTD)
      continue;
    const struct sample_plan* sp = &job->sample_plans[c->sample_idx_in_batch];
    if (!sp->layout_probed)
      return 0;
    if (!sp->layout.dont_split)
      return 0;
  }
  return 1;
}

static uint32_t
chunk_compressed_offset(const struct chunk_plan* c)
{
  return (uint32_t)(c->host_buf_offset + (uint64_t)c->offset_in_read);
}

static int
chunk_effectively_fill(const struct chunk_plan* c, uint8_t force_fill)
{
  return c->is_fill || c->codec_id == (uint8_t)CODEC_FILL || force_fill;
}

static void
set_parse_chunk(struct gpu_parse_chunk* out,
                const struct chunk_plan* c,
                uint8_t force_fill)
{
  out->compressed_offset = chunk_compressed_offset(c);
  out->compressed_nbytes = c->compressed_nbytes;
  out->decompressed_offset = c->dev_decompressed_offset;
  out->decompressed_nbytes = c->decompressed_nbytes;
  out->sample_idx_in_batch = c->sample_idx_in_batch;
  out->codec_id = c->codec_id;
  out->is_fill = c->is_fill | force_fill;
}

static void
clear_parse_shuffle(struct assemble_chunk* out)
{
  out->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_NONE;
  out->shuffle_typesize = 0;
  out->shuffle_blocksize = 0;
}

static void
append_memcpy_prefix(struct damacy_wave* wave,
                     const struct chunk_plan* c,
                     uint32_t comp_off,
                     uint32_t* count)
{
  uint8_t* d_comp_base = (uint8_t*)wave->dev_compressed;
  uint8_t* d_decomp_base = (uint8_t*)wave->dev_decompressed;
  struct gpu_memcpy_op* op = &wave->h_memcpy_ops[(*count)++];
  op->d_src = d_comp_base + comp_off;
  op->d_dst = d_decomp_base + c->dev_decompressed_offset;
  op->nbytes = c->decompressed_nbytes;
}

static void
append_zstd_prefix(struct damacy_wave* wave,
                   const struct chunk_plan* c,
                   uint32_t comp_off,
                   uint32_t* count)
{
  uint8_t* d_comp_base = (uint8_t*)wave->dev_compressed;
  uint8_t* d_decomp_base = (uint8_t*)wave->dev_decompressed;
  uint32_t idx = (*count)++;
  wave->h_zstd_fan.comp_ptrs[idx] = d_comp_base + comp_off;
  wave->h_zstd_fan.comp_sizes[idx] = c->compressed_nbytes;
  wave->h_zstd_fan.decomp_ptrs[idx] =
    d_decomp_base + c->dev_decompressed_offset;
  wave->h_zstd_fan.decomp_buf_sizes[idx] = c->decompressed_nbytes;
}

static void
append_blosc_parse_work(struct damacy_wave* wave,
                        uint32_t wave_chunk_idx,
                        const struct sample_plan* sp,
                        uint32_t* n_blosc_chunks,
                        uint32_t* n_blosc_blocks)
{
  wave->h_blosc_chunk_indices[(*n_blosc_chunks)++] = wave_chunk_idx;
  for (uint32_t b = 0; b < sp->layout.nblocks; ++b)
    wave->h_block_chunk_map[(*n_blosc_blocks)++] = (wave_chunk_idx << 16) | b;
}

static void
build_gpu_parse_chunks(const struct wave_pool* wp, struct damacy_wave* wave)
{
  struct render_job* job = wave_job(wp, wave);
  uint32_t n_host_memcpy = 0;
  uint32_t n_host_zstd = 0;
  uint32_t n_blosc_chunks = 0;
  uint32_t n_blosc_blocks = 0;

  uint8_t force_fill = wp->bypass_decode;
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    struct chunk_plan* c = &job->chunk_plans[wave->batch_chunk_offset + i];
    uint32_t comp_off = chunk_compressed_offset(c);
    set_parse_chunk(&wave->h_parse_chunks[i], c, force_fill);
    clear_parse_shuffle(&wave->h_assemble_chunks[i]);

    if (chunk_effectively_fill(c, force_fill))
      continue;

    if (c->codec_id == (uint8_t)CODEC_NONE) {
      append_memcpy_prefix(wave, c, comp_off, &n_host_memcpy);
      continue;
    }

    if (c->codec_id == (uint8_t)CODEC_ZSTD) {
      append_zstd_prefix(wave, c, comp_off, &n_host_zstd);
      continue;
    }

    const struct sample_plan* sp = &job->sample_plans[c->sample_idx_in_batch];
    append_blosc_parse_work(wave, i, sp, &n_blosc_chunks, &n_blosc_blocks);
  }

  wave->n_host_memcpy = n_host_memcpy;
  wave->n_host_zstd = n_host_zstd;
  wave->n_blosc_zstd_chunks = n_blosc_chunks;
  wave->n_blosc_zstd_blocks = n_blosc_blocks;
}

static void
record_io_metric(const struct wave_pool* wp, const struct damacy_wave* wave)
{
  metric_record(&wp->stats->io, wave->io_ms, wave->io_bytes, wave->io_bytes);
}

enum damacy_status
chunk_substreams_upper_bound(const struct chunk_plan* c,
                             const struct sample_plan* sp,
                             uint32_t* out)
{
  if (c->is_fill || c->codec_id == (uint8_t)CODEC_FILL ||
      c->codec_id == (uint8_t)CODEC_NONE) {
    *out = 0;
    return DAMACY_OK;
  }
  if (c->codec_id == (uint8_t)CODEC_BLOSC_ZSTD) {
    if (!sp->layout_probed)
      return DAMACY_INVAL;
    *out = sp->layout.nblocks;
    return DAMACY_OK;
  }
  *out = 1;
  return DAMACY_OK;
}

static enum damacy_status
count_wave_substreams(const struct wave_pool* wp,
                      const struct damacy_wave* wave,
                      size_t* out)
{
  *out = 0;
  if (wp->bypass_decode)
    return DAMACY_OK;

  const struct render_job* job = wave_job(wp, wave);
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    const struct chunk_plan* c =
      &job->chunk_plans[wave->batch_chunk_offset + i];
    const struct sample_plan* sp = &job->sample_plans[c->sample_idx_in_batch];
    uint32_t n = 0;
    enum damacy_status s = chunk_substreams_upper_bound(c, sp, &n);
    if (s != DAMACY_OK) {
      log_error("prepare_decode_caps: chunk %u failed substream sizing", i);
      return s;
    }
    *out += n;
  }
  return DAMACY_OK;
}

static enum damacy_status
prepare_decode_caps(struct wave_pool* wp, struct damacy_wave* wave)
{
  size_t need_substreams = 0;
  enum damacy_status s = count_wave_substreams(wp, wave, &need_substreams);
  if (s != DAMACY_OK)
    return s;

  enum damacy_status gs = fanout_grow(&wave->h_zstd_fan,
                                      &wave->zstd_fan,
                                      &wave->fanout_cap,
                                      need_substreams,
                                      wp->max_substreams_per_wave,
                                      wp->budget);
  if (gs != DAMACY_OK)
    return gs;
  return decoder_scratch_grow(wp->zstd_decoder,
                              wp->stream_decode,
                              wp->max_chunks_per_wave,
                              wp->max_substreams_per_wave,
                              wp->dev_per_wave,
                              wp->max_chunk_uncompressed_bytes,
                              wp->budget,
                              need_substreams);
}

static enum damacy_status
queue_parse_metadata(struct wave_pool* wp, struct damacy_wave* wave)
{
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_parse_chunks),
                       wave->h_parse_chunks,
                       (size_t)wave->n_chunks * sizeof(struct gpu_parse_chunk),
                       wp->stream_input));
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_assemble_chunks),
                       wave->h_assemble_chunks,
                       (size_t)wave->n_chunks * sizeof(struct assemble_chunk),
                       wp->stream_input));

  if (wave->n_blosc_zstd_chunks > 0)
    CU(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_blosc_chunk_indices),
                         wave->h_blosc_chunk_indices,
                         (size_t)wave->n_blosc_zstd_chunks * sizeof(uint32_t),
                         wp->stream_input));
  if (wave->n_blosc_zstd_blocks > 0)
    CU(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_block_chunk_map),
                         wave->h_block_chunk_map,
                         (size_t)wave->n_blosc_zstd_blocks * sizeof(uint32_t),
                         wp->stream_input));
  return DAMACY_OK;
CudaFail:
  return DAMACY_CUDA;
}

static enum damacy_status
queue_host_decode_prefixes(struct wave_pool* wp, struct damacy_wave* wave)
{
  if (wave->n_host_memcpy > 0)
    CU(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_memcpy_ops),
                         wave->h_memcpy_ops,
                         (size_t)wave->n_host_memcpy *
                           sizeof(struct gpu_memcpy_op),
                         wp->stream_input));
  if (wave->n_host_zstd > 0 && fanout_upload(wp->stream_input,
                                             &wave->zstd_fan,
                                             &wave->h_zstd_fan,
                                             wave->n_host_zstd) != DAMACY_OK)
    goto CudaFail;
  return DAMACY_OK;
CudaFail:
  return DAMACY_CUDA;
}

static enum damacy_status
queue_parse_state_init(struct wave_pool* wp, struct damacy_wave* wave)
{
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_n_zstd),
                       &wave->n_host_zstd,
                       sizeof(uint32_t),
                       wp->stream_input));
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_n_memcpy),
                       &wave->n_host_memcpy,
                       sizeof(uint32_t),
                       wp->stream_input));
  CU(CudaFail,
     cuMemsetD8Async(
       CUDPTR(wave->d_parse_err), 0, sizeof(uint32_t), wp->stream_input));
  // Bitset: one bit per wave-local chunk_idx. Kernel A atomic-ors a 1 in
  // for memcpyed chunks; Kernel B reads it and skips those chunks'
  // blocks.
  CU(CudaFail,
     cuMemsetD8Async(CUDPTR(wave->d_is_memcpyed),
                     0,
                     (size_t)((wave->n_chunks + 31u) / 32u) * sizeof(uint32_t),
                     wp->stream_input));
  CU(CudaFail,
     cuMemsetD8Async(CUDPTR(wave->d_blosc1_totals),
                     0,
                     sizeof(struct blosc1_totals),
                     wp->stream_input));
  return DAMACY_OK;
CudaFail:
  return DAMACY_CUDA;
}

static enum damacy_status
queue_parse_counter_readback(struct wave_pool* wp, struct damacy_wave* wave)
{
  CU(CudaFail,
     cuMemcpyDtoHAsync(wave->h_parse_counters,
                       CUDPTR(wave->d_n_zstd),
                       sizeof(uint32_t),
                       wp->stream_input));
  CU(CudaFail,
     cuMemcpyDtoHAsync(wave->h_parse_counters + 1,
                       CUDPTR(wave->d_n_memcpy),
                       sizeof(uint32_t),
                       wp->stream_input));
  CU(CudaFail,
     cuMemcpyDtoHAsync(wave->h_parse_counters + 2,
                       CUDPTR(wave->d_parse_err),
                       sizeof(uint32_t),
                       wp->stream_input));
  CU(CudaFail, cuEventRecord(wave->ev.input_parse_done, wp->stream_input));
  return DAMACY_OK;
CudaFail:
  return DAMACY_CUDA;
}

static enum damacy_status
gpu_parse(struct wave_pool* wp, struct damacy_wave* wave)
{
  damacy_nvtx_range_push("gpu_parse");
  enum damacy_status s = queue_parse_metadata(wp, wave);
  if (s != DAMACY_OK)
    goto Done;
  s = queue_host_decode_prefixes(wp, wave);
  if (s != DAMACY_OK)
    goto Done;
  s = queue_parse_state_init(wp, wave);
  if (s != DAMACY_OK)
    goto Done;

  struct render_job* job = wave_job(wp, wave);
  struct blosc1_parse_args pargs = {
    .d_compressed = (const uint8_t*)wave->dev_compressed,
    .d_decompressed = (uint8_t*)wave->dev_decompressed,
    .d_chunks = wave->d_parse_chunks,
    .d_sample_plans = (const struct sample_plan*)job->d_sample_plans,
    .n_sample_plans = job->n_sample_plans,
    .d_blosc_chunk_indices = wave->d_blosc_chunk_indices,
    .n_blosc_zstd_chunks = wave->n_blosc_zstd_chunks,
    .d_block_chunk_map = wave->d_block_chunk_map,
    .n_blosc_zstd_blocks = wave->n_blosc_zstd_blocks,
    .d_is_memcpyed = wave->d_is_memcpyed,
    .zstd = wave->zstd_fan,
    .d_memcpy_ops = wave->d_memcpy_ops,
    .d_assemble_chunks = wave->d_assemble_chunks,
    .d_n_zstd = wave->d_n_zstd,
    .d_n_memcpy = wave->d_n_memcpy,
    .d_parse_err = wave->d_parse_err,
  };
  if (blosc1_parse_launch(wp->stream_input, &pargs)) {
    s = DAMACY_CUDA;
    goto Done;
  }
  s = queue_parse_counter_readback(wp, wave);
Done:
  damacy_nvtx_range_pop(); // gpu_parse
  return s;
}

static void
wave_mark_input(struct damacy_wave* wave)
{
  wave->state = WAVE_INPUT;
}

static enum damacy_status
prepare_input_parse(struct wave_pool* wp, struct damacy_wave* wave)
{
  if (!wave_chunks_eligible(wp, wave)) {
    // Probe gap (e.g. all-fill preceding waves never seeded the layout
    // cache) or unsupported dont_split=0 — not garbage in compressed
    // bytes, so DAMACY_INVAL rather than DAMACY_DECODE.
    log_error("blosc1: wave has a BLOSC_ZSTD chunk with unprobed layout "
              "or dont_split=0; cannot parse on device");
    return DAMACY_INVAL;
  }

  enum damacy_status s = prepare_decode_caps(wp, wave);
  if (s != DAMACY_OK)
    return s;

  build_gpu_parse_chunks(wp, wave);
  return DAMACY_OK;
}

static enum damacy_status
queue_input_and_parse(struct wave_pool* wp,
                      struct damacy_wave* wave,
                      struct input_transfer_queue_state* state)
{
  state->queued_stream_work = 0;

  enum damacy_status s = wp->input->queue_input(wp->stream_input, wave, state);
  if (s != DAMACY_OK)
    return s;

  return gpu_parse(wp, wave);
}

static enum damacy_status
drain_failed_input_work(struct wave_pool* wp,
                        const struct input_transfer_queue_state* state)
{
  if (!state->queued_stream_work)
    return DAMACY_OK;

  CUresult r = cuStreamSynchronize(wp->stream_input);
  return r == CUDA_SUCCESS ? DAMACY_OK : DAMACY_CUDA;
}

static enum damacy_status
kick_input(struct wave_pool* wp, struct damacy_wave* wave)
{
  damacy_nvtx_range_pushf("kick_input/w%td", wave_index_of(wp, wave));
  struct input_transfer_queue_state queue = { 0 };
  enum damacy_status s = prepare_input_parse(wp, wave);
  if (s != DAMACY_OK)
    goto Error;

  s = queue_input_and_parse(wp, wave, &queue);
  if (s != DAMACY_OK)
    goto Error;

  wave_mark_input(wave);
  damacy_nvtx_range_pop(); // kick_input
  return DAMACY_OK;

Error:
  // Record IO metric on the failure path too so it doesn't bias rolling totals.
  record_io_metric(wp, wave);
  enum damacy_status drain = drain_failed_input_work(wp, &queue);
  if (drain != DAMACY_OK)
    s = drain;
  damacy_nvtx_range_pop(); // kick_input
  return s;
}

static void
set_assemble_chunk_base(struct assemble_chunk* out,
                        const struct chunk_plan* c,
                        uint8_t spatial_rank,
                        uint8_t force_fill)
{
  out->src_base_byte_off = (uint64_t)c->dev_decompressed_offset;
  out->sample_idx_in_batch = c->sample_idx_in_batch;
  out->is_fill = c->is_fill | force_fill;
  memcpy(out->chunk_d, c->chunk_d, spatial_rank * sizeof(*c->chunk_d));
}

static uint64_t
chunk_output_elements(const struct chunk_plan* c,
                      const struct sample_plan* sp,
                      uint8_t spatial_rank)
{
  uint64_t out = 1;
  for (uint8_t d = 0; d < spatial_rank; ++d) {
    int64_t chunk_shape = (int64_t)sp->dims[d].chunk_shape;
    int64_t origin =
      (int64_t)c->chunk_d[d] * chunk_shape - sp->dims[d].aabb_lo_relative;
    int64_t lo = origin > 0 ? origin : 0;
    int64_t hi = origin + chunk_shape < sp->dims[d].aabb_extent
                   ? origin + chunk_shape
                   : sp->dims[d].aabb_extent;
    out *= (uint64_t)(hi > lo ? hi - lo : 0);
  }
  return out;
}

static void
build_assemble_meta(const struct wave_pool* wp, struct damacy_wave* wave)
{
  struct render_job* job = wave_job(wp, wave);
  uint32_t bpe = damacy_dtype_bpe(wp->dtype);
  uint8_t spatial_rank = (uint8_t)(wp->pool->rank - 1);
  uint32_t max_bpc = 0;
  wave->assemble_rank = spatial_rank;
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    struct chunk_plan* c = &job->chunk_plans[wave->batch_chunk_offset + i];
    const struct sample_plan* sp = &job->sample_plans[c->sample_idx_in_batch];
    set_assemble_chunk_base(
      &wave->h_assemble_chunks[i], c, spatial_rank, wp->bypass_decode);

    uint32_t bpc = assemble_blocks_per_chunk(spatial_rank, sp->dims);
    if (bpc > max_bpc)
      max_bpc = bpc;

    wave->assemble_out_bytes +=
      chunk_output_elements(c, sp, spatial_rank) * (uint64_t)bpe;
  }
  if (max_bpc == 0)
    max_bpc = 1;
  wave->assemble_max_blocks_per_chunk = max_bpc;
}

static CUevent
decode_anchor_reserve(struct wave_pool* wp,
                      struct damacy_wave* wave,
                      size_t* anchor_idx)
{
  size_t prev_idx = wp->decode_done_ring_idx;
  *anchor_idx = (prev_idx + 1) % countof(wp->decode_done_ring);
  wave->prev_decode_anchor = wp->decode_done_ring[prev_idx];
  return wp->decode_done_ring[*anchor_idx];
}

static void
decode_anchor_commit(struct wave_pool* wp, size_t anchor_idx)
{
  wp->decode_done_ring_idx = (uint8_t)anchor_idx;
}

static enum damacy_status
queue_zstd_decode(struct wave_pool* wp,
                  struct damacy_wave* wave,
                  const struct blosc1_totals* tot)
{
  if (tot->n_zstd == 0)
    return DAMACY_OK;

  CUstream s = wp->stream_decode;
  uint32_t* d_err = &wave->d_blosc1_totals->n_codec_errors;
  if (decoder_zstd_batch_device(wp->zstd_decoder,
                                s,
                                wave->zstd_fan.d_comp_ptrs,
                                wave->zstd_fan.d_comp_sizes,
                                wave->zstd_fan.d_decomp_ptrs,
                                wave->zstd_fan.d_decomp_buf_sizes,
                                tot->n_zstd))
    return DAMACY_DECODE;
  if (decoder_status_reduce_launch(
        s, decoder_zstd_d_statuses(wp->zstd_decoder), d_err, tot->n_zstd))
    return DAMACY_DECODE;
  return DAMACY_OK;
}

static enum damacy_status
kick_decode(struct wave_pool* wp,
            struct damacy_wave* wave,
            const struct blosc1_totals* tot)
{
  CUstream s = wp->stream_decode;
  enum damacy_status rs;
  damacy_nvtx_range_pushf("kick_decode/w%td", wave_index_of(wp, wave));
  size_t anchor_idx = 0;
  CUevent anchor = decode_anchor_reserve(wp, wave, &anchor_idx);

  CU(CudaFail, cuEventRecord(wave->ev.decomp_start, s));
  rs = queue_zstd_decode(wp, wave, tot);
  if (rs != DAMACY_OK)
    goto Done;

  CU(CudaFail, cuEventRecord(wave->ev.decode_done, s));
  CU(CudaFail, cuEventRecord(anchor, s));
  decode_anchor_commit(wp, anchor_idx);
  rs = DAMACY_OK;
  goto Done;
CudaFail:
  rs = DAMACY_CUDA;
Done:
  damacy_nvtx_range_pop();
  return rs;
}

static enum damacy_status
queue_post_decode(struct wave_pool* wp,
                  struct damacy_wave* wave,
                  const struct blosc1_totals* tot)
{
  CUstream s = wp->stream_post;
  CU(CudaFail, cuStreamWaitEvent(s, wave->ev.decode_done, 0));

  if (tot->n_memcpy > 0 &&
      decoder_memcpy_launch(s, wave->d_memcpy_ops, tot->n_memcpy))
    goto DecodeFail;
  CU(CudaFail,
     cuMemcpyDtoHAsync(&wave->h_blosc1_totals->n_codec_errors,
                       CUDPTR(&wave->d_blosc1_totals->n_codec_errors),
                       sizeof(uint32_t),
                       s));
  CU(CudaFail, cuEventRecord(wave->ev.decomp_end, s));

  return DAMACY_OK;
DecodeFail:
  return DAMACY_DECODE;
CudaFail:
  return DAMACY_CUDA;
}

static enum damacy_status
queue_assemble_output(struct wave_pool* wp, struct damacy_wave* wave)
{
  CUstream s = wp->stream_post;
  struct render_job* job = wave_job(wp, wave);
  struct damacy_batch_slot* batch = &wp->pool->slots[wave->batch_pool_slot];
  CU(CudaFail, cuEventRecord(wave->ev.asm_start, s));
  if (assemble_launch(s,
                      wave->assemble_rank,
                      (const struct sample_plan*)job->d_sample_plans,
                      job->n_sample_plans,
                      wave->d_assemble_chunks,
                      wave->n_chunks,
                      wave->assemble_max_blocks_per_chunk,
                      wave->dev_decompressed,
                      batch->dev_ptr,
                      wp->dtype))
    goto CudaFail;
  CU(CudaFail, cuEventRecord(wave->ev.asm_end, s));
  return DAMACY_OK;
CudaFail:
  return DAMACY_CUDA;
}

static enum damacy_status
kick_assemble(struct wave_pool* wp,
              struct damacy_wave* wave,
              const struct blosc1_totals* tot)
{
  enum damacy_status rs;
  damacy_nvtx_range_pushf("kick_assemble/w%td", wave_index_of(wp, wave));

  rs = queue_post_decode(wp, wave, tot);
  if (rs != DAMACY_OK)
    goto Done;

  rs = queue_assemble_output(wp, wave);
Done:
  damacy_nvtx_range_pop();
  return rs;
}

static void
wave_mark_post(struct damacy_wave* wave)
{
  wave->state = WAVE_POST;
}

static void
wave_mark_free(struct damacy_wave* wave)
{
  wave->state = WAVE_FREE;
  wave->n_chunks = 0;
}

static enum damacy_status
wait_decode_on_input_parse_done(struct wave_pool* wp,
                                const struct damacy_wave* wave)
{
  CU(CudaFail,
     cuStreamWaitEvent(wp->stream_decode, wave->ev.input_parse_done, 0));
  return DAMACY_OK;
CudaFail:
  return DAMACY_CUDA;
}

static struct blosc1_totals
read_parse_totals(const struct damacy_wave* wave)
{
  // h_parse_counters were D2H'd to pinned host on stream_input before
  // input_parse_done was recorded; cuEventQuery(input_parse_done) returning
  // CUDA_SUCCESS (the wave_pool_advance gate that called us) means these host
  // reads need no extra synchronization.
  return (struct blosc1_totals){
    .n_zstd = wave->h_parse_counters[0],
    .n_memcpy = wave->h_parse_counters[1],
    .n_parse_errors = wave->h_parse_counters[2],
  };
}

static enum damacy_status
validate_parse_totals(const struct blosc1_totals* tot)
{
  if (tot->n_parse_errors != 0) {
    log_error("blosc1 gpu_parse: first parse err code=%u", tot->n_parse_errors);
    return DAMACY_DECODE;
  }
  return DAMACY_OK;
}

static enum damacy_status
submit_decode_and_assemble(struct wave_pool* wp,
                           struct damacy_wave* wave,
                           const struct blosc1_totals* tot)
{
  enum damacy_status st = kick_decode(wp, wave, tot);
  if (st != DAMACY_OK)
    return st;
  st = kick_assemble(wp, wave, tot);
  if (st != DAMACY_OK)
    return st;
  return DAMACY_OK;
}

// Wait on input_parse_done, kick decode on stream_decode, kick post + assemble
// on stream_post.
static enum damacy_status
kick_compute(struct wave_pool* wp, struct damacy_wave* wave)
{
  enum damacy_status st = wait_decode_on_input_parse_done(wp, wave);
  if (st != DAMACY_OK)
    return st;

  struct blosc1_totals tot = read_parse_totals(wave);
  st = validate_parse_totals(&tot);
  if (st != DAMACY_OK)
    return st;

  st = submit_decode_and_assemble(wp, wave, &tot);
  if (st != DAMACY_OK)
    return st;

  wave_mark_post(wave);
  return DAMACY_OK;
}

// All wave events have fired; pull elapsed times into stats.
static void
drain_wave_metrics(const struct wave_pool* wp, struct damacy_wave* wave)
{
  struct damacy_stats* st = wp->stats;
  record_io_metric(wp, wave);

  float ms = 0.f;
  if (cuEventElapsedTime(&ms,
                         wave->ev.input_start,
                         wave->ev.input_transfer_done) == CUDA_SUCCESS)
    metric_record(&st->h2d, ms, wave->io_bytes, wave->io_bytes);
  if (cuEventElapsedTime(&ms, wave->ev.decomp_start, wave->ev.decode_done) ==
      CUDA_SUCCESS)
    metric_record(
      &st->decode, ms, wave->decomp_in_bytes, wave->decomp_out_bytes);
  if (cuEventElapsedTime(&ms, wave->ev.decode_done, wave->ev.decomp_end) ==
      CUDA_SUCCESS)
    metric_record(&st->post_decode, ms, 0, 0);
  if (wave->prev_decode_anchor &&
      cuEventElapsedTime(
        &ms, wave->prev_decode_anchor, wave->ev.decomp_start) == CUDA_SUCCESS)
    metric_record(&st->decode_gap, ms, 0, 0);
  if (cuEventElapsedTime(&ms, wave->ev.asm_start, wave->ev.asm_end) ==
      CUDA_SUCCESS)
    metric_record(
      &st->assemble, ms, wave->decomp_out_bytes, wave->assemble_out_bytes);
}

// Drain metrics, return what the caller needs to advance batch state:
// which batch slot retired how many chunks, plus the wave's own error
// status (nvcomp codec errors surface as DAMACY_DECODE). The wave is
// reset to WAVE_FREE before return; the caller applies the batch-state
// transition via batch_slot_consume_chunks.
struct wave_outcome
{
  enum damacy_status status;
  uint16_t batch_pool_slot;
  uint32_t n_chunks_consumed;
};

static struct wave_outcome
finalize_wave(struct wave_pool* wp, struct damacy_wave* wave)
{
  damacy_nvtx_range_pushf("finalize_wave/w%td", wave_index_of(wp, wave));
  drain_wave_metrics(wp, wave);
  struct wave_outcome out = {
    .status = DAMACY_OK,
    .batch_pool_slot = wave->batch_pool_slot,
    .n_chunks_consumed = wave->n_chunks,
  };
  if (wave->h_blosc1_totals->n_codec_errors > 0) {
    log_error("nvcomp: %u substream(s) reported non-success status",
              wave->h_blosc1_totals->n_codec_errors);
    out.status = DAMACY_DECODE;
  }
  wave_mark_free(wave);
  damacy_nvtx_range_pop();
  return out;
}

static void
mark_changed(int* changed);
static void
slot_rollback_peel(struct wave_pool* wp,
                   struct input_slot* slot,
                   const struct wave_desc* desc,
                   int* changed);
static void
wave_unbind_slot(struct wave_pool* wp, struct damacy_wave* wave, int* changed);
static void
record_wave_input_bytes(struct damacy_wave* wave, const struct render_job* job);

struct wave_pool_peel_ticket
wave_pool_peel_reserve(struct wave_pool* wp,
                       uint16_t render_job_idx,
                       enum damacy_status* err)
{
  *err = DAMACY_OK;
  struct wave_pool_peel_ticket t = { .slot_idx = -1,
                                     .n_reads = 0,
                                     .consumed = 0 };
  struct render_job* job = render_job_pool_get(wp->render_jobs, render_job_idx);
  if (!job) {
    *err = DAMACY_INVAL;
    return t;
  }
  if (!render_job_has_work(job))
    return t;
  int slot_idx = input_slot_find_free(wp->slots, wp->n_slots);
  if (slot_idx < 0)
    return t;
  struct input_slot* slot = &wp->slots[slot_idx];

  const struct wave_pack_limits limits = {
    .input_cap = slot->cap,
    .dev_decompressed_cap = wp->waves[0].dev_decompressed_cap,
    .max_chunks_per_wave = wp->max_chunks_per_wave,
  };
  struct wave_desc desc = { 0 };
  enum damacy_status s = wave_dispatcher_reserve(job,
                                                 render_job_idx,
                                                 &limits,
                                                 slot->store_reads,
                                                 wp->input->read_base(slot),
                                                 &desc);
  if (s != DAMACY_OK) {
    *err = s;
    return t;
  }
  input_slot_begin_peel(slot, &desc);
  wp->stats->waves_emitted++;
  wp->stats->chunks_dispatched += desc.n_chunks;

  t.slot_idx = slot_idx;
  t.n_reads = desc.n_reads;
  t.desc = desc;
  return t;
}

struct store_event
wave_pool_peel_submit(struct wave_pool* wp,
                      const struct wave_pool_peel_ticket* t)
{
  if (t->slot_idx < 0 || t->n_reads == 0)
    return (struct store_event){ .seq = 0 };
  struct input_slot* slot = &wp->slots[t->slot_idx];
  return wp->input->submit_reads(wp->store, slot->store_reads, t->n_reads);
}

enum damacy_status
wave_pool_peel_commit(struct wave_pool* wp,
                      struct wave_pool_peel_ticket* t,
                      struct store_event ev,
                      int* changed)
{
  if (t->consumed) {
    log_error("wave: peel_commit called twice on slot %d", t->slot_idx);
    return DAMACY_OK;
  }
  t->consumed = 1;
  if (t->slot_idx < 0)
    return DAMACY_OK;
  struct input_slot* slot = &wp->slots[t->slot_idx];
  if (t->n_reads > 0 && ev.seq == 0) {
    slot_rollback_peel(wp, slot, &t->desc, changed);
    return DAMACY_IO;
  }
  input_slot_commit_io(slot, ev);
  mark_changed(changed);
  return DAMACY_OK;
}

static enum damacy_status
bind_slot_to_wave(struct wave_pool* wp, struct damacy_wave* wave, int slot_idx)
{
  struct input_slot* slot = &wp->slots[slot_idx];
  damacy_nvtx_range_pushf(
    "bind/w%td/slot%d", wave_index_of(wp, wave), slot_idx);
  struct render_job* job =
    render_job_pool_get(wp->render_jobs, slot->render_job_idx);
  if (!job) {
    damacy_nvtx_range_pop();
    return DAMACY_INVAL;
  }
  metric_record(&wp->stats->bind_wait, input_slot_bind_wait_ms(slot), 0, 0);
  wave_bind_input_slot(wave, slot_idx, slot, wp->input->wave_input(wave, slot));
  record_wave_input_bytes(wave, job);

  input_slot_mark_busy(slot);
  build_assemble_meta(wp, wave);
  enum damacy_status s = kick_input(wp, wave);
  if (s != DAMACY_OK) {
    wave_unbind_slot(wp, wave, NULL);
  }
  damacy_nvtx_range_pop();
  return s;
}

static void
mark_changed(int* changed)
{
  if (changed)
    *changed = 1;
}

static void
slot_rollback_peel(struct wave_pool* wp,
                   struct input_slot* slot,
                   const struct wave_desc* desc,
                   int* changed)
{
  render_job_rollback_wave(
    render_job_pool_get(wp->render_jobs, slot->render_job_idx), desc);
  wp->stats->waves_emitted--;
  wp->stats->chunks_dispatched -= slot->n_chunks;
  input_slot_release(slot);
  mark_changed(changed);
}

static void
wave_unbind_slot(struct wave_pool* wp, struct damacy_wave* wave, int* changed)
{
  if (wave->bound_slot >= 0)
    input_slot_release(&wp->slots[wave->bound_slot]);
  wave_unbind_input_slot(wave);
  mark_changed(changed);
}

static void
record_wave_input_bytes(struct damacy_wave* wave, const struct render_job* job)
{
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    const struct chunk_plan* c =
      &job->chunk_plans[wave->batch_chunk_offset + i];
    wave->decomp_in_bytes += c->compressed_nbytes;
    wave->decomp_out_bytes += c->decompressed_nbytes;
  }
}

static enum damacy_status
poll_io_slots(struct wave_pool* wp, int* changed)
{
  for (uint8_t s = 0; s < wp->n_slots; ++s) {
    struct input_slot* slot = &wp->slots[s];
    if (slot->state != SLOT_IO)
      continue;
    int ready =
      slot->is_fill_wave || store_event_query(wp->store, slot->io_event);
    if (!ready)
      continue;

    input_slot_mark_ready(slot);
    mark_changed(changed);
  }
  return DAMACY_OK;
}

static enum damacy_status
bind_ready_slots_to_free_waves(struct wave_pool* wp, int* changed)
{
  for (int w = 0; w < DAMACY_N_WAVES; ++w) {
    struct damacy_wave* wave = &wp->waves[w];
    if (wave->state != WAVE_FREE)
      continue;
    int rs = input_slot_find_ready(wp->slots, wp->n_slots);
    if (rs < 0)
      break;
    enum damacy_status s = bind_slot_to_wave(wp, wave, rs);
    if (s != DAMACY_OK)
      return s;
    mark_changed(changed);
  }
  return DAMACY_OK;
}

static enum damacy_status
release_bound_slot_after_input_consumed(struct wave_pool* wp,
                                        struct damacy_wave* wave,
                                        int* changed)
{
  if (wave->bound_slot < 0)
    return DAMACY_OK;

  int ready = 0;
  enum damacy_status s = wp->input->slot_reuse_ready(wave, &ready);
  if (s != DAMACY_OK)
    return s;
  if (ready) {
    wave_unbind_slot(wp, wave, changed);
    return DAMACY_OK;
  }
  return DAMACY_OK;
}

static enum damacy_status
poll_input_wave(struct wave_pool* wp, struct damacy_wave* wave, int* changed)
{
  enum damacy_status s =
    release_bound_slot_after_input_consumed(wp, wave, changed);
  if (s != DAMACY_OK)
    return s;

  CUresult qe = cuEventQuery(wave->ev.input_parse_done);
  if (qe == CUDA_ERROR_NOT_READY)
    return DAMACY_OK;
  if (qe != CUDA_SUCCESS)
    return DAMACY_CUDA;

  // Both kicks enqueue decode + status-reduce against the shared decoder
  // scratch on stream_decode. Safety relies on stream FIFO: wave A fully
  // retires those uses before wave B's decode launches.
  s = kick_compute(wp, wave);
  if (s != DAMACY_OK)
    return s;
  mark_changed(changed);
  return DAMACY_OK;
}

static enum damacy_status
retire_posted_wave(struct wave_pool* wp, struct damacy_wave* wave, int* changed)
{
  struct wave_outcome o = finalize_wave(wp, wave);
  struct damacy_batch_slot* batch = &wp->pool->slots[o.batch_pool_slot];
  batch_slot_consume_chunks(batch, o.n_chunks_consumed);
  if (batch->state == BATCH_READY)
    render_job_finish(
      render_job_pool_get(wp->render_jobs, wave->render_job_idx));
  mark_changed(changed);
  return o.status;
}

static enum damacy_status
poll_post_wave(struct wave_pool* wp, struct damacy_wave* wave, int* changed)
{
  enum damacy_status s =
    release_bound_slot_after_input_consumed(wp, wave, changed);
  if (s != DAMACY_OK)
    return s;

  CUresult qe = cuEventQuery(wave->ev.asm_end);
  if (qe == CUDA_ERROR_NOT_READY)
    return DAMACY_OK;
  if (qe != CUDA_SUCCESS)
    return DAMACY_CUDA;
  return retire_posted_wave(wp, wave, changed);
}

static enum damacy_status
poll_one_wave(struct wave_pool* wp, struct damacy_wave* wave, int* changed)
{
  switch (wave->state) {
    case WAVE_FREE:
      return DAMACY_OK;
    case WAVE_INPUT:
      return poll_input_wave(wp, wave, changed);
    case WAVE_POST:
      return poll_post_wave(wp, wave, changed);
  }
  return DAMACY_OK;
}

static enum damacy_status
poll_active_waves(struct wave_pool* wp, int* changed)
{
  for (int w = 0; w < DAMACY_N_WAVES; ++w) {
    enum damacy_status s = poll_one_wave(wp, &wp->waves[w], changed);
    if (s != DAMACY_OK)
      return s;
  }
  return DAMACY_OK;
}

enum damacy_status
wave_pool_advance(struct wave_pool* wp, int* changed)
{
  enum damacy_status s = poll_io_slots(wp, changed);
  if (s != DAMACY_OK)
    return s;

  s = bind_ready_slots_to_free_waves(wp, changed);
  if (s != DAMACY_OK)
    return s;

  return poll_active_waves(wp, changed);
}
