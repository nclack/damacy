#include "wave.h"

#include "batch_pool/batch_pool.h"
#include "damacy_config.h"
#include "damacy_stats.h"
#include "decoder/bitshuffle.h"
#include "decoder/shuffle.h"
#include "decoder/status_reduce.h"
#include "log/log.h"
#include "planner/planner.h"
#include "store/store.h"
#include "util/cuda_check.h"
#include "util/prelude.h"
#include "util/strbuf.h"

#include <stdlib.h>
#include <string.h>

int
fanout_alloc_pinned(struct blosc1_host_fanout* h,
                    struct nvcomp_fanout* d,
                    size_t n)
{
  CUdeviceptr dptr = 0;
  CU(Fail, cuMemAllocHost((void**)&h->comp_ptrs, n * sizeof(void*)));
  CU(Fail, cuMemAllocHost((void**)&h->comp_sizes, n * sizeof(size_t)));
  CU(Fail, cuMemAllocHost((void**)&h->decomp_ptrs, n * sizeof(void*)));
  CU(Fail, cuMemAllocHost((void**)&h->decomp_buf_sizes, n * sizeof(size_t)));
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(void*)));
  d->d_comp_ptrs = (const void**)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(size_t)));
  d->d_comp_sizes = (size_t*)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(void*)));
  d->d_decomp_ptrs = (void**)(uintptr_t)dptr;
  CU(Fail, cuMemAlloc(&dptr, n * sizeof(size_t)));
  d->d_decomp_buf_sizes = (size_t*)(uintptr_t)dptr;
  return 0;
Fail:
  return 1;
}

enum damacy_status
fanout_upload(CUstream s,
              const struct nvcomp_fanout* d,
              const struct blosc1_host_fanout* h,
              size_t n)
{
  CU(Fail,
     cuMemcpyHtoDAsync(
       CUDPTR(d->d_comp_ptrs), h->comp_ptrs, n * sizeof(void*), s));
  CU(Fail,
     cuMemcpyHtoDAsync(
       CUDPTR(d->d_comp_sizes), h->comp_sizes, n * sizeof(size_t), s));
  CU(Fail,
     cuMemcpyHtoDAsync(
       CUDPTR(d->d_decomp_ptrs), h->decomp_ptrs, n * sizeof(void*), s));
  CU(Fail,
     cuMemcpyHtoDAsync(CUDPTR(d->d_decomp_buf_sizes),
                       h->decomp_buf_sizes,
                       n * sizeof(size_t),
                       s));
  return DAMACY_OK;
Fail:
  return DAMACY_CUDA;
}

// Substream/batch caps the shared zstd decoder is sized against. zsubs
// is the worst-case substream count, zstd_per is the per-substream
// uncompressed upper bound, total_uncompressed is the per-batch sum.
// Single source of truth used by wave_pool_init (decoder scratch
// sizing) and wave_pool_shared_predict_bytes (GPU budget).
static void
wave_decoder_caps(uint64_t dev_decompressed_bytes,
                  uint64_t max_chunk_uncompressed_bytes,
                  uint64_t* out_zsubs,
                  uint64_t* out_zstd_per,
                  uint64_t* out_total_uncompressed)
{
  *out_zsubs = (uint64_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;
  // Caller's config validation should preclude this; guard explicitly
  // so downstream nvcomp queries never see zero-sized inputs.
  if (dev_decompressed_bytes == 0 || max_chunk_uncompressed_bytes == 0) {
    *out_zstd_per = 0;
    *out_total_uncompressed = 0;
    return;
  }
  const uint64_t cap_chunks = (uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE;
  const uint64_t runtime_chunk_cap = max_chunk_uncompressed_bytes;
  const uint64_t cap_worst = cap_chunks * runtime_chunk_cap;
  const uint64_t total_uncompressed =
    dev_decompressed_bytes < cap_worst ? dev_decompressed_bytes : cap_worst;
  uint64_t zstd_per = runtime_chunk_cap;
  if (zstd_per > total_uncompressed)
    zstd_per = total_uncompressed;
  *out_zstd_per = zstd_per;
  *out_total_uncompressed = total_uncompressed;
}

enum damacy_status
wave_predict_bytes(uint64_t host_slab_bytes,
                   uint64_t dev_decompressed_bytes,
                   struct wave_alloc_summary* out)
{
  const uint64_t cap = (uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE;
  const uint64_t zsubs = (uint64_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;

  out->dev_compressed = host_slab_bytes;
  out->dev_decompressed = dev_decompressed_bytes;
  out->dev_unshuffle_scratch = dev_decompressed_bytes;
  out->blosc1_meta =
    cap * sizeof(struct assemble_chunk) + sizeof(struct blosc1_totals);
  out->fanout_soa = zsubs * (2 * sizeof(void*) + 2 * sizeof(size_t)) +
                    (uint64_t)DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                      sizeof(struct gpu_memcpy_op) +
                    2ull * (uint64_t)DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                      sizeof(struct gpu_shuffle_op);
  return DAMACY_OK;
}

enum damacy_status
wave_pool_shared_predict_bytes(uint64_t dev_decompressed_bytes,
                               uint64_t max_chunk_uncompressed_bytes,
                               uint64_t* out_nvcomp_temp)
{
  uint64_t zsubs = 0, zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps(dev_decompressed_bytes,
                    max_chunk_uncompressed_bytes,
                    &zsubs,
                    &zstd_per,
                    &total_uncompressed);

  size_t zstd_temp = 0;
  if (decoder_zstd_query_temp_bytes(
        zsubs, (size_t)zstd_per, (size_t)total_uncompressed, &zstd_temp))
    return DAMACY_CUDA;
  *out_nvcomp_temp =
    (uint64_t)zstd_temp + zsubs * sizeof(size_t) + zsubs * sizeof(int);
  return DAMACY_OK;
}

int
wave_init(struct damacy_wave* wave,
          uint64_t host_slab_bytes,
          uint64_t dev_decompressed_bytes)
{
  // Self-zero so wave_destroy on the failure path doesn't free
  // uninitialized pointers — caller may have passed stack memory.
  memset(wave, 0, sizeof(*wave));
  wave->state = WAVE_FREE;
  wave->host_slab_cap = host_slab_bytes;
  wave->dev_decompressed_cap = dev_decompressed_bytes;

  CU(Error, cuMemAllocHost(&wave->host_slab, host_slab_bytes));
  CUdeviceptr dptr = 0;
  CU(Error, cuMemAlloc(&dptr, host_slab_bytes));
  wave->dev_compressed = (void*)(uintptr_t)dptr;
  CU(Error, cuMemAlloc(&dptr, dev_decompressed_bytes));
  wave->dev_decompressed = (void*)(uintptr_t)dptr;

  uint32_t cap = DAMACY_MAX_CHUNKS_PER_WAVE;
  CU(Error,
     cuMemAllocHost((void**)&wave->h_chunks,
                    (size_t)cap * sizeof(struct blosc1_host_chunk)));
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.hdrs,
                    (size_t)cap * sizeof(struct blosc1_chunk_hdr)));
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.counts,
                    (size_t)cap * sizeof(struct blosc1_chunk_counts)));
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.offsets,
                    (size_t)cap * sizeof(struct blosc1_chunk_offsets)));
  // Pure host scratch, but pinned for consistency with the rest of the
  // scratch arrays. cap * MAX_BLOCKS uint32_t each (~64 KB at current caps).
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.bstarts,
                    (size_t)cap * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK *
                      sizeof(uint32_t)));
  CU(Error,
     cuMemAllocHost((void**)&wave->scratch.block_ends,
                    (size_t)cap * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK *
                      sizeof(uint32_t)));
  CU(Error,
     cuMemAllocHost((void**)&wave->h_blosc1_totals,
                    sizeof(struct blosc1_totals)));
  CU(Error,
     cuMemAllocHost((void**)&wave->h_assemble_chunks,
                    (size_t)cap * sizeof(struct assemble_chunk)));
  wave->store_reads =
    (struct store_read*)calloc(cap, sizeof(struct store_read));
  CHECK(Error, wave->store_reads);

  CU(Error, cuMemAlloc(&dptr, (size_t)cap * sizeof(struct assemble_chunk)));
  wave->d_assemble_chunks = (struct assemble_chunk*)(uintptr_t)dptr;

  CU(Error, cuMemAlloc(&dptr, sizeof(struct blosc1_totals)));
  wave->d_blosc1_totals = (struct blosc1_totals*)(uintptr_t)dptr;

  const size_t zsubs = DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;
  if (fanout_alloc_pinned(&wave->h_zstd_fan, &wave->zstd_fan, zsubs))
    goto Error;

  CU(Error,
     cuMemAllocHost((void**)&wave->h_memcpy_ops,
                    DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                      sizeof(struct gpu_memcpy_op)));
  CU(Error,
     cuMemAllocHost((void**)&wave->h_unshuffle_ops,
                    DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                      sizeof(struct gpu_shuffle_op)));
  CU(Error,
     cuMemAllocHost((void**)&wave->h_bitunshuffle_ops,
                    DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                      sizeof(struct gpu_shuffle_op)));
  CU(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                  sizeof(struct gpu_memcpy_op)));
  wave->d_memcpy_ops = (struct gpu_memcpy_op*)(uintptr_t)dptr;
  CU(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                  sizeof(struct gpu_shuffle_op)));
  wave->d_unshuffle_ops = (struct gpu_shuffle_op*)(uintptr_t)dptr;
  CU(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                  sizeof(struct gpu_shuffle_op)));
  wave->d_bitunshuffle_ops = (struct gpu_shuffle_op*)(uintptr_t)dptr;
  CU(Error, cuMemAlloc(&dptr, dev_decompressed_bytes));
  wave->dev_unshuffle_scratch = (void*)(uintptr_t)dptr;

  CU(Error, cuEventCreate(&wave->ev.h2d_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.bulk_h2d_end, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.h2d_end, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.decomp_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.decomp_end, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.asm_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.asm_end, CU_EVENT_DEFAULT));

  return 0;
Error:
  // Make the partial-init cleanup explicit instead of relying on the
  // outer destroy_inner walking a zero-initialized wave.
  wave_destroy(wave, 0);
  return 1;
}

void
wave_destroy(struct damacy_wave* wave, int cuda_skip)
{
  if (!wave)
    return;
  if (!cuda_skip) {
    void* const host_ptrs[] = {
      wave->host_slab,
      wave->h_chunks,
      wave->scratch.hdrs,
      wave->scratch.counts,
      wave->scratch.offsets,
      wave->scratch.bstarts,
      wave->scratch.block_ends,
      wave->h_blosc1_totals,
      (void*)wave->h_zstd_fan.comp_ptrs,
      wave->h_zstd_fan.comp_sizes,
      wave->h_zstd_fan.decomp_ptrs,
      wave->h_zstd_fan.decomp_buf_sizes,
      wave->h_memcpy_ops,
      wave->h_unshuffle_ops,
      wave->h_bitunshuffle_ops,
      wave->h_assemble_chunks,
    };
    for (size_t i = 0; i < countof(host_ptrs); ++i)
      if (host_ptrs[i])
        cuMemFreeHost(host_ptrs[i]);
    void* const dev_ptrs[] = {
      wave->dev_compressed,
      wave->dev_decompressed,
      wave->d_assemble_chunks,
      wave->d_blosc1_totals,
      (void*)wave->zstd_fan.d_comp_ptrs,
      wave->zstd_fan.d_comp_sizes,
      wave->zstd_fan.d_decomp_ptrs,
      wave->zstd_fan.d_decomp_buf_sizes,
      wave->d_memcpy_ops,
      wave->d_unshuffle_ops,
      wave->d_bitunshuffle_ops,
      wave->dev_unshuffle_scratch,
    };
    for (size_t i = 0; i < countof(dev_ptrs); ++i)
      if (dev_ptrs[i])
        cuMemFree(CUDPTR(dev_ptrs[i]));
    CUevent* const events[] = { &wave->ev.h2d_start,  &wave->ev.bulk_h2d_end,
                                &wave->ev.h2d_end,    &wave->ev.decomp_start,
                                &wave->ev.decomp_end, &wave->ev.asm_start,
                                &wave->ev.asm_end };
    for (size_t i = 0; i < countof(events); ++i)
      if (*events[i])
        cuEventDestroy_v2(*events[i]);
  }
  free(wave->store_reads);
  memset(wave, 0, sizeof(*wave));
}

int
wave_pool_init(struct wave_pool* wp,
               struct damacy_batch_pool* pool,
               struct store* store,
               struct threadpool* compute_pool,
               struct damacy_stats* stats,
               enum damacy_status* failed_status,
               enum damacy_dtype dtype,
               uint64_t host_buffer_bytes,
               uint64_t device_buffer_bytes,
               uint64_t max_chunk_uncompressed_bytes)
{
  memset(wp, 0, sizeof(*wp));
  wp->pool = pool;
  wp->store = store;
  wp->compute_pool = compute_pool;
  wp->stats = stats;
  wp->failed_status = failed_status;
  wp->dtype = dtype;

  // Non-blocking so damacy doesn't force-serialize against the legacy
  // default stream that some user code still lands on.
  CU(Fail, cuStreamCreate(&wp->stream_h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&wp->stream_decode, CU_STREAM_NON_BLOCKING));

  const uint64_t host_per_wave = host_buffer_bytes / 2;
  const uint64_t dev_per_wave = device_buffer_bytes / 2;

  // Shared decoder is sized for one wave's worth of substreams since
  // decodes serialize FIFO on stream_decode (M=1 in-flight).
  uint64_t zsubs = 0, zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps(dev_per_wave,
                    max_chunk_uncompressed_bytes,
                    &zsubs,
                    &zstd_per,
                    &total_uncompressed);
  wp->zstd_decoder = decoder_zstd_create(
    (size_t)zsubs, (size_t)zstd_per, (size_t)total_uncompressed);
  CHECK(Fail, wp->zstd_decoder);

  for (int w = 0; w < 2; ++w)
    if (wave_init(&wp->waves[w], host_per_wave, dev_per_wave) != 0)
      goto Fail;
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
  CUstream* const streams[] = { &wp->stream_decode, &wp->stream_h2d };
  if (!cuda_skip) {
    // Sync streams (but don't destroy yet) so any pending GPU work
    // touching wave + shared-decoder buffers has retired before we
    // free those buffers.
    for (size_t i = 0; i < countof(streams); ++i)
      if (*streams[i])
        cuStreamSynchronize(*streams[i]);
  }
  for (int w = 0; w < 2; ++w)
    wave_destroy(&wp->waves[w], cuda_skip);
  decoder_zstd_destroy(wp->zstd_decoder, cuda_skip);
  wp->zstd_decoder = NULL;
  if (!cuda_skip) {
    for (size_t i = 0; i < countof(streams); ++i)
      if (*streams[i]) {
        cuStreamDestroy(*streams[i]);
        *streams[i] = NULL;
      }
  }
  // Borrowed pointers — owner frees them.
  wp->pool = NULL;
  wp->store = NULL;
  wp->compute_pool = NULL;
  wp->stats = NULL;
  wp->failed_status = NULL;
}

int
find_free_wave(const struct wave_pool* wp)
{
  for (int w = 0; w < 2; ++w)
    if (wp->waves[w].state == WAVE_FREE)
      return w;
  return -1;
}

int
any_wave_in_flight(const struct wave_pool* wp)
{
  for (int w = 0; w < 2; ++w)
    if (wp->waves[w].state != WAVE_FREE)
      return 1;
  return 0;
}

// --- peel / advance -------------------------------------------------------

// host_slab and dev_compressed share offsets because kick_h2d copies
// the slab byte-for-byte.
static void
build_blosc1_host_chunks(const struct wave_pool* wp, struct damacy_wave* wave)
{
  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    struct chunk_plan* c = &slot->chunk_plans[wave->batch_chunk_offset + i];
    struct read_op* r = &slot->read_ops[wave->batch_chunk_offset + i];
    struct blosc1_host_chunk* hc = &wave->h_chunks[i];
    size_t base_off = (size_t)r->dst_buf_offset + (size_t)c->offset_in_read;
    hc->h_compressed = (const uint8_t*)wave->host_slab + base_off;
    hc->d_compressed = (uint8_t*)wave->dev_compressed + base_off;
    hc->d_decompressed =
      (uint8_t*)wave->dev_decompressed + c->dev_decompressed_offset;
    hc->compressed_nbytes = c->compressed_nbytes;
    hc->decompressed_nbytes = c->decompressed_nbytes;
    hc->codec_id = c->codec_id;
  }
}

// One log line per failing chunk (blosc1_host logs only the count).
static void
log_blosc1_parse_errors(const struct wave_pool* wp, struct damacy_wave* wave)
{
  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];
  struct strbuf coords = { 0 };
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    const struct blosc1_chunk_hdr* h = &wave->scratch.hdrs[i];
    if (h->err == 0)
      continue;
    const struct chunk_plan* c =
      &slot->chunk_plans[wave->batch_chunk_offset + i];
    const struct sample_plan* sp = &slot->sample_plans[c->sample_idx_in_batch];
    strbuf_reset(&coords);
    strbuf_append_cstr(&coords, "[");
    for (uint8_t d = 0; d < sp->rank; ++d)
      strbuf_appendf(&coords, d == 0 ? "%u" : ",%u", c->chunk_d[d]);
    strbuf_append_cstr(&coords, "]");
    log_error("blosc1: parse failed: batch_id=%llu sample=%u chunk_d=%s "
              "codec_id=%u err=%u (%s)",
              (unsigned long long)slot->batch_id,
              (unsigned)c->sample_idx_in_batch,
              strbuf_cstr(&coords),
              (unsigned)c->codec_id,
              (unsigned)h->err,
              blosc1_host_parse_err_str(h->err));
  }
  strbuf_free(&coords);
}

// IO already retired into io_t_end_ns by the caller. Push io timing into
// stats.io so failure paths don't bias the rolling totals.
static void
record_io_metric(const struct wave_pool* wp, const struct damacy_wave* wave)
{
  float io_ms = (float)((wave->io_t_end_ns - wave->io_t_start_ns) / 1.0e6);
  metric_record(&wp->stats->io, io_ms, wave->io_bytes, wave->io_bytes);
}

// Bulk H2D, host parse overlapping the DMA, fanout/op H2Ds, then
// h2d_end. stream_decode gates on h2d_end before nvcomp launch.
static enum damacy_status
kick_h2d(struct wave_pool* wp, struct damacy_wave* wave)
{
  CU(CudaFail, cuEventRecord(wave->ev.h2d_start, wp->stream_h2d));
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->dev_compressed),
                       wave->host_slab,
                       wave->host_used_bytes,
                       wp->stream_h2d));
  // Record bulk_h2d_end before queueing fanout/op H2Ds so stats.h2d
  // measures just the slab copy.
  CU(CudaFail, cuEventRecord(wave->ev.bulk_h2d_end, wp->stream_h2d));

  build_blosc1_host_chunks(wp, wave);

  uint64_t parse_t0 = monotonic_ns();
  int rc = blosc1_host_parse(&(struct blosc1_host_parse_args){
    .pool = wp->compute_pool,
    .chunks = wave->h_chunks,
    .n_chunks = wave->n_chunks,
    .scratch = wave->scratch,
    .zstd = wave->h_zstd_fan,
    .memcpy_ops = wave->h_memcpy_ops,
    .unshuffle_ops = wave->h_unshuffle_ops,
    .bitunshuffle_ops = wave->h_bitunshuffle_ops,
    .out_totals = wave->h_blosc1_totals,
  });
  wave->parse_ms = (float)((monotonic_ns() - parse_t0) / 1.0e6);
  if (rc) {
    // Drain the IO metric before bailing — finalize_wave won't run for
    // this wave and we don't want failed runs to silently undercount IO.
    record_io_metric(wp, wave);
    log_blosc1_parse_errors(wp, wave);
    cuEventRecord(wave->ev.h2d_end, wp->stream_h2d);
    *wp->failed_status = DAMACY_DECODE;
    return DAMACY_DECODE;
  }

  const struct blosc1_totals* tot = wave->h_blosc1_totals;
  if (tot->n_zstd > 0 &&
      fanout_upload(wp->stream_h2d,
                    &wave->zstd_fan,
                    &wave->h_zstd_fan,
                    (size_t)tot->n_zstd) != DAMACY_OK)
    goto CudaFail;
  if (tot->n_memcpy > 0)
    CU(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_memcpy_ops),
                         wave->h_memcpy_ops,
                         (size_t)tot->n_memcpy * sizeof(struct gpu_memcpy_op),
                         wp->stream_h2d));
  if (tot->n_unshuffle > 0)
    CU(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_unshuffle_ops),
                         wave->h_unshuffle_ops,
                         (size_t)tot->n_unshuffle *
                           sizeof(struct gpu_shuffle_op),
                         wp->stream_h2d));
  if (tot->n_bitunshuffle > 0)
    CU(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_bitunshuffle_ops),
                         wave->h_bitunshuffle_ops,
                         (size_t)tot->n_bitunshuffle *
                           sizeof(struct gpu_shuffle_op),
                         wp->stream_h2d));

  // Zero so status_reduce's atomicAdds land in a clean n_codec_errors.
  CU(CudaFail,
     cuMemsetD8Async(CUDPTR(wave->d_blosc1_totals),
                     0,
                     sizeof(struct blosc1_totals),
                     wp->stream_h2d));

  CU(CudaFail, cuEventRecord(wave->ev.h2d_end, wp->stream_h2d));
  wave->state = WAVE_H2D;
  return DAMACY_OK;
CudaFail:
  *wp->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
}

static void
build_assemble_meta(const struct wave_pool* wp, struct damacy_wave* wave)
{
  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];
  uint32_t bpe = damacy_dtype_bpe(wp->dtype);
  uint8_t spatial_rank = (uint8_t)(wp->pool->rank - 1);
  uint32_t max_bpc = 0;
  wave->assemble_rank = spatial_rank;
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    struct chunk_plan* c = &slot->chunk_plans[wave->batch_chunk_offset + i];
    struct assemble_chunk* a = &wave->h_assemble_chunks[i];
    a->src_base_byte_off = (uint64_t)c->dev_decompressed_offset;
    a->sample_idx_in_batch = c->sample_idx_in_batch;
    for (uint8_t d = 0; d < spatial_rank; ++d)
      a->chunk_d[d] = c->chunk_d[d];

    const struct sample_plan* sp = &slot->sample_plans[c->sample_idx_in_batch];
    uint32_t bpc = assemble_blocks_per_chunk(spatial_rank, sp->dims);
    if (bpc > max_bpc)
      max_bpc = bpc;

    // Effective in-AABB extent for this chunk along each axis.
    uint64_t eff = 1;
    for (uint8_t d = 0; d < spatial_rank; ++d) {
      int64_t S = (int64_t)sp->dims[d].chunk_shape;
      int64_t origin_in_sample =
        (int64_t)c->chunk_d[d] * S - sp->dims[d].aabb_lo_relative;
      int64_t lo = origin_in_sample > 0 ? origin_in_sample : 0;
      int64_t hi = origin_in_sample + S < sp->dims[d].aabb_extent
                     ? origin_in_sample + S
                     : sp->dims[d].aabb_extent;
      int64_t extent = hi > lo ? hi - lo : 0;
      eff *= (uint64_t)extent;
    }
    wave->assemble_out_bytes += eff * (uint64_t)bpe;
  }
  if (max_bpc == 0)
    max_bpc = 1;
  wave->assemble_max_blocks_per_chunk = max_bpc;
}

// Nvcomp zstd batch + post-decode (memcpy / (bit)unshuffle) on
// stream_decode in FIFO order. decomp_start brackets the nvcomp launch;
// decomp_end fires after the post-decode kernels complete.
static enum damacy_status
kick_decode(struct wave_pool* wp,
            struct damacy_wave* wave,
            const struct blosc1_totals* tot)
{
  CUstream s = wp->stream_decode;
  uint32_t* d_err = &wave->d_blosc1_totals->n_codec_errors;
  CU(CudaFail, cuEventRecord(wave->ev.decomp_start, s));
  if (tot->n_zstd > 0) {
    if (decoder_zstd_batch_device(wp->zstd_decoder,
                                  s,
                                  wave->zstd_fan.d_comp_ptrs,
                                  wave->zstd_fan.d_comp_sizes,
                                  wave->zstd_fan.d_decomp_ptrs,
                                  wave->zstd_fan.d_decomp_buf_sizes,
                                  tot->n_zstd))
      goto DecodeFail;
    if (decoder_status_reduce_launch(
          s, decoder_zstd_d_statuses(wp->zstd_decoder), d_err, tot->n_zstd))
      goto DecodeFail;
  }
  if (tot->n_memcpy > 0 &&
      decoder_memcpy_launch(s, wave->d_memcpy_ops, tot->n_memcpy))
    goto DecodeFail;
  if (tot->n_unshuffle > 0 && gpu_unshuffle_launch(s,
                                                   wave->d_unshuffle_ops,
                                                   tot->n_unshuffle,
                                                   wave->dev_decompressed,
                                                   wave->dev_unshuffle_scratch))
    goto DecodeFail;
  if (tot->n_bitunshuffle > 0 &&
      gpu_bitunshuffle_launch(s,
                              wave->d_bitunshuffle_ops,
                              tot->n_bitunshuffle,
                              wave->dev_decompressed,
                              wave->dev_unshuffle_scratch))
    goto DecodeFail;
  // Narrowed to the 4-byte n_codec_errors so the host parse's count
  // fields in h_blosc1_totals stay intact for drain_wave_metrics.
  CU(CudaFail,
     cuMemcpyDtoHAsync(&wave->h_blosc1_totals->n_codec_errors,
                       CUDPTR(&wave->d_blosc1_totals->n_codec_errors),
                       sizeof(uint32_t),
                       s));
  CU(CudaFail, cuEventRecord(wave->ev.decomp_end, s));
  return DAMACY_OK;
DecodeFail:
  *wp->failed_status = DAMACY_DECODE;
  return DAMACY_DECODE;
CudaFail:
  *wp->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
}

static enum damacy_status
kick_assemble(struct wave_pool* wp, struct damacy_wave* wave)
{
  CUstream s = wp->stream_decode;
  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];

  build_assemble_meta(wp, wave);
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_assemble_chunks),
                       wave->h_assemble_chunks,
                       (size_t)wave->n_chunks * sizeof(struct assemble_chunk),
                       s));
  CU(CudaFail, cuEventRecord(wave->ev.asm_start, s));
  if (assemble_launch(s,
                      wave->assemble_rank,
                      (const struct sample_plan*)slot->d_sample_plans,
                      slot->n_sample_plans,
                      wave->d_assemble_chunks,
                      wave->n_chunks,
                      wave->assemble_max_blocks_per_chunk,
                      wave->dev_decompressed,
                      slot->dev_ptr,
                      wp->dtype)) {
    *wp->failed_status = DAMACY_CUDA;
    return DAMACY_CUDA;
  }
  CU(CudaFail, cuEventRecord(wave->ev.asm_end, s));
  return DAMACY_OK;
CudaFail:
  *wp->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
}

// stream_decode gates on stream_h2d's h2d_end, then nvcomp decode +
// post-decode + assemble all run in FIFO on stream_decode.
static enum damacy_status
kick_compute(struct wave_pool* wp, struct damacy_wave* wave)
{
  CU(CudaFail, cuStreamWaitEvent(wp->stream_decode, wave->ev.h2d_end, 0));

  const struct blosc1_totals tot = *wave->h_blosc1_totals;
  enum damacy_status st = kick_decode(wp, wave, &tot);
  if (st != DAMACY_OK)
    return st;
  st = kick_assemble(wp, wave);
  if (st != DAMACY_OK)
    return st;

  wave->state = WAVE_ASSEMBLE;
  return DAMACY_OK;
CudaFail:
  *wp->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
}

// All wave events have fired (asm_end signaled implies everything
// earlier on the same stream did too). Pull elapsed times into stats.
static void
drain_wave_metrics(const struct wave_pool* wp, struct damacy_wave* wave)
{
  struct damacy_stats* st = wp->stats;
  record_io_metric(wp, wave);

  float ms = 0.f;
  if (cuEventElapsedTime(&ms, wave->ev.h2d_start, wave->ev.bulk_h2d_end) ==
      CUDA_SUCCESS)
    metric_record(&st->h2d, ms, wave->io_bytes, wave->io_bytes);
  if (cuEventElapsedTime(&ms, wave->ev.decomp_start, wave->ev.decomp_end) ==
      CUDA_SUCCESS)
    metric_record(&st->decompress,
                  ms,
                  wave->decomp_in_bytes,
                  wave->decomp_out_bytes);
  metric_record(&st->decompress_parse, wave->parse_ms, 0, 0);
  if (cuEventElapsedTime(&ms, wave->ev.asm_start, wave->ev.asm_end) ==
      CUDA_SUCCESS)
    metric_record(&st->assemble,
                  ms,
                  wave->decomp_out_bytes,
                  wave->assemble_out_bytes);
}

// Surfaces nvcomp errors before the slot transitions so damacy_pop's
// failed_status check can bail before handing out the batch.
static void
finalize_wave(struct wave_pool* wp, struct damacy_wave* wave)
{
  drain_wave_metrics(wp, wave);
  if (wave->h_blosc1_totals->n_codec_errors > 0 &&
      *wp->failed_status == DAMACY_OK) {
    log_error("nvcomp: %u substream(s) reported non-success status",
              wave->h_blosc1_totals->n_codec_errors);
    *wp->failed_status = DAMACY_DECODE;
  }
  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];
  slot->chunks_remaining -= (int32_t)wave->n_chunks;
  if (slot->chunks_remaining <= 0) {
    slot->chunks_remaining = 0;
    if (slot->state == BATCH_FILLING)
      slot->state = BATCH_READY;
  }
  wave->state = WAVE_FREE;
  wave->n_chunks = 0;
}

enum damacy_status
wave_pool_peel(struct wave_pool* wp, uint16_t wave_idx, uint16_t slot_idx)
{
  struct damacy_wave* wave = &wp->waves[wave_idx];
  struct damacy_batch_slot* slot = &wp->pool->slots[slot_idx];
  uint32_t base = slot->n_chunks_dispatched;
  uint32_t remaining = slot->n_chunks - base;
  if (remaining == 0)
    return DAMACY_OK;

  uint64_t host_cursor = 0;
  uint64_t dev_cursor = 0;
  uint32_t take = 0;
  for (; take < remaining && take < DAMACY_MAX_CHUNKS_PER_WAVE; ++take) {
    struct read_op* r = &slot->read_ops[base + take];
    struct chunk_plan* c = &slot->chunk_plans[base + take];
    if (host_cursor + r->nbytes > wave->host_slab_cap)
      break;
    if (dev_cursor + c->decompressed_nbytes > wave->dev_decompressed_cap)
      break;
    r->dst_buf_offset = host_cursor;
    c->dev_decompressed_offset = dev_cursor;
    host_cursor += r->nbytes;
    dev_cursor += c->decompressed_nbytes;
  }
  if (take == 0) {
    // Single chunk doesn't fit. Per-wave caps too tight for this workload;
    // surface it loudly rather than livelocking.
    log_error("wave: chunk too large for wave slab "
              "(host_slab_cap=%llu device_buf_cap=%llu)",
              (unsigned long long)wave->host_slab_cap,
              (unsigned long long)wave->dev_decompressed_cap);
    *wp->failed_status = DAMACY_OOM;
    return DAMACY_OOM;
  }

  for (uint32_t i = 0; i < take; ++i) {
    struct read_op* r = &slot->read_ops[base + i];
    wave->store_reads[i] = (struct store_read){
      .key = r->shard_path,
      .dst = (uint8_t*)wave->host_slab + r->dst_buf_offset,
      .offset = r->file_offset,
      .len = r->nbytes,
    };
  }
  wave->io_t_start_ns = monotonic_ns();
  wave->io_event = store_read_submit(wp->store, wave->store_reads, take);
  if (wave->io_event.seq == 0) {
    *wp->failed_status = DAMACY_IO;
    return DAMACY_IO;
  }

  wave->batch_pool_slot = slot_idx;
  wave->batch_chunk_offset = base;
  wave->n_chunks = take;
  wave->host_used_bytes = host_cursor;
  wave->io_bytes = host_cursor;
  wave->decomp_in_bytes = 0;
  wave->decomp_out_bytes = 0;
  wave->assemble_out_bytes = 0;
  for (uint32_t i = 0; i < take; ++i) {
    struct chunk_plan* c = &slot->chunk_plans[base + i];
    wave->decomp_in_bytes += c->compressed_nbytes;
    wave->decomp_out_bytes += c->decompressed_nbytes;
  }
  wave->state = WAVE_IO;
  slot->n_chunks_dispatched += take;
  wp->stats->waves_emitted++;
  wp->stats->chunks_dispatched += take;
  return DAMACY_OK;
}

enum damacy_status
wave_pool_advance(struct wave_pool* wp)
{
  for (int w = 0; w < 2; ++w) {
    struct damacy_wave* wave = &wp->waves[w];
    switch (wave->state) {
      case WAVE_FREE:
        break;
      case WAVE_IO:
        if (store_event_query(wp->store, wave->io_event)) {
          wave->io_t_end_ns = monotonic_ns();
          enum damacy_status s = kick_h2d(wp, wave);
          if (s != DAMACY_OK)
            return s;
        }
        break;
      case WAVE_H2D: {
        CUresult qe = cuEventQuery(wave->ev.h2d_end);
        if (qe == CUDA_SUCCESS) {
          // Both kicks enqueue decode + status-reduce against the
          // shared decoder scratch (d_temp, d_statuses,
          // d_uncompressed_actual_sizes) on stream_decode. Safety
          // relies on stream FIFO: wave A's decode + status-reduce
          // fully retire before wave B's decode launches. Reordering
          // this loop would break the invariant.
          enum damacy_status s = kick_compute(wp, wave);
          if (s != DAMACY_OK)
            return s;
        } else if (qe != CUDA_ERROR_NOT_READY) {
          *wp->failed_status = DAMACY_CUDA;
          return DAMACY_CUDA;
        }
      } break;
      case WAVE_ASSEMBLE: {
        CUresult qe = cuEventQuery(wave->ev.asm_end);
        if (qe == CUDA_SUCCESS) {
          finalize_wave(wp, wave);
        } else if (qe != CUDA_ERROR_NOT_READY) {
          *wp->failed_status = DAMACY_CUDA;
          return DAMACY_CUDA;
        }
      } break;
    }
  }
  return DAMACY_OK;
}
