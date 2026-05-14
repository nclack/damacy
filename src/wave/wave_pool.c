#include "wave_pool.h"

#include "batch_pool/batch_pool.h"
#include "damacy_config.h"
#include "damacy_stats.h"
#include "decoder/status_reduce.h"
#include "fanout.h"
#include "gpu_budget/gpu_budget.h"
#include "log/log.h"
#include "nvtx/nvtx.h"
#include "planner/planner.h"
#include "store/store.h"
#include "util/cuda_check.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "wave_budget.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Wave-index helper for NVTX range labels.
static inline ptrdiff_t
wave_index_of(const struct wave_pool* wp, const struct damacy_wave* wave)
{
  return wave - wp->waves;
}

int
wave_pool_init(struct wave_pool* wp,
               struct damacy_batch_pool* pool,
               struct store* store,
               struct threadpool* compute_pool,
               struct damacy_stats* stats,
               enum damacy_dtype dtype,
               uint8_t host_buffer_waves,
               uint64_t host_slab_per_wave,
               uint64_t dev_decompressed_per_wave,
               uint64_t max_chunk_uncompressed_bytes,
               struct gpu_budget* budget)
{
  memset(wp, 0, sizeof(*wp));
  wp->pool = pool;
  wp->store = store;
  wp->compute_pool = compute_pool;
  wp->stats = stats;
  wp->dtype = dtype;
  wp->budget = budget;
  wp->n_slots = host_buffer_waves;

  // NON_BLOCKING so we don't serialize against the legacy default stream.
  CU(Fail, cuStreamCreate(&wp->stream_h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&wp->stream_decode, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&wp->stream_post, CU_STREAM_NON_BLOCKING));
  for (size_t i = 0; i < countof(wp->decode_done_ring); ++i)
    CU(Fail, cuEventCreate(&wp->decode_done_ring[i], CU_EVENT_DEFAULT));
  damacy_nvtx_stream_name(wp->stream_h2d, "damacy:h2d");
  damacy_nvtx_stream_name(wp->stream_decode, "damacy:decode");
  damacy_nvtx_stream_name(wp->stream_post, "damacy:post");

  const uint64_t host_per_wave = host_slab_per_wave;
  const uint64_t dev_per_wave = dev_decompressed_per_wave;
  wp->dev_per_wave = dev_per_wave;
  wp->max_chunk_uncompressed_bytes = max_chunk_uncompressed_bytes;

  // Initial floor for the shared decoder; decoder_scratch_grow bumps
  // it lazily when a wave's substream count exceeds the current cap.
  // Per-wave fanout SOAs grow independently via fanout_grow.
  size_t zsubs = 0, zstd_per = 0, total_uncompressed = 0;
  decoder_initial_caps(dev_per_wave,
                       max_chunk_uncompressed_bytes,
                       &zsubs,
                       &zstd_per,
                       &total_uncompressed);
  wp->zstd_decoder = decoder_zstd_create(zsubs, zstd_per, total_uncompressed);
  CHECK(Fail, wp->zstd_decoder);

  for (uint8_t s = 0; s < host_buffer_waves; ++s)
    if (slot_init(&wp->slots[s], host_per_wave) != 0)
      goto Fail;
  for (int w = 0; w < DAMACY_N_WAVES; ++w) {
    if (wave_init(&wp->waves[w], host_per_wave, dev_per_wave) != 0)
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
                                &wp->stream_h2d };
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
  for (uint8_t s = 0; s < wp->n_slots; ++s)
    slot_destroy(&wp->slots[s], cuda_skip);
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
  wp->store = NULL;
  wp->compute_pool = NULL;
  wp->stats = NULL;
}

int
find_free_wave(const struct wave_pool* wp)
{
  for (int w = 0; w < DAMACY_N_WAVES; ++w)
    if (wp->waves[w].state == WAVE_FREE)
      return w;
  return -1;
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
  return host_slab_any_in_flight(wp->slots, wp->n_slots);
}

int
any_slot_free(const struct wave_pool* wp)
{
  return host_slab_any_free(wp->slots, wp->n_slots);
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
  damacy_nvtx_range_pushf("kick_h2d/w%td", wave_index_of(wp, wave));
  enum damacy_status rs;

  CU(CudaFail, cuEventRecord(wave->ev.h2d_start, wp->stream_h2d));
  damacy_nvtx_range_push("bulk_h2d");
  CU(BulkCudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->dev_compressed),
                       wave->host_slab,
                       wave->host_used_bytes,
                       wp->stream_h2d));
  // Record bulk_h2d_end before queueing fanout/op H2Ds so stats.h2d
  // measures just the slab copy.
  CU(BulkCudaFail, cuEventRecord(wave->ev.bulk_h2d_end, wp->stream_h2d));
  damacy_nvtx_range_pop(); // bulk_h2d

  build_blosc1_host_chunks(wp, wave);

  // Tight upper bound on zstd substream count for THIS wave — every
  // chunk could be blosc(zstd) with the maximum number of blosc-blocks.
  // need_zsubs <= DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE by the
  // damacy_limits static_assert (peel caps n_chunks). The fanout grow
  // is per-wave: the OTHER wave's SOA is a separate allocation and
  // untouched here, so this can run safely while the other wave is in
  // WAVE_H2D / WAVE_ASSEMBLE. The decoder-scratch grow synchronizes
  // stream_decode first.
  const size_t need_zsubs =
    (size_t)wave->n_chunks * (size_t)DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK;
  enum damacy_status gs = fanout_grow(&wave->h_zstd_fan,
                                      &wave->zstd_fan,
                                      &wave->fanout_cap,
                                      need_zsubs,
                                      wp->budget);
  if (gs == DAMACY_OK)
    gs = decoder_scratch_grow(wp->zstd_decoder,
                              wp->stream_decode,
                              wp->dev_per_wave,
                              wp->max_chunk_uncompressed_bytes,
                              wp->budget,
                              need_zsubs);
  if (gs != DAMACY_OK) {
    record_io_metric(wp, wave);
    cuEventRecord(wave->ev.h2d_end, wp->stream_h2d);
    rs = gs;
    goto Done;
  }

  damacy_nvtx_range_push("host_parse");
  uint64_t parse_t0 = monotonic_ns();
  int rc = blosc1_host_parse(&(struct blosc1_host_parse_args){
    .pool = wp->compute_pool,
    .chunks = wave->h_chunks,
    .n_chunks = wave->n_chunks,
    .scratch = wave->scratch,
    .zstd = wave->h_zstd_fan,
    .memcpy_ops = wave->h_memcpy_ops,
    .assemble_chunks = wave->h_assemble_chunks,
    .out_totals = wave->h_blosc1_totals,
  });
  wave->parse_ms = (float)((monotonic_ns() - parse_t0) / 1.0e6);
  damacy_nvtx_range_pop(); // host_parse
  if (rc) {
    // Drain the IO metric before bailing — finalize_wave won't run for
    // this wave and we don't want failed runs to silently undercount IO.
    record_io_metric(wp, wave);
    log_blosc1_parse_errors(wp, wave);
    cuEventRecord(wave->ev.h2d_end, wp->stream_h2d);
    rs = DAMACY_DECODE;
    goto Done;
  }

  damacy_nvtx_range_push("fanout_h2d");
  const struct blosc1_totals* tot = wave->h_blosc1_totals;
  if (tot->n_zstd > 0 && fanout_upload(wp->stream_h2d,
                                       &wave->zstd_fan,
                                       &wave->h_zstd_fan,
                                       (size_t)tot->n_zstd) != DAMACY_OK) {
    damacy_nvtx_range_pop(); // fanout_h2d
    goto CudaFail;
  }
  if (tot->n_memcpy > 0)
    CU(FanoutCudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_memcpy_ops),
                         wave->h_memcpy_ops,
                         (size_t)tot->n_memcpy * sizeof(struct gpu_memcpy_op),
                         wp->stream_h2d));

  // Assemble metadata: pinned host buffer was built in peel_wave; pushing
  // the H2D here folds it into h2d_end's gating so stream_decode sees it
  // ready before assemble_launch (which itself queues after the decode
  // kernels). Overlaps with the bulk slab H2D + nvcomp decode on the
  // device.
  CU(FanoutCudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_assemble_chunks),
                       wave->h_assemble_chunks,
                       (size_t)wave->n_chunks * sizeof(struct assemble_chunk),
                       wp->stream_h2d));

  // Zero so status_reduce's atomicAdds land in a clean n_codec_errors.
  CU(FanoutCudaFail,
     cuMemsetD8Async(CUDPTR(wave->d_blosc1_totals),
                     0,
                     sizeof(struct blosc1_totals),
                     wp->stream_h2d));

  CU(FanoutCudaFail, cuEventRecord(wave->ev.h2d_end, wp->stream_h2d));
  damacy_nvtx_range_pop(); // fanout_h2d
  wave->state = WAVE_H2D;
  rs = DAMACY_OK;
  goto Done;

FanoutCudaFail:
  // h2d_end intentionally not recorded: kick_h2d returns DAMACY_CUDA so
  // wave_pool_advance bails before the wave reaches WAVE_ASSEMBLE,
  // finalize_wave never runs for this wave, and drain_wave_metrics
  // never reads h2d_end. Recording it here would be dead work.
  damacy_nvtx_range_pop(); // fanout_h2d
  goto CudaFail;
BulkCudaFail:
  damacy_nvtx_range_pop(); // bulk_h2d
CudaFail:
  rs = DAMACY_CUDA;
Done:
  damacy_nvtx_range_pop(); // kick_h2d
  return rs;
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

// Nvcomp + status_reduce on stream_decode. status_reduce stays here
// (shared d_statuses across waves; stream_decode FIFO orders it).
static enum damacy_status
kick_decode(struct wave_pool* wp,
            struct damacy_wave* wave,
            const struct blosc1_totals* tot)
{
  CUstream s = wp->stream_decode;
  uint32_t* d_err = &wave->d_blosc1_totals->n_codec_errors;
  enum damacy_status rs;
  damacy_nvtx_range_pushf("kick_decode/w%td", wave_index_of(wp, wave));
  // Borrow the most-recently-anchored slot for our gap measurement, then
  // claim the next slot for our own anchor (recorded after decode_done).
  // First kick's prev points at a never-recorded slot — cuEventElapsedTime
  // returns NOT_READY and drain skips.
  size_t prev_idx = wp->decode_done_ring_idx;
  size_t this_idx = (prev_idx + 1) % countof(wp->decode_done_ring);
  wave->prev_decode_anchor = wp->decode_done_ring[prev_idx];
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
  CU(CudaFail, cuEventRecord(wave->ev.decode_done, s));
  CU(CudaFail, cuEventRecord(wp->decode_done_ring[this_idx], s));
  wp->decode_done_ring_idx = this_idx;
  rs = DAMACY_OK;
  goto Done;
DecodeFail:
  rs = DAMACY_DECODE;
  goto Done;
CudaFail:
  rs = DAMACY_CUDA;
Done:
  damacy_nvtx_range_pop();
  return rs;
}

// Post-decode + 4B D2H + assemble on stream_post, gated on
// ev.decode_done. Lets wave N+1's decode on stream_decode overlap
// wave N's assemble.
static enum damacy_status
kick_assemble(struct wave_pool* wp,
              struct damacy_wave* wave,
              const struct blosc1_totals* tot)
{
  CUstream s = wp->stream_post;
  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];
  enum damacy_status rs;
  damacy_nvtx_range_pushf("kick_assemble/w%td", wave_index_of(wp, wave));

  CU(CudaFail, cuStreamWaitEvent(s, wave->ev.decode_done, 0));

  if (tot->n_memcpy > 0 &&
      decoder_memcpy_launch(s, wave->d_memcpy_ops, tot->n_memcpy))
    goto DecodeFail;
  // (Bit)unshuffle is now folded into assemble_kernel via per-chunk
  // shuffle_mode in assemble_chunk.
  // Narrowed to the 4-byte n_codec_errors so the host parse's count
  // fields in h_blosc1_totals stay intact for drain_wave_metrics.
  CU(CudaFail,
     cuMemcpyDtoHAsync(&wave->h_blosc1_totals->n_codec_errors,
                       CUDPTR(&wave->d_blosc1_totals->n_codec_errors),
                       sizeof(uint32_t),
                       s));
  CU(CudaFail, cuEventRecord(wave->ev.decomp_end, s));

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
                      wp->dtype))
    goto CudaFail;
  CU(CudaFail, cuEventRecord(wave->ev.asm_end, s));
  rs = DAMACY_OK;
  goto Done;
DecodeFail:
  rs = DAMACY_DECODE;
  goto Done;
CudaFail:
  rs = DAMACY_CUDA;
Done:
  damacy_nvtx_range_pop();
  return rs;
}

// Wait on h2d_end, kick decode on stream_decode, kick post + assemble on
// stream_post.
static enum damacy_status
kick_compute(struct wave_pool* wp, struct damacy_wave* wave)
{
  CU(CudaFail, cuStreamWaitEvent(wp->stream_decode, wave->ev.h2d_end, 0));

  const struct blosc1_totals tot = *wave->h_blosc1_totals;
  enum damacy_status st = kick_decode(wp, wave, &tot);
  if (st != DAMACY_OK)
    return st;
  st = kick_assemble(wp, wave, &tot);
  if (st != DAMACY_OK)
    return st;

  wave->state = WAVE_ASSEMBLE;
  return DAMACY_OK;
CudaFail:
  return DAMACY_CUDA;
}

// All wave events have fired; pull elapsed times into stats.
static void
drain_wave_metrics(const struct wave_pool* wp, struct damacy_wave* wave)
{
  struct damacy_stats* st = wp->stats;
  record_io_metric(wp, wave);

  float ms = 0.f;
  if (cuEventElapsedTime(&ms, wave->ev.h2d_start, wave->ev.bulk_h2d_end) ==
      CUDA_SUCCESS)
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
  metric_record(&st->decompress_parse, wave->parse_ms, 0, 0);
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
  wave->state = WAVE_FREE;
  wave->n_chunks = 0;
  damacy_nvtx_range_pop();
  return out;
}

// Pack chunks from `batch_slot`'s remaining work into a free
// host_slab_slot. Submits IO and transitions the slot to SLOT_IO.
// No-op if no slot is free or the batch has nothing left.
enum damacy_status
wave_pool_peel(struct wave_pool* wp, uint16_t batch_slot_idx)
{
  struct damacy_batch_slot* batch = &wp->pool->slots[batch_slot_idx];
  uint32_t base = batch->n_chunks_dispatched;
  uint32_t remaining = batch->n_chunks - base;
  if (remaining == 0)
    return DAMACY_OK;
  int slot_idx = host_slab_find_free(wp->slots, wp->n_slots);
  if (slot_idx < 0)
    return DAMACY_OK;
  struct host_slab_slot* hs = &wp->slots[slot_idx];
  damacy_nvtx_range_pushf("peel/slot%d", slot_idx);

  uint64_t host_cursor = 0;
  uint64_t dev_cursor = 0;
  uint32_t take = 0;
  // Cap dev_cursor against any wave's dev_decompressed_cap — all waves
  // share the same size by resolver construction. Read the first wave's
  // cap as the runtime value.
  const uint64_t dev_cap = wp->waves[0].dev_decompressed_cap;
  for (; take < remaining && take < DAMACY_MAX_CHUNKS_PER_WAVE; ++take) {
    struct read_op* r = &batch->read_ops[base + take];
    struct chunk_plan* c = &batch->chunk_plans[base + take];
    if (host_cursor + r->nbytes > hs->cap)
      break;
    if (dev_cursor + c->decompressed_nbytes > dev_cap)
      break;
    r->dst_buf_offset = host_cursor;
    c->dev_decompressed_offset = dev_cursor;
    host_cursor += r->nbytes;
    dev_cursor += c->decompressed_nbytes;
  }
  if (take == 0) {
    // Single chunk doesn't fit. Per-wave caps too tight for this workload;
    // surface it loudly rather than livelocking. Slot stays FREE.
    log_error("wave: chunk too large for slot "
              "(slot_cap=%llu dev_cap=%llu)",
              (unsigned long long)hs->cap,
              (unsigned long long)dev_cap);
    damacy_nvtx_range_pop();
    return DAMACY_OOM;
  }

  for (uint32_t i = 0; i < take; ++i) {
    struct read_op* r = &batch->read_ops[base + i];
    hs->store_reads[i] = (struct store_read){
      .key = r->shard_path,
      .dst = (uint8_t*)hs->buf + r->dst_buf_offset,
      .offset = r->file_offset,
      .len = r->nbytes,
    };
  }
  hs->io_t_start_ns = monotonic_ns();
  hs->io_event = store_read_submit(wp->store, hs->store_reads, take);
  if (hs->io_event.seq == 0) {
    damacy_nvtx_range_pop();
    return DAMACY_IO;
  }

  hs->batch_pool_slot = batch_slot_idx;
  hs->batch_chunk_offset = base;
  hs->n_chunks = take;
  hs->used_bytes = host_cursor;
  hs->io_bytes = host_cursor;
  hs->state = SLOT_IO;
  batch->n_chunks_dispatched += take;
  wp->stats->waves_emitted++;
  wp->stats->chunks_dispatched += take;
  damacy_nvtx_range_pop();
  return DAMACY_OK;
}

// Bind a SLOT_READY slot to a WAVE_FREE wave: copies slot fields onto
// the wave, builds the assemble metadata, kicks H2D. The slot
// transitions SLOT_READY → SLOT_BUSY; the wave WAVE_FREE → WAVE_H2D.
// On kick_h2d failure the slot is released back to SLOT_FREE so the
// scheduler doesn't deadlock on a stuck-busy slot.
static enum damacy_status
bind_slot_to_wave(struct wave_pool* wp, struct damacy_wave* wave, int slot_idx)
{
  struct host_slab_slot* hs = &wp->slots[slot_idx];
  damacy_nvtx_range_pushf(
    "bind/w%td/slot%d", wave_index_of(wp, wave), slot_idx);
  metric_record(&wp->stats->bind_wait,
                (float)((monotonic_ns() - hs->io_t_end_ns) / 1.0e6),
                0,
                0);
  wave->bound_slot = (int8_t)slot_idx;
  wave->host_slab = hs->buf;
  wave->batch_pool_slot = hs->batch_pool_slot;
  wave->batch_chunk_offset = hs->batch_chunk_offset;
  wave->n_chunks = hs->n_chunks;
  wave->host_used_bytes = hs->used_bytes;
  wave->io_bytes = hs->io_bytes;
  wave->io_t_start_ns = hs->io_t_start_ns;
  wave->io_t_end_ns = hs->io_t_end_ns;
  wave->decomp_in_bytes = 0;
  wave->decomp_out_bytes = 0;
  wave->assemble_out_bytes = 0;

  struct damacy_batch_slot* batch = &wp->pool->slots[hs->batch_pool_slot];
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    struct chunk_plan* c = &batch->chunk_plans[wave->batch_chunk_offset + i];
    wave->decomp_in_bytes += c->compressed_nbytes;
    wave->decomp_out_bytes += c->decompressed_nbytes;
  }

  hs->state = SLOT_BUSY;
  build_assemble_meta(wp, wave);
  enum damacy_status s = kick_h2d(wp, wave);
  if (s != DAMACY_OK) {
    // kick_h2d failed before recording bulk_h2d_end (or after, but
    // either way the slot is no longer in flight on the GPU since the
    // wave never reaches WAVE_H2D's poll). Release the slot.
    slot_release(hs);
    wave->bound_slot = -1;
    wave->host_slab = NULL;
  }
  damacy_nvtx_range_pop();
  return s;
}

enum damacy_status
wave_pool_advance(struct wave_pool* wp)
{
  // Pass 1: SLOT_IO → SLOT_READY when IO completes.
  for (uint8_t s = 0; s < wp->n_slots; ++s) {
    struct host_slab_slot* hs = &wp->slots[s];
    if (hs->state == SLOT_IO && store_event_query(wp->store, hs->io_event)) {
      hs->io_t_end_ns = monotonic_ns();
      hs->state = SLOT_READY;
    }
  }

  // Pass 2: bind SLOT_READY to WAVE_FREE waves.
  for (int w = 0; w < DAMACY_N_WAVES; ++w) {
    struct damacy_wave* wave = &wp->waves[w];
    if (wave->state != WAVE_FREE)
      continue;
    int rs = host_slab_find_ready(wp->slots, wp->n_slots);
    if (rs < 0)
      break;
    enum damacy_status s = bind_slot_to_wave(wp, wave, rs);
    if (s != DAMACY_OK)
      return s;
  }

  // Pass 3: drive wave state machine. Release slots on bulk_h2d_end
  // (independent of h2d_end) so peel can refill them as early as
  // possible.
  for (int w = 0; w < DAMACY_N_WAVES; ++w) {
    struct damacy_wave* wave = &wp->waves[w];
    switch (wave->state) {
      case WAVE_FREE:
        break;
      case WAVE_H2D: {
        if (wave->bound_slot >= 0) {
          CUresult qb = cuEventQuery(wave->ev.bulk_h2d_end);
          if (qb == CUDA_SUCCESS) {
            slot_release(&wp->slots[wave->bound_slot]);
            wave->bound_slot = -1;
            wave->host_slab = NULL;
          } else if (qb != CUDA_ERROR_NOT_READY) {
            return DAMACY_CUDA;
          }
        }
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
          return DAMACY_CUDA;
        }
      } break;
      case WAVE_ASSEMBLE: {
        CUresult qe = cuEventQuery(wave->ev.asm_end);
        if (qe == CUDA_SUCCESS) {
          struct wave_outcome o = finalize_wave(wp, wave);
          batch_slot_consume_chunks(&wp->pool->slots[o.batch_pool_slot],
                                    o.n_chunks_consumed);
          if (o.status != DAMACY_OK)
            return o.status;
        } else if (qe != CUDA_ERROR_NOT_READY) {
          return DAMACY_CUDA;
        }
      } break;
    }
  }
  return DAMACY_OK;
}
