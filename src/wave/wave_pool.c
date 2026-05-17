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
#include "store/store.h"
#include "store/store_fs_gds.h" // store_fs_gds_set_stream (GDS-only)
#include "util/cuda_check.h"
#include "util/prelude.h"
#include "wave_budget.h"
#include "zarr/zarr_metadata.h" // CODEC_*

#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Defined in wave_budget.c — kept private to wave_pool because the
// "grow" surface is an orchestration concern, while the math it
// reuses (wave_decoder_caps, predict_decoder_scratch_bytes) lives
// alongside the pure predictor.
enum damacy_status
decoder_scratch_grow(struct decoder_zstd* decoder,
                     CUstream stream_decode,
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

int
wave_pool_init(struct wave_pool* wp,
               struct damacy_batch_pool* pool,
               struct store* store,
               struct damacy_stats* stats,
               enum damacy_dtype dtype,
               uint8_t host_buffer_waves,
               uint64_t host_slab_per_wave,
               uint64_t dev_decompressed_per_wave,
               uint64_t max_chunk_uncompressed_bytes,
               int enable_gds,
               int bypass_decode,
               struct gpu_budget* budget)
{
  memset(wp, 0, sizeof(*wp));
  wp->pool = pool;
  wp->store = store;
  wp->stats = stats;
  wp->dtype = dtype;
  wp->budget = budget;
  wp->n_slots = host_buffer_waves;
  wp->use_gds = (uint8_t)(enable_gds != 0);
  wp->bypass_decode = (uint8_t)(bypass_decode != 0);

  // NON_BLOCKING so we don't serialize against the legacy default stream.
  CU(Fail, cuStreamCreate(&wp->stream_h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&wp->stream_decode, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&wp->stream_post, CU_STREAM_NON_BLOCKING));
  // GDS path: cuFileReadAsync rides on stream_h2d. Set this before the
  // first peel so submit_dev has a valid stream to schedule against.
  // No-op (and harmless) when the store isn't a GDS-enabled store_fs.
  if (wp->use_gds)
    store_fs_gds_set_stream(store, wp->stream_h2d);
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

  // GDS-only slots skip the pinned-host alloc; non-GDS slots skip the
  // device alloc. Both code paths still need the store_reads array.
  const uint64_t slot_host_cap = wp->use_gds ? 0ull : host_per_wave;
  const uint64_t slot_dev_cap = wp->use_gds ? host_per_wave : 0ull;
  for (uint8_t s = 0; s < host_buffer_waves; ++s)
    if (slot_init(&wp->slots[s], slot_host_cap, slot_dev_cap) != 0)
      goto Fail;
  for (int w = 0; w < DAMACY_N_WAVES; ++w) {
    if (wave_init(&wp->waves[w], host_per_wave, dev_per_wave, wp->use_gds) != 0)
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
  return host_slab_any_in_flight(wp->slots, wp->n_slots);
}

int
any_slot_free(const struct wave_pool* wp)
{
  return host_slab_any_free(wp->slots, wp->n_slots);
}

// --- peel / advance -------------------------------------------------------

// Precondition for the GPU parse: every BLOSC_ZSTD chunk in the wave
// must have a probed per-sample layout with dont_split set. Kernel B
// reads layout.blocksize / layout.nblocks per chunk; without a probed
// layout we don't know nblocks ahead of launch, and dont_split=0 lets
// blosc emit multiple substreams per block (out of scope for the
// per-block fan-out kernel). dont_split is always 1 for blosc1-zstd in
// practice; layout_probed is set on the first non-fill emit for any
// sample touching the array. A missing probe is a configuration/probe
// gap (all preceding waves were fill, so no non-fill emit ever fed the
// cache) — surfaced as DAMACY_INVAL by the caller, distinct from
// DAMACY_DECODE which signals malformed compressed bytes.
static int
wave_chunks_eligible(const struct wave_pool* wp, const struct damacy_wave* wave)
{
  // bypass_decode: parse/assemble flip every chunk to is_fill, so the
  // BLOSC_ZSTD layout-probe preconditions don't apply.
  if (wp->bypass_decode)
    return 1;
  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    const struct chunk_plan* c =
      &slot->chunk_plans[wave->batch_chunk_offset + i];
    if (c->is_fill)
      continue;
    if (c->codec_id != (uint8_t)CODEC_BLOSC_ZSTD)
      continue;
    const struct sample_plan* sp = &slot->sample_plans[c->sample_idx_in_batch];
    if (!sp->layout_probed)
      return 0;
    if (!sp->layout.dont_split)
      return 0;
  }
  return 1;
}

// Build the per-wave parse inputs for the GPU path:
//  - h_parse_chunks: arena-offset view of every chunk (kernel-resident).
//  - h_assemble_chunks shuffle_* trio: defaulted to NONE for every chunk
//    (Kernel A overrides for non-memcpyed BLOSC_ZSTD chunks).
//  - h_memcpy_ops / h_zstd_fan prefixes: pre-emitted ops for CODEC_NONE
//    and CODEC_ZSTD whole-chunk chunks. The kernel atomic-adds slots
//    starting after these prefixes.
//  - h_blosc_chunk_indices: list of wave-local indices of BLOSC_ZSTD
//    chunks (input to Kernel A).
//  - h_block_chunk_map: packed (chunk_idx<<16)|block_local_idx over
//    every blosc block in every BLOSC_ZSTD chunk (input to Kernel B).
//
// Counts written to the wave: n_host_memcpy, n_host_zstd,
// n_blosc_zstd_chunks, n_blosc_zstd_blocks.
static void
build_gpu_parse_chunks(const struct wave_pool* wp, struct damacy_wave* wave)
{
  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];
  uint8_t* d_comp_base = (uint8_t*)wave->dev_compressed;
  uint8_t* d_decomp_base = (uint8_t*)wave->dev_decompressed;
  uint32_t n_host_memcpy = 0;
  uint32_t n_host_zstd = 0;
  uint32_t n_blosc_chunks = 0;
  uint32_t n_blosc_blocks = 0;

  // bypass_decode flips every chunk to fill; kick_decode still records
  // its events on stream_decode, but decode + decode_gap metrics are
  // meaningless under bypass.
  uint8_t effective_fill_force = wp->bypass_decode;
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    struct chunk_plan* c = &slot->chunk_plans[wave->batch_chunk_offset + i];
    struct gpu_parse_chunk* gc = &wave->h_parse_chunks[i];
    uint32_t comp_off =
      (uint32_t)(c->host_buf_offset + (uint64_t)c->offset_in_read);
    gc->compressed_offset = comp_off;
    gc->compressed_nbytes = c->compressed_nbytes;
    gc->decompressed_offset = c->dev_decompressed_offset;
    gc->decompressed_nbytes = c->decompressed_nbytes;
    gc->sample_idx_in_batch = c->sample_idx_in_batch;
    gc->codec_id = c->codec_id;
    gc->is_fill = c->is_fill | effective_fill_force;

    struct assemble_chunk* a = &wave->h_assemble_chunks[i];
    a->shuffle_mode = (uint8_t)ASSEMBLE_SHUFFLE_NONE;
    a->shuffle_typesize = 0;
    a->shuffle_blocksize = 0;

    if (c->is_fill || c->codec_id == (uint8_t)CODEC_FILL ||
        effective_fill_force)
      continue;

    if (c->codec_id == (uint8_t)CODEC_NONE) {
      struct gpu_memcpy_op* op = &wave->h_memcpy_ops[n_host_memcpy++];
      op->d_src = d_comp_base + comp_off;
      op->d_dst = d_decomp_base + c->dev_decompressed_offset;
      op->nbytes = c->decompressed_nbytes;
      continue;
    }

    if (c->codec_id == (uint8_t)CODEC_ZSTD) {
      wave->h_zstd_fan.comp_ptrs[n_host_zstd] = d_comp_base + comp_off;
      wave->h_zstd_fan.comp_sizes[n_host_zstd] = c->compressed_nbytes;
      wave->h_zstd_fan.decomp_ptrs[n_host_zstd] =
        d_decomp_base + c->dev_decompressed_offset;
      wave->h_zstd_fan.decomp_buf_sizes[n_host_zstd] = c->decompressed_nbytes;
      n_host_zstd++;
      continue;
    }

    // CODEC_BLOSC_ZSTD: defer to Kernel A (memcpyed flag) + Kernel B
    // (per-block bstart/cb walk). nblocks is known per-sample from the
    // probed layout; wave_can_use_gpu_parse asserted layout_probed.
    const struct sample_plan* sp = &slot->sample_plans[c->sample_idx_in_batch];
    uint32_t nblocks = sp->layout.nblocks;
    wave->h_blosc_chunk_indices[n_blosc_chunks++] = i;
    for (uint32_t b = 0; b < nblocks; ++b)
      // guarded by static_assert in damacy_limits.h
      wave->h_block_chunk_map[n_blosc_blocks++] = (i << 16) | b;
  }

  wave->n_host_memcpy = n_host_memcpy;
  wave->n_host_zstd = n_host_zstd;
  wave->n_blosc_zstd_chunks = n_blosc_chunks;
  wave->n_blosc_zstd_blocks = n_blosc_blocks;
}

// IO already retired into io_t_end_ns by the caller. Push io timing into
// stats.io so failure paths don't bias the rolling totals.
static void
record_io_metric(const struct wave_pool* wp, const struct damacy_wave* wave)
{
  float io_ms = (float)((wave->io_t_end_ns - wave->io_t_start_ns) / 1.0e6);
  metric_record(&wp->stats->io, io_ms, wave->io_bytes, wave->io_bytes);
}

// Phase 1: record h2d_start, queue the slab memcpy, record bulk_h2d_end.
// On the GDS path the cuFileReadAsync calls submitted in peel were
// queued on this same stream_h2d, so the memcpy is skipped — stream
// FIFO orders the parse kernel after the reads. bulk_h2d_end still
// records on stream_h2d, now measuring the read time itself rather
// than a separate H2D copy.
static enum damacy_status
submit_bulk_h2d(struct wave_pool* wp, struct damacy_wave* wave)
{
  CU(CudaFail, cuEventRecord(wave->ev.h2d_start, wp->stream_h2d));
  damacy_nvtx_range_push("bulk_h2d");
  if (!wp->use_gds) {
    CU(BulkCudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->dev_compressed),
                         wave->host_slab,
                         wave->host_used_bytes,
                         wp->stream_h2d));
  }
  // Record bulk_h2d_end before queueing fanout/op H2Ds so stats.h2d
  // measures just the slab copy.
  CU(BulkCudaFail, cuEventRecord(wave->ev.bulk_h2d_end, wp->stream_h2d));
  damacy_nvtx_range_pop(); // bulk_h2d
  return DAMACY_OK;
BulkCudaFail:
  damacy_nvtx_range_pop(); // bulk_h2d
CudaFail:
  return DAMACY_CUDA;
}

// Per-chunk substream upper bound. Uses the probed blosc1 layout when
// available; falls back to MAX_BLOCKS_PER_CHUNK for unprobed blosc-zstd,
// 1 for plain zstd, and 0 for fill / raw-memcpy chunks (those don't
// consume zstd substreams).
static inline uint32_t
chunk_zsubs_upper_bound(const struct chunk_plan* c,
                        const struct sample_plan* sp)
{
  if (c->is_fill || c->codec_id == (uint8_t)CODEC_FILL ||
      c->codec_id == (uint8_t)CODEC_NONE)
    return 0;
  if (c->codec_id == (uint8_t)CODEC_BLOSC_ZSTD) {
    if (sp->layout_probed)
      return sp->layout.nblocks;
    return DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK;
  }
  // CODEC_ZSTD: one substream per chunk.
  return 1;
}

// Phase 2: grow per-wave fanout SOA + shared decoder scratch to cover
// this wave's tight substream upper bound, computed from probed
// chunk_layouts when available. need_zsubs <=
// DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE by the damacy_limits static_assert
// (peel caps n_chunks). Fanout grow is per-wave: the OTHER wave's SOA is
// a separate allocation and untouched here, so this can run safely
// while the other wave is in WAVE_H2D / WAVE_ASSEMBLE.
// decoder_scratch_grow synchronizes stream_decode first.
static enum damacy_status
prepare_decode_caps(struct wave_pool* wp, struct damacy_wave* wave)
{
  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];
  size_t need_zsubs = 0;
  // bypass_decode flips every chunk to is_fill at build time, so the
  // wave needs zero zstd substream scratch.
  if (!wp->bypass_decode) {
    for (uint32_t i = 0; i < wave->n_chunks; ++i) {
      const struct chunk_plan* c =
        &slot->chunk_plans[wave->batch_chunk_offset + i];
      const struct sample_plan* sp =
        &slot->sample_plans[c->sample_idx_in_batch];
      need_zsubs += chunk_zsubs_upper_bound(c, sp);
    }
  }
  enum damacy_status gs = fanout_grow(&wave->h_zstd_fan,
                                      &wave->zstd_fan,
                                      &wave->fanout_cap,
                                      need_zsubs,
                                      wp->budget);
  if (gs != DAMACY_OK)
    return gs;
  return decoder_scratch_grow(wp->zstd_decoder,
                              wp->stream_decode,
                              wp->dev_per_wave,
                              wp->max_chunk_uncompressed_bytes,
                              wp->budget,
                              need_zsubs);
}

// GPU parse: upload inputs (parse_chunks, assemble_chunks,
// per-codec indirection tables, host-emitted SOA prefixes), seed device
// counters from the host-emitted counts (the kernels atomicAdd above),
// clear the memcpyed bitset + parse_err + totals, launch the two parse
// kernels, D2H counters, record h2d_end. The kernels write into
// d_memcpy_ops / zstd_fan / d_assemble_chunks in place; kick_compute
// reads h_parse_counters after h2d_end fires.
static enum damacy_status
gpu_parse(struct wave_pool* wp, struct damacy_wave* wave)
{
  damacy_nvtx_range_push("gpu_parse");
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_parse_chunks),
                       wave->h_parse_chunks,
                       (size_t)wave->n_chunks * sizeof(struct gpu_parse_chunk),
                       wp->stream_h2d));
  // assemble_chunks: build_assemble_meta filled geometry; build_gpu_parse
  // _chunks defaulted shuffle_* to NONE. Kernel A overrides for non-
  // memcpyed BLOSC_ZSTD chunks.
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_assemble_chunks),
                       wave->h_assemble_chunks,
                       (size_t)wave->n_chunks * sizeof(struct assemble_chunk),
                       wp->stream_h2d));

  if (wave->n_blosc_zstd_chunks > 0)
    CU(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_blosc_chunk_indices),
                         wave->h_blosc_chunk_indices,
                         (size_t)wave->n_blosc_zstd_chunks * sizeof(uint32_t),
                         wp->stream_h2d));
  if (wave->n_blosc_zstd_blocks > 0)
    CU(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_block_chunk_map),
                         wave->h_block_chunk_map,
                         (size_t)wave->n_blosc_zstd_blocks * sizeof(uint32_t),
                         wp->stream_h2d));

  // Host-pre-filled SOA prefixes (FILL/NONE/ZSTD whole-chunk ops). The
  // kernels' atomicAdd seeds skip past these slots.
  if (wave->n_host_memcpy > 0)
    CU(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_memcpy_ops),
                         wave->h_memcpy_ops,
                         (size_t)wave->n_host_memcpy *
                           sizeof(struct gpu_memcpy_op),
                         wp->stream_h2d));
  if (wave->n_host_zstd > 0 && fanout_upload(wp->stream_h2d,
                                             &wave->zstd_fan,
                                             &wave->h_zstd_fan,
                                             wave->n_host_zstd) != DAMACY_OK)
    goto CudaFail;

  // Seed counters with host-emitted counts so kernel atomicAdds claim
  // slots after the prefix.
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_n_zstd),
                       &wave->n_host_zstd,
                       sizeof(uint32_t),
                       wp->stream_h2d));
  CU(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_n_memcpy),
                       &wave->n_host_memcpy,
                       sizeof(uint32_t),
                       wp->stream_h2d));
  CU(CudaFail,
     cuMemsetD8Async(
       CUDPTR(wave->d_parse_err), 0, sizeof(uint32_t), wp->stream_h2d));
  // Bitset: one bit per wave-local chunk_idx. Kernel A atomic-ors a 1 in
  // for memcpyed chunks; Kernel B reads it and skips those chunks'
  // blocks.
  CU(CudaFail,
     cuMemsetD8Async(CUDPTR(wave->d_is_memcpyed),
                     0,
                     (size_t)((wave->n_chunks + 31u) / 32u) * sizeof(uint32_t),
                     wp->stream_h2d));
  CU(CudaFail,
     cuMemsetD8Async(CUDPTR(wave->d_blosc1_totals),
                     0,
                     sizeof(struct blosc1_totals),
                     wp->stream_h2d));

  struct damacy_batch_slot* slot = &wp->pool->slots[wave->batch_pool_slot];
  struct blosc1_parse_args pargs = {
    .d_compressed = (const uint8_t*)wave->dev_compressed,
    .d_decompressed = (uint8_t*)wave->dev_decompressed,
    .d_chunks = wave->d_parse_chunks,
    .d_sample_plans = (const struct sample_plan*)slot->d_sample_plans,
    .n_sample_plans = slot->n_sample_plans,
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
  if (blosc1_parse_launch(wp->stream_h2d, &pargs))
    goto CudaFail;

  // D2H the 12 bytes (n_zstd, n_memcpy, parse_err). h_parse_counters is
  // pinned; the read in kick_compute sees this value because the
  // wave_pool_advance polls h2d_end (recorded after this D2H) before
  // calling kick_compute.
  CU(CudaFail,
     cuMemcpyDtoHAsync(wave->h_parse_counters,
                       CUDPTR(wave->d_n_zstd),
                       sizeof(uint32_t),
                       wp->stream_h2d));
  CU(CudaFail,
     cuMemcpyDtoHAsync(wave->h_parse_counters + 1,
                       CUDPTR(wave->d_n_memcpy),
                       sizeof(uint32_t),
                       wp->stream_h2d));
  CU(CudaFail,
     cuMemcpyDtoHAsync(wave->h_parse_counters + 2,
                       CUDPTR(wave->d_parse_err),
                       sizeof(uint32_t),
                       wp->stream_h2d));

  CU(CudaFail, cuEventRecord(wave->ev.h2d_end, wp->stream_h2d));
  damacy_nvtx_range_pop(); // gpu_parse
  return DAMACY_OK;
CudaFail:
  damacy_nvtx_range_pop(); // gpu_parse
  return DAMACY_CUDA;
}

// Bulk H2D, GPU parse on stream_h2d, then h2d_end. stream_decode gates
// on h2d_end before nvcomp launch.
static enum damacy_status
kick_h2d(struct wave_pool* wp, struct damacy_wave* wave)
{
  damacy_nvtx_range_pushf("kick_h2d/w%td", wave_index_of(wp, wave));
  enum damacy_status s;
  // Set between phases where bulk_h2d_end has been recorded but h2d_end
  // has not. On those failures the cleanup path records h2d_end so the
  // polling state machine doesn't hang on a never-recorded event.
  int needs_end_record = 0;

  s = submit_bulk_h2d(wp, wave);
  if (s != DAMACY_OK)
    goto Error;

  needs_end_record = 1;
  s = prepare_decode_caps(wp, wave);
  if (s != DAMACY_OK)
    goto Error;

  if (!wave_chunks_eligible(wp, wave)) {
    // Probe gap (e.g. all-fill preceding waves never seeded the layout
    // cache) or unsupported dont_split=0 — not garbage in compressed
    // bytes, so DAMACY_INVAL rather than DAMACY_DECODE.
    log_error("blosc1: wave has a BLOSC_ZSTD chunk with unprobed layout "
              "or dont_split=0; cannot parse on device");
    s = DAMACY_INVAL;
    goto Error;
  }
  build_gpu_parse_chunks(wp, wave);
  needs_end_record = 0; // gpu_parse records h2d_end itself
  s = gpu_parse(wp, wave);
  if (s != DAMACY_OK)
    goto Error;

  wave->state = WAVE_H2D;
  damacy_nvtx_range_pop(); // kick_h2d
  return DAMACY_OK;

Error:
  // IO already retired into io_t_end_ns at bind; record it unconditionally
  // so failure paths don't bias the rolling stats.
  record_io_metric(wp, wave);
  if (needs_end_record) {
    // Keep the polling state machine from hanging on a never-recorded
    // h2d_end. submit_bulk_h2d failed before bulk_h2d_end was recorded,
    // so this branch handles only the post-bulk_h2d failures.
    cuEventRecord(wave->ev.h2d_end, wp->stream_h2d);
  }
  damacy_nvtx_range_pop(); // kick_h2d
  return s;
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
    a->is_fill = c->is_fill | wp->bypass_decode;
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

  // h_parse_counters were D2H'd to pinned host on stream_h2d before
  // h2d_end was recorded; cuEventQuery(h2d_end) returning CUDA_SUCCESS
  // (the wave_pool_advance gate that called us) means this is safe to
  // read without an explicit host sync.
  struct blosc1_totals tot = {
    .n_zstd = wave->h_parse_counters[0],
    .n_memcpy = wave->h_parse_counters[1],
    .n_parse_errors = wave->h_parse_counters[2],
  };
  if (tot.n_parse_errors != 0) {
    log_error("blosc1 gpu_parse: first parse err code=%u", tot.n_parse_errors);
    return DAMACY_DECODE;
  }
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

// --- peel: reserve [locked] → submit [unlocked] → commit [locked] -----------
// submit does the async IO submit off the scheduler_lock. The slot
// sits in SLOT_PEELING for the window so nothing else binds it.

struct wave_pool_peel_ticket
wave_pool_peel_reserve(struct wave_pool* wp,
                       uint16_t batch_slot_idx,
                       enum damacy_status* err)
{
  *err = DAMACY_OK;
  struct wave_pool_peel_ticket t = { .slot_idx = -1,
                                     .n_reads = 0,
                                     .consumed = 0 };
  struct damacy_batch_slot* batch = &wp->pool->slots[batch_slot_idx];
  uint32_t base = batch->n_chunks_dispatched;
  if (base >= batch->n_chunks)
    return t;
  int slot_idx = host_slab_find_free(wp->slots, wp->n_slots);
  if (slot_idx < 0)
    return t;
  struct host_slab_slot* hs = &wp->slots[slot_idx];

  uint64_t host_cursor = 0;
  uint64_t dev_cursor = 0;
  uint32_t take = 0;
  uint32_t n_reads = 0;
  uint32_t completed_groups = batch->n_groups_dispatched;
  const uint64_t dev_cap = wp->waves[0].dev_decompressed_cap;

  // coalesce caps each group at DAMACY_MAX_CHUNKS_PER_WAVE chunks, so
  // groups consume atomically — either the whole group fits this wave
  // or it's deferred to the next.
  {
    struct read_op_group_iterator it;
    read_op_group_iterator_init(&it,
                                batch->read_op_groups,
                                batch->n_read_op_groups,
                                batch->n_groups_dispatched);
    struct read_op_group g;
    while (read_op_group_iterator_next(&it, &g)) {
      struct read_op* r = &batch->read_ops[g.read_op_idx];
      // Fill chunks each get a dedicated read_op (empty shard_path),
      // so fill never mixes with non-fill in a group.
      int is_fill_group = batch->chunk_plans[g.first_chunk].is_fill;
      uint64_t host_add = is_fill_group ? 0 : r->nbytes;
      if (host_cursor + host_add > hs->cap)
        break;
      if (take + g.n_chunks > DAMACY_MAX_CHUNKS_PER_WAVE)
        break;
      if (dev_cursor + g.total_decompressed > dev_cap)
        break;

      uint64_t reserved_host_off = host_cursor;
      if (!is_fill_group) {
        void* dst = wp->use_gds ? (void*)((uint8_t*)hs->dev_buf + host_cursor)
                                : (void*)((uint8_t*)hs->buf + host_cursor);
        hs->store_reads[n_reads++] = (struct store_read){
          .key = r->shard_path,
          .dst = dst,
          .offset = r->file_offset,
          .len = r->nbytes,
        };
        host_cursor += host_add;
      }
      for (uint32_t i = 0; i < g.n_chunks; ++i) {
        struct chunk_plan* c = &batch->chunk_plans[g.first_chunk + i];
        c->host_buf_offset = is_fill_group ? 0 : reserved_host_off;
        c->dev_decompressed_offset = dev_cursor;
        dev_cursor += c->decompressed_nbytes;
        take++;
      }
      completed_groups++;
    }
  }

  if (take == 0) {
    const struct read_op_group* g0 =
      (batch->n_groups_dispatched < batch->n_read_op_groups)
        ? &batch->read_op_groups[batch->n_groups_dispatched]
        : NULL;
    log_error("wave: group too large for slot "
              "(group n_chunks=%u total_decompressed=%llu; "
              "slot_cap=%llu dev_cap=%llu)",
              g0 ? g0->n_chunks : 0u,
              g0 ? (unsigned long long)g0->total_decompressed : 0ull,
              (unsigned long long)hs->cap,
              (unsigned long long)dev_cap);
    *err = DAMACY_OOM;
    return t;
  }
  hs->io_t_start_ns = monotonic_ns();
  hs->is_fill_wave = (n_reads == 0);
  hs->batch_pool_slot = batch_slot_idx;
  hs->batch_chunk_offset = base;
  hs->n_chunks = take;
  hs->used_bytes = host_cursor;
  hs->io_bytes = host_cursor;
  hs->state = SLOT_PEELING;
  t.prev_n_groups_dispatched = batch->n_groups_dispatched;
  batch->n_chunks_dispatched += take;
  batch->n_groups_dispatched = completed_groups;
  wp->stats->waves_emitted++;
  wp->stats->chunks_dispatched += take;

  t.slot_idx = slot_idx;
  t.n_reads = n_reads;
  return t;
}

struct store_event
wave_pool_peel_submit(struct wave_pool* wp,
                      const struct wave_pool_peel_ticket* t)
{
  if (t->slot_idx < 0 || t->n_reads == 0)
    return (struct store_event){ .seq = 0 };
  struct host_slab_slot* hs = &wp->slots[t->slot_idx];
  return wp->use_gds
           ? store_read_submit_dev(wp->store, hs->store_reads, t->n_reads)
           : store_read_submit(wp->store, hs->store_reads, t->n_reads);
}

enum damacy_status
wave_pool_peel_commit(struct wave_pool* wp,
                      struct wave_pool_peel_ticket* t,
                      struct store_event ev)
{
  if (t->consumed) {
    log_error("wave: peel_commit called twice on slot %d", t->slot_idx);
    return DAMACY_OK;
  }
  t->consumed = 1;
  if (t->slot_idx < 0)
    return DAMACY_OK;
  struct host_slab_slot* hs = &wp->slots[t->slot_idx];
  if (t->n_reads > 0 && ev.seq == 0) {
    struct damacy_batch_slot* batch = &wp->pool->slots[hs->batch_pool_slot];
    batch->n_chunks_dispatched -= hs->n_chunks;
    batch->n_groups_dispatched = t->prev_n_groups_dispatched;
    wp->stats->waves_emitted--;
    wp->stats->chunks_dispatched -= hs->n_chunks;
    slot_release(hs);
    return DAMACY_IO;
  }
  hs->io_event = ev;
  hs->state = SLOT_IO;
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
  // GDS path: cuFile wrote directly into the slot's device staging
  // buffer; alias wave->dev_compressed to it so the parse + decode
  // kernels read from the same memory the IO landed in. The slot
  // retains ownership; the alias clears on finalize.
  if (wp->use_gds)
    wave->dev_compressed = hs->dev_buf;
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
  // Pass 1: SLOT_IO → SLOT_READY when IO completes. Fill-only waves
  // (is_fill_wave) skip the IO query — no submission was made.
  for (uint8_t s = 0; s < wp->n_slots; ++s) {
    struct host_slab_slot* hs = &wp->slots[s];
    if (hs->state != SLOT_IO)
      continue;
    int ready = hs->is_fill_wave || store_event_query(wp->store, hs->io_event);
    if (ready) {
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
