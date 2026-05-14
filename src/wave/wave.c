#include "wave.h"

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

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Wave-index helper for NVTX range labels — call sites use
// wave_index_of(wp, wave) to compute "w0"/"w1" without touching
// damacy_wave's storage. Returns ptrdiff_t so the pointer subtraction
// isn't narrowed; format with %td at call sites.
static inline ptrdiff_t
wave_index_of(const struct wave_pool* wp, const struct damacy_wave* wave)
{
  return wave - wp->waves;
}

// Per-substream + per-batch upper bounds for the shared zstd decoder
// given an externally-chosen substream-batch cap `zsubs`. zstd_per is
// the per-substream uncompressed upper bound (controls nvcomp's
// per-block scratch); total_uncompressed is the per-batch sum, capped
// at the per-wave decompressed arena since that's the real ceiling.
// Used by both initial sizing (zsubs = initial floor) and grow
// (zsubs = post-grow cap).
static void
wave_decoder_caps(uint64_t zsubs,
                  uint64_t dev_decompressed_bytes,
                  uint64_t max_chunk_uncompressed_bytes,
                  uint64_t* out_zstd_per,
                  uint64_t* out_total_uncompressed)
{
  // Caller's config validation should preclude this; guard explicitly
  // so downstream nvcomp queries never see zero-sized inputs.
  if (dev_decompressed_bytes == 0 || max_chunk_uncompressed_bytes == 0) {
    *out_zstd_per = 0;
    *out_total_uncompressed = 0;
    return;
  }
  const uint64_t runtime_chunk_cap = max_chunk_uncompressed_bytes;
  // Tight upper bound on bytes nvcomp will emit per call: structural
  // chunk-count cap × per-chunk cap. dev_decompressed_bytes is the arena
  // limit, but with chunks-per-wave capped this is usually smaller and
  // keeps nvcomp from over-allocating workspace for memory it won't use.
  const uint64_t chunks_cap_bytes =
    (uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE * runtime_chunk_cap;
  const uint64_t subs_cap_bytes = zsubs * runtime_chunk_cap;
  uint64_t total_uncompressed = dev_decompressed_bytes;
  if (chunks_cap_bytes < total_uncompressed)
    total_uncompressed = chunks_cap_bytes;
  if (subs_cap_bytes < total_uncompressed)
    total_uncompressed = subs_cap_bytes;
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
  // Initial floor — pool may grow this at runtime (observe-and-grow).
  // Predict the *initial* footprint; later grows are accounted via
  // wave_pool_shared_predict_bytes the same way.
  const uint64_t zsubs = (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;

  out->dev_compressed = host_slab_bytes;
  out->dev_decompressed = dev_decompressed_bytes;
  out->blosc1_meta =
    cap * sizeof(struct assemble_chunk) + sizeof(struct blosc1_totals);
  out->fanout_soa = zsubs * (2 * sizeof(void*) + 2 * sizeof(size_t)) +
                    (uint64_t)DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                      sizeof(struct gpu_memcpy_op);
  return DAMACY_OK;
}

enum damacy_status
wave_pool_shared_predict_bytes(uint64_t dev_decompressed_bytes,
                               uint64_t max_chunk_uncompressed_bytes,
                               uint64_t* out_nvcomp_temp)
{
  // Predict the *initial* decoder scratch; observe-and-grow may raise
  // this at runtime up to DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE.
  const uint64_t zsubs = (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;
  uint64_t zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps(zsubs,
                    dev_decompressed_bytes,
                    max_chunk_uncompressed_bytes,
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

// Worst-case GPU footprint for one resolved geometry: assumes both
// per-wave fanout SOAs and the shared decoder scratch grow all the way
// to DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE. Used by the resolver so the
// chosen geometry leaves headroom for observe-and-grow without the
// grow paths surprise-tripping the budget.
static enum damacy_status
predict_pool_total(uint64_t host_slab_per_wave,
                   uint64_t dev_per_wave,
                   uint64_t max_chunk_uncompressed_bytes,
                   uint32_t batch_size,
                   uint64_t* out_total)
{
  struct wave_alloc_summary per_wave = { 0 };
  enum damacy_status s =
    wave_predict_bytes(host_slab_per_wave, dev_per_wave, &per_wave);
  if (s != DAMACY_OK)
    return s;

  // Worst-case shared decoder scratch.
  const uint64_t zsubs_max = (uint64_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;
  uint64_t zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps(zsubs_max,
                    dev_per_wave,
                    max_chunk_uncompressed_bytes,
                    &zstd_per,
                    &total_uncompressed);
  size_t zstd_temp_max = 0;
  if (decoder_zstd_query_temp_bytes((size_t)zsubs_max,
                                    (size_t)zstd_per,
                                    (size_t)total_uncompressed,
                                    &zstd_temp_max))
    return DAMACY_CUDA;
  const uint64_t nvcomp_temp_max = (uint64_t)zstd_temp_max +
                                   zsubs_max * sizeof(size_t) +
                                   zsubs_max * sizeof(int);

  // Per-wave fanout SOA grown to the structural ceiling. wave_predict_bytes
  // counted the *initial* slice; rewrite that component to assume max.
  const uint64_t zsubs_init = (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;
  const uint64_t bytes_per_sub = 2ull * sizeof(void*) + 2ull * sizeof(size_t);
  const uint64_t fanout_slice_init = zsubs_init * bytes_per_sub;
  const uint64_t fanout_slice_max = zsubs_max * bytes_per_sub;
  const uint64_t fanout_soa_worst =
    per_wave.fanout_soa - fanout_slice_init + fanout_slice_max;

  uint64_t total = 2ull * (per_wave.dev_compressed + per_wave.dev_decompressed +
                           per_wave.blosc1_meta + fanout_soa_worst) +
                   nvcomp_temp_max +
                   2ull * (uint64_t)batch_size * sizeof(struct sample_plan);
  *out_total = total;
  return DAMACY_OK;
}

enum damacy_status
wave_pool_resolve_sizing(uint64_t max_gpu_memory_bytes,
                         uint64_t max_chunk_uncompressed_bytes,
                         uint32_t batch_size,
                         struct wave_pool_sizing* out)
{
  // Smallest viable geometry: hold at least one chunk at the runtime
  // cap per wave. host_slab_per_wave needs to fit a compressed chunk;
  // the dev arena holds it uncompressed. Use max_chunk_uncompressed_bytes
  // for both — real compressed payload is bounded by the uncompressed
  // size, so this is a tight lower bound.
  const uint64_t min_per_wave = max_chunk_uncompressed_bytes;
  uint64_t total_min = 0;
  enum damacy_status s = predict_pool_total(min_per_wave,
                                            min_per_wave,
                                            max_chunk_uncompressed_bytes,
                                            batch_size,
                                            &total_min);
  if (s != DAMACY_OK)
    return s;
  if (total_min > max_gpu_memory_bytes) {
    log_error("damacy: max_gpu_memory_bytes=%llu too small to fit even one "
              "chunk per wave (predicted=%llu, max_chunk_uncompressed=%llu)",
              (unsigned long long)max_gpu_memory_bytes,
              (unsigned long long)total_min,
              (unsigned long long)max_chunk_uncompressed_bytes);
    return DAMACY_OOM;
  }

  // dev_compressed + dev_decompressed scale 1:1 with per_wave; 2× for
  // two waves. 4 = 2 × 2.
  //   total_min + 4 * delta_per_wave <= cap
  //   ⇒ delta_per_wave_max = (cap - total_min) / 4
  // Round down to 1 MB so logs are readable; the back-off loop below
  // catches any overshoot. The assert after the assignment locks the
  // assumption that host_slab_per_wave == dev_decompressed_per_wave.
  const uint64_t headroom = max_gpu_memory_bytes - total_min;
  const uint64_t delta_per_wave_cap = headroom / 4;
  const uint64_t step = 1ull << 20; // 1 MB granularity
  uint64_t per_wave = min_per_wave + (delta_per_wave_cap / step) * step;
  if (per_wave < min_per_wave)
    per_wave = min_per_wave;
  // Cap at the maximum useful size: chunks-per-wave is bounded at
  // DAMACY_MAX_CHUNKS_PER_WAVE, so any per_wave beyond that × per-chunk
  // cap is wasted memory + slows nvcomp through inflated workspace.
  const uint64_t useful_max =
    (uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE * max_chunk_uncompressed_bytes;
  if (per_wave > useful_max)
    per_wave = useful_max;

  uint64_t predicted = 0;
  s = predict_pool_total(
    per_wave, per_wave, max_chunk_uncompressed_bytes, batch_size, &predicted);
  if (s != DAMACY_OK)
    return s;
  while (predicted > max_gpu_memory_bytes && per_wave > min_per_wave) {
    per_wave -= step;
    if (per_wave < min_per_wave)
      per_wave = min_per_wave;
    s = predict_pool_total(
      per_wave, per_wave, max_chunk_uncompressed_bytes, batch_size, &predicted);
    if (s != DAMACY_OK)
      return s;
  }
  // The back-off loop is guaranteed to terminate at per_wave ==
  // min_per_wave with predicted <= max_gpu_memory_bytes: the top-level
  // check above already rejected total_min > max, and per_wave ==
  // min_per_wave reproduces total_min. So no post-loop OOM branch is
  // needed — the top-level reject is the only path that surfaces a
  // too-small budget.

  out->host_slab_per_wave = per_wave;
  out->dev_decompressed_per_wave = per_wave;
  out->worst_case_total_bytes = predicted;
  // Lock in the invariant the divisor=6 derivation depends on.
  // Cheap: this resolver runs once per damacy_create.
  if (out->host_slab_per_wave != out->dev_decompressed_per_wave) {
    log_error("damacy: resolver invariant broken: host_slab_per_wave=%llu "
              "!= dev_decompressed_per_wave=%llu (the 6=2*3 divisor "
              "assumes a single per_wave scales both)",
              (unsigned long long)out->host_slab_per_wave,
              (unsigned long long)out->dev_decompressed_per_wave);
    return DAMACY_INVAL;
  }
  return DAMACY_OK;
}

int
wave_init(struct damacy_wave* wave,
          uint64_t slot_cap_bytes,
          uint64_t dev_decompressed_bytes)
{
  // Self-zero so wave_destroy on the failure path doesn't free
  // uninitialized pointers — caller may have passed stack memory.
  memset(wave, 0, sizeof(*wave));
  wave->state = WAVE_FREE;
  wave->bound_slot = -1;
  wave->dev_decompressed_cap = dev_decompressed_bytes;

  CUdeviceptr dptr = 0;
  CU(Error, cuMemAlloc(&dptr, slot_cap_bytes));
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

  CU(Error, cuMemAlloc(&dptr, (size_t)cap * sizeof(struct assemble_chunk)));
  wave->d_assemble_chunks = (struct assemble_chunk*)(uintptr_t)dptr;

  CU(Error, cuMemAlloc(&dptr, sizeof(struct blosc1_totals)));
  wave->d_blosc1_totals = (struct blosc1_totals*)(uintptr_t)dptr;

  // Initial per-wave fanout cap — kick_h2d grows this wave's SOA when
  // n_chunks * MAX_BLOCKS exceeds it. Independent of the other wave.
  const size_t zsubs = DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;
  if (fanout_alloc_pinned(&wave->h_zstd_fan, &wave->zstd_fan, zsubs))
    goto Error;
  wave->fanout_cap = (uint32_t)zsubs;

  CU(Error,
     cuMemAllocHost((void**)&wave->h_memcpy_ops,
                    DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                      sizeof(struct gpu_memcpy_op)));
  CU(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                  sizeof(struct gpu_memcpy_op)));
  wave->d_memcpy_ops = (struct gpu_memcpy_op*)(uintptr_t)dptr;

  CU(Error, cuEventCreate(&wave->ev.h2d_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.bulk_h2d_end, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.h2d_end, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.decomp_start, CU_EVENT_DEFAULT));
  CU(Error, cuEventCreate(&wave->ev.decode_done, CU_EVENT_DEFAULT));
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
      wave->h_chunks,        wave->scratch.hdrs,    wave->scratch.counts,
      wave->scratch.offsets, wave->scratch.bstarts, wave->scratch.block_ends,
      wave->h_blosc1_totals, wave->h_memcpy_ops,    wave->h_assemble_chunks,
    };
    for (size_t i = 0; i < countof(host_ptrs); ++i)
      if (host_ptrs[i])
        cuMemFreeHost(host_ptrs[i]);
    // Pinned fanout (host + device) — same NULL-safe per-pointer pattern.
    fanout_free_pinned(&wave->h_zstd_fan, &wave->zstd_fan);
    void* const dev_ptrs[] = {
      wave->dev_compressed,  wave->dev_decompressed, wave->d_assemble_chunks,
      wave->d_blosc1_totals, wave->d_memcpy_ops,
    };
    for (size_t i = 0; i < countof(dev_ptrs); ++i)
      if (dev_ptrs[i])
        cuMemFree(CUDPTR(dev_ptrs[i]));
    CUevent* const events[] = { &wave->ev.h2d_start,   &wave->ev.bulk_h2d_end,
                                &wave->ev.h2d_end,     &wave->ev.decomp_start,
                                &wave->ev.decode_done, &wave->ev.decomp_end,
                                &wave->ev.asm_start,   &wave->ev.asm_end };
    for (size_t i = 0; i < countof(events); ++i)
      if (*events[i])
        cuEventDestroy_v2(*events[i]);
  }
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
  wp->failed_status = failed_status;
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

  // Initial floor for the shared decoder; wave_pool_grow_decoder bumps
  // it lazily when a wave's substream count exceeds the current cap.
  // Per-wave fanout SOAs grow independently via fanout_grow.
  const uint64_t zsubs = (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;
  uint64_t zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps(zsubs,
                    dev_per_wave,
                    max_chunk_uncompressed_bytes,
                    &zstd_per,
                    &total_uncompressed);
  wp->zstd_decoder = decoder_zstd_create(
    (size_t)zsubs, (size_t)zstd_per, (size_t)total_uncompressed);
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
  wp->failed_status = NULL;
}

// Grow the pool's shared zstd decoder scratch (d_temp + d_statuses +
// d_uncompressed_actual_sizes) to fit `need` substreams in a single
// batch. Triggered from kick_h2d when a wave's upper-bound substream
// count exceeds the decoder's current cap. The scratch is monotonic and
// wave-agnostic; growing it never disturbs any wave's fanout SOA.
//
// Synchronizes stream_decode first so any prior decode reading the
// to-be-freed scratch has retired. stream_h2d is NOT touched — the
// other wave's queued fanout_upload writes to ITS OWN SOA, not the
// decoder scratch we're freeing.
//
// On decoder_zstd_grow failure the decoder is left zombie
// (cur_max_batch == 0); subsequent kick_h2d calls short-circuit on
// that and surface DAMACY_CUDA without re-attempting the grow.
// Predicted nvcomp scratch bytes for the shared decoder at the given
// substream batch cap. Includes nvcomp temp + actual-size + status
// arrays — matches wave_pool_shared_predict_bytes's accounting.
static enum damacy_status
predict_decoder_scratch_bytes(uint64_t zsubs,
                              uint64_t dev_per_wave,
                              uint64_t max_chunk_uncompressed_bytes,
                              uint64_t* out_bytes)
{
  uint64_t zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps(zsubs,
                    dev_per_wave,
                    max_chunk_uncompressed_bytes,
                    &zstd_per,
                    &total_uncompressed);
  size_t zstd_temp = 0;
  if (decoder_zstd_query_temp_bytes((size_t)zsubs,
                                    (size_t)zstd_per,
                                    (size_t)total_uncompressed,
                                    &zstd_temp))
    return DAMACY_CUDA;
  *out_bytes =
    (uint64_t)zstd_temp + zsubs * sizeof(size_t) + zsubs * sizeof(int);
  return DAMACY_OK;
}

static enum damacy_status
wave_pool_grow_decoder(struct wave_pool* wp, size_t need)
{
  const size_t cur = decoder_zstd_cur_max_batch(wp->zstd_decoder);
  if (cur == 0) {
    // Prior grow left the decoder destroyed-internally. Re-growing
    // would leak the partial state and still fail; surface CUDA.
    log_error("zstd decoder: cur_max_batch=0 (prior grow failed)");
    return DAMACY_CUDA;
  }
  if (need <= cur)
    return DAMACY_OK;
  size_t new_cap = fanout_next_pow2(need);
  if (new_cap > (size_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE)
    new_cap = (size_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;

  // Enforce the GPU budget. Compute the byte delta from the current
  // decoder scratch to the proposed one. If the change would push
  // past the cap, return OOM before touching the device.
  uint64_t old_bytes = 0, new_bytes = 0;
  enum damacy_status sp =
    predict_decoder_scratch_bytes((uint64_t)cur,
                                  wp->dev_per_wave,
                                  wp->max_chunk_uncompressed_bytes,
                                  &old_bytes);
  if (sp != DAMACY_OK)
    return sp;
  sp = predict_decoder_scratch_bytes((uint64_t)new_cap,
                                     wp->dev_per_wave,
                                     wp->max_chunk_uncompressed_bytes,
                                     &new_bytes);
  if (sp != DAMACY_OK)
    return sp;
  const uint64_t delta_bytes =
    new_bytes > old_bytes ? new_bytes - old_bytes : 0;
  enum damacy_status bs =
    gpu_budget_try_commit(wp->budget, delta_bytes, "zstd decoder grow");
  if (bs != DAMACY_OK)
    return bs;

  // Any prior decode launched against the shared scratch must retire
  // before we free it. FIFO on stream_decode means this also drains
  // pending post-decode + assemble.
  CU(CudaFail, cuStreamSynchronize(wp->stream_decode));

  uint64_t zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps((uint64_t)new_cap,
                    wp->dev_per_wave,
                    wp->max_chunk_uncompressed_bytes,
                    &zstd_per,
                    &total_uncompressed);
  if (decoder_zstd_grow(wp->zstd_decoder,
                        new_cap,
                        (size_t)zstd_per,
                        (size_t)total_uncompressed) != 0)
    goto CudaFail;

  log_info("zstd decoder: grew %zu -> %zu (need=%zu, +%llu bytes)",
           cur,
           new_cap,
           need,
           (unsigned long long)delta_bytes);
  return DAMACY_OK;

CudaFail:
  // grow tried + failed: roll back the commit so committed reflects
  // the un-grown state. decoder_zstd_grow leaves the decoder zombie;
  // subsequent kick_h2d calls short-circuit on cur_max_batch==0.
  gpu_budget_release(wp->budget, delta_bytes);
  *wp->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
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
    gs = wave_pool_grow_decoder(wp, need_zsubs);
  if (gs != DAMACY_OK) {
    record_io_metric(wp, wave);
    cuEventRecord(wave->ev.h2d_end, wp->stream_h2d);
    *wp->failed_status = gs;
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
    *wp->failed_status = DAMACY_DECODE;
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
  *wp->failed_status = DAMACY_CUDA;
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
  *wp->failed_status = DAMACY_DECODE;
  rs = DAMACY_DECODE;
  goto Done;
CudaFail:
  *wp->failed_status = DAMACY_CUDA;
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
  *wp->failed_status = DAMACY_DECODE;
  rs = DAMACY_DECODE;
  goto Done;
CudaFail:
  *wp->failed_status = DAMACY_CUDA;
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
  *wp->failed_status = DAMACY_CUDA;
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

// Surfaces nvcomp errors before the slot transitions so damacy_pop's
// failed_status check can bail before handing out the batch.
static void
finalize_wave(struct wave_pool* wp, struct damacy_wave* wave)
{
  damacy_nvtx_range_pushf("finalize_wave/w%td", wave_index_of(wp, wave));
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
  damacy_nvtx_range_pop();
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
    *wp->failed_status = DAMACY_OOM;
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
    *wp->failed_status = DAMACY_IO;
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
            *wp->failed_status = DAMACY_CUDA;
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
