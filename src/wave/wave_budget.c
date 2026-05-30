#include "wave_budget.h"

#include "assemble/assemble.h"
#include "damacy_config.h" // resolve_max_chunk_uncompressed
#include "damacy_limits.h"
#include "decoder/blosc1.h"
#include "decoder/decoder_memcpy.h"
#include "decoder/decoder_zstd.h"
#include "gpu_budget/gpu_budget.h"
#include "log/log.h"
#include "planner/planner.h"
#include "util/cuda_check.h"
#include "wave/fanout.h" // fanout_next_pow2

#include <stddef.h>

// Per-substream + per-batch upper bounds for the shared zstd decoder
// given an externally-chosen substream-batch cap `substreams`. zstd_per is
// the per-substream uncompressed upper bound (controls nvcomp's
// per-block scratch); total_uncompressed is the per-batch sum, capped
// at the per-wave decompressed arena since that's the real ceiling.
// Used by initial sizing (substreams = initial floor), grow (substreams =
// post-grow cap), and worst-case (substreams = structural ceiling).
static void
wave_decoder_caps(uint64_t substreams,
                  uint32_t max_chunks_per_wave,
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
    (uint64_t)max_chunks_per_wave * runtime_chunk_cap;
  const uint64_t subs_cap_bytes = substreams * runtime_chunk_cap;
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
wave_predict_bytes(uint32_t max_chunks_per_wave,
                   uint64_t host_slab_bytes,
                   uint64_t dev_decompressed_bytes,
                   struct wave_alloc_summary* out)
{
  const uint64_t cap = (uint64_t)max_chunks_per_wave;
  const uint64_t substreams = (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;

  out->dev_compressed = host_slab_bytes;
  out->dev_decompressed = dev_decompressed_bytes;
  out->blosc1_meta =
    cap * sizeof(struct assemble_chunk) + sizeof(struct blosc1_totals);
  out->fanout_soa = substreams * (2 * sizeof(void*) + 2 * sizeof(size_t)) +
                    cap * sizeof(struct gpu_memcpy_op);
  return DAMACY_OK;
}

// Predicted nvcomp scratch bytes for the shared decoder at the given
// substream batch cap. Includes nvcomp temp + actual-size + status
// arrays. Shared between the initial-floor predictor, the resolver,
// and the runtime grow path.
static enum damacy_status
predict_decoder_scratch_bytes(uint64_t substreams,
                              uint32_t max_chunks_per_wave,
                              uint64_t dev_per_wave,
                              uint64_t max_chunk_uncompressed_bytes,
                              uint64_t* out_bytes)
{
  uint64_t zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps(substreams,
                    max_chunks_per_wave,
                    dev_per_wave,
                    max_chunk_uncompressed_bytes,
                    &zstd_per,
                    &total_uncompressed);
  size_t zstd_temp = 0;
  if (decoder_zstd_query_temp_bytes((size_t)substreams,
                                    (size_t)zstd_per,
                                    (size_t)total_uncompressed,
                                    &zstd_temp))
    return DAMACY_CUDA;
  *out_bytes = (uint64_t)zstd_temp + substreams * sizeof(size_t) +
               substreams * sizeof(int);
  return DAMACY_OK;
}

enum damacy_status
wave_pool_shared_predict_bytes(uint32_t max_chunks_per_wave,
                               uint64_t dev_decompressed_bytes,
                               uint64_t max_chunk_uncompressed_bytes,
                               uint64_t* out_nvcomp_temp)
{
  return predict_decoder_scratch_bytes(
    (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP,
    max_chunks_per_wave,
    dev_decompressed_bytes,
    max_chunk_uncompressed_bytes,
    out_nvcomp_temp);
}

enum damacy_status
gpu_budget_predict(const struct damacy_config* cfg,
                   uint64_t host_slab_per_wave,
                   uint64_t dev_decompressed_per_wave,
                   struct gpu_budget_breakdown* out)
{
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  const uint32_t max_chunks_per_wave = resolve_max_chunks_per_wave(cfg);

  // Single source of truth for one wave's device-resident bytes lives
  // in wave_predict_bytes. Pool-level totals are 2× because the
  // orchestrator keeps two waves in flight. The shared nvcomp scratch
  // is queried separately and counted once.
  struct wave_alloc_summary per_wave = { 0 };
  enum damacy_status s = wave_predict_bytes(max_chunks_per_wave,
                                            host_slab_per_wave,
                                            dev_decompressed_per_wave,
                                            &per_wave);
  if (s != DAMACY_OK)
    return s;

  uint64_t nvcomp_temp = 0;
  s = wave_pool_shared_predict_bytes(max_chunks_per_wave,
                                     dev_decompressed_per_wave,
                                     runtime_chunk_cap,
                                     &nvcomp_temp);
  if (s != DAMACY_OK)
    return s;

  out->dev_compressed = 2ull * per_wave.dev_compressed;
  out->dev_decompressed = 2ull * per_wave.dev_decompressed;
  out->blosc1_meta = 2ull * per_wave.blosc1_meta;
  out->fanout_soa = 2ull * per_wave.fanout_soa;
  out->nvcomp_temp = nvcomp_temp;
  out->batch_metadata =
    2ull * (uint64_t)cfg->samples_per_batch * sizeof(struct sample_plan);
  out->total = out->dev_compressed + out->dev_decompressed + out->blosc1_meta +
               out->fanout_soa + out->nvcomp_temp + out->batch_metadata;
  return DAMACY_OK;
}

void
decoder_initial_caps(uint32_t max_chunks_per_wave,
                     uint64_t dev_per_wave,
                     uint64_t max_chunk_uncompressed_bytes,
                     size_t* out_substreams,
                     size_t* out_zstd_per,
                     size_t* out_total_uncompressed)
{
  const uint64_t substreams = (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;
  uint64_t zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps(substreams,
                    max_chunks_per_wave,
                    dev_per_wave,
                    max_chunk_uncompressed_bytes,
                    &zstd_per,
                    &total_uncompressed);
  *out_substreams = (size_t)substreams;
  *out_zstd_per = (size_t)zstd_per;
  *out_total_uncompressed = (size_t)total_uncompressed;
}

// Worst-case GPU footprint for one resolved geometry: assumes both
// per-wave fanout SOAs and the shared decoder scratch grow all the way
// to max_chunks_per_wave * max_substreams_per_chunk. Used by the
// resolver so the chosen geometry leaves headroom for observe-and-grow
// without the grow paths surprise-tripping the budget.
static enum damacy_status
predict_pool_total(uint32_t max_chunks_per_wave,
                   uint32_t max_substreams_per_wave,
                   uint64_t host_slab_per_wave,
                   uint64_t dev_per_wave,
                   uint64_t max_chunk_uncompressed_bytes,
                   uint32_t samples_per_batch,
                   uint64_t* out_total)
{
  struct wave_alloc_summary per_wave = { 0 };
  enum damacy_status s = wave_predict_bytes(
    max_chunks_per_wave, host_slab_per_wave, dev_per_wave, &per_wave);
  if (s != DAMACY_OK)
    return s;

  const uint64_t substreams_max = (uint64_t)max_substreams_per_wave;
  uint64_t nvcomp_temp_max = 0;
  s = predict_decoder_scratch_bytes(substreams_max,
                                    max_chunks_per_wave,
                                    dev_per_wave,
                                    max_chunk_uncompressed_bytes,
                                    &nvcomp_temp_max);
  if (s != DAMACY_OK)
    return s;

  // Per-wave fanout SOA grown to the structural ceiling. wave_predict_bytes
  // counted the *initial* slice; rewrite that component to assume max.
  const uint64_t substreams_init =
    (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;
  const uint64_t bytes_per_sub = 2ull * sizeof(void*) + 2ull * sizeof(size_t);
  const uint64_t fanout_slice_init = substreams_init * bytes_per_sub;
  const uint64_t fanout_slice_max = substreams_max * bytes_per_sub;
  const uint64_t fanout_soa_worst =
    per_wave.fanout_soa - fanout_slice_init + fanout_slice_max;

  uint64_t total =
    2ull * (per_wave.dev_compressed + per_wave.dev_decompressed +
            per_wave.blosc1_meta + fanout_soa_worst) +
    nvcomp_temp_max +
    2ull * (uint64_t)samples_per_batch * sizeof(struct sample_plan);
  *out_total = total;
  return DAMACY_OK;
}

enum damacy_status
wave_pool_resolve_sizing(uint32_t max_chunks_per_wave,
                         uint32_t max_substreams_per_chunk,
                         uint64_t max_gpu_memory_bytes,
                         uint64_t max_chunk_uncompressed_bytes,
                         uint32_t samples_per_batch,
                         struct wave_pool_sizing* out)
{
  const uint32_t max_substreams_per_wave = damacy_max_substreams_per_wave(
    max_chunks_per_wave, max_substreams_per_chunk);
  // Smallest viable geometry: hold at least one chunk at the runtime
  // cap per wave. host_slab_per_wave needs to fit a compressed chunk;
  // the dev arena holds it uncompressed. Use max_chunk_uncompressed_bytes
  // for both — real compressed payload is bounded by the uncompressed
  // size, so this is a tight lower bound.
  const uint64_t min_per_wave = max_chunk_uncompressed_bytes;
  uint64_t total_min = 0;
  enum damacy_status s = predict_pool_total(max_chunks_per_wave,
                                            max_substreams_per_wave,
                                            min_per_wave,
                                            min_per_wave,
                                            max_chunk_uncompressed_bytes,
                                            samples_per_batch,
                                            &total_min);
  if (s != DAMACY_OK)
    return s;
  if (total_min > max_gpu_memory_bytes) {
    log_error("damacy: max_gpu_memory_bytes=%llu too small to fit even one "
              "chunk per wave (predicted=%llu, max_chunk_uncompressed=%llu)",
              (unsigned long long)max_gpu_memory_bytes,
              (unsigned long long)total_min,
              (unsigned long long)max_chunk_uncompressed_bytes);
    return DAMACY_BUDGET;
  }

  // dev_compressed + dev_decompressed scale 1:1 with per_wave; 2× for
  // two waves. 4 = 2 × 2.
  //   total_min + 4 * delta_per_wave <= cap
  //   ⇒ delta_per_wave_max = (cap - total_min) / 4
  // Round down to 1 MB so logs are readable; the back-off loop catches
  // any overshoot.
  const uint64_t headroom = max_gpu_memory_bytes - total_min;
  const uint64_t delta_per_wave_cap = headroom / 4;
  const uint64_t step = 1ull << 20; // 1 MB granularity
  uint64_t per_wave = min_per_wave + (delta_per_wave_cap / step) * step;
  if (per_wave < min_per_wave)
    per_wave = min_per_wave;
  // chunks-per-wave caps per_wave usefulness — beyond that × per-chunk
  // cap is wasted memory + slows nvcomp through inflated workspace.
  const uint64_t useful_max =
    (uint64_t)max_chunks_per_wave * max_chunk_uncompressed_bytes;
  if (per_wave > useful_max)
    per_wave = useful_max;

  uint64_t predicted = 0;
  s = predict_pool_total(max_chunks_per_wave,
                         max_substreams_per_wave,
                         per_wave,
                         per_wave,
                         max_chunk_uncompressed_bytes,
                         samples_per_batch,
                         &predicted);
  if (s != DAMACY_OK)
    return s;
  while (predicted > max_gpu_memory_bytes && per_wave > min_per_wave) {
    per_wave -= step;
    if (per_wave < min_per_wave)
      per_wave = min_per_wave;
    s = predict_pool_total(max_chunks_per_wave,
                           max_substreams_per_wave,
                           per_wave,
                           per_wave,
                           max_chunk_uncompressed_bytes,
                           samples_per_batch,
                           &predicted);
    if (s != DAMACY_OK)
      return s;
  }
  // Back-off loop terminates at per_wave == min_per_wave with
  // predicted <= max_gpu_memory_bytes: the top-level check above
  // already rejected total_min > max, and per_wave == min_per_wave
  // reproduces total_min.

  out->host_slab_per_wave = per_wave;
  out->dev_decompressed_per_wave = per_wave;
  out->worst_case_total_bytes = predicted;
  return DAMACY_OK;
}

enum damacy_status
decoder_scratch_grow(struct decoder_zstd* decoder,
                     CUstream stream_decode,
                     uint32_t max_chunks_per_wave,
                     uint32_t max_substreams_per_wave,
                     uint64_t dev_per_wave,
                     uint64_t max_chunk_uncompressed_bytes,
                     struct gpu_budget* budget,
                     size_t need)
{
  const size_t cur = decoder_zstd_cur_max_batch(decoder);
  if (cur == 0) {
    // Prior grow left the decoder destroyed-internally. Re-growing
    // would leak the partial state and still fail; surface CUDA.
    log_error("zstd decoder: cur_max_batch=0 (prior grow failed)");
    return DAMACY_CUDA;
  }
  if (need <= cur)
    return DAMACY_OK;
  size_t new_cap = fanout_next_pow2(need);
  if (new_cap > max_substreams_per_wave)
    new_cap = max_substreams_per_wave;

  uint64_t old_bytes = 0, new_bytes = 0;
  enum damacy_status sp =
    predict_decoder_scratch_bytes((uint64_t)cur,
                                  max_chunks_per_wave,
                                  dev_per_wave,
                                  max_chunk_uncompressed_bytes,
                                  &old_bytes);
  if (sp != DAMACY_OK)
    return sp;
  sp = predict_decoder_scratch_bytes((uint64_t)new_cap,
                                     max_chunks_per_wave,
                                     dev_per_wave,
                                     max_chunk_uncompressed_bytes,
                                     &new_bytes);
  if (sp != DAMACY_OK)
    return sp;
  const uint64_t delta_bytes =
    new_bytes > old_bytes ? new_bytes - old_bytes : 0;
  enum damacy_status bs =
    gpu_budget_try_commit(budget, delta_bytes, "zstd decoder grow");
  if (bs != DAMACY_OK)
    return bs;

  // Any prior decode launched against the shared scratch must retire
  // before we free it. FIFO on stream_decode means this also drains
  // pending post-decode + assemble.
  CU(CudaFail, cuStreamSynchronize(stream_decode));

  uint64_t zstd_per = 0, total_uncompressed = 0;
  wave_decoder_caps((uint64_t)new_cap,
                    max_chunks_per_wave,
                    dev_per_wave,
                    max_chunk_uncompressed_bytes,
                    &zstd_per,
                    &total_uncompressed);
  if (decoder_zstd_grow(
        decoder, new_cap, (size_t)zstd_per, (size_t)total_uncompressed) != 0)
    goto CudaFail;

  // Internal observe-and-grow event — DEBUG to keep INFO clean.
  log_debug("zstd decoder: grew %zu -> %zu (need=%zu, +%llu bytes)",
            cur,
            new_cap,
            need,
            (unsigned long long)delta_bytes);
  return DAMACY_OK;

CudaFail:
  // grow tried + failed: roll back the commit so committed reflects
  // the un-grown state. decoder_zstd_grow leaves the decoder zombie;
  // subsequent kick_h2d calls short-circuit on cur_max_batch==0.
  gpu_budget_release(budget, delta_bytes);
  return DAMACY_CUDA;
}
