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

static void
wave_decoder_caps(uint64_t substreams,
                  uint32_t max_chunks_per_wave,
                  uint64_t dev_decompressed_bytes,
                  uint64_t max_chunk_uncompressed_bytes,
                  uint64_t* out_zstd_per,
                  uint64_t* out_total_uncompressed)
{
  if (dev_decompressed_bytes == 0 || max_chunk_uncompressed_bytes == 0) {
    *out_zstd_per = 0;
    *out_total_uncompressed = 0;
    return;
  }
  const uint64_t runtime_chunk_cap = max_chunk_uncompressed_bytes;
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
                   uint64_t input_staging_bytes,
                   uint64_t dev_decompressed_bytes,
                   struct wave_alloc_summary* out)
{
  const uint64_t cap = (uint64_t)max_chunks_per_wave;
  const uint64_t substreams = (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;

  out->dev_compressed = input_staging_bytes;
  out->dev_decompressed = dev_decompressed_bytes;
  out->blosc1_meta =
    cap * sizeof(struct assemble_chunk) + sizeof(struct blosc1_totals);
  out->fanout_soa = substreams * (2 * sizeof(void*) + 2 * sizeof(size_t)) +
                    cap * sizeof(struct gpu_memcpy_op);
  return DAMACY_OK;
}

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
                   const struct input_transfer_resources* input,
                   uint64_t dev_decompressed_per_wave,
                   struct gpu_budget_breakdown* out)
{
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  const uint32_t max_chunks_per_wave = resolve_max_chunks_per_wave(cfg);

  struct wave_alloc_summary per_wave = { 0 };
  enum damacy_status s = wave_predict_bytes(max_chunks_per_wave,
                                            input->wave_device_bytes,
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

  out->dev_compressed = input->device_staging_bytes;
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

static enum damacy_status
predict_pool_total(uint32_t max_chunks_per_wave,
                   uint32_t max_substreams_per_wave,
                   uint8_t input_device_staging_buffers,
                   uint64_t input_staging_per_wave,
                   uint64_t dev_per_wave,
                   uint64_t max_chunk_uncompressed_bytes,
                   uint32_t samples_per_batch,
                   uint64_t* out_total)
{
  struct wave_alloc_summary per_wave = { 0 };
  enum damacy_status s =
    wave_predict_bytes(max_chunks_per_wave, 0, dev_per_wave, &per_wave);
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

  const uint64_t substreams_init =
    (uint64_t)DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP;
  const uint64_t bytes_per_sub = 2ull * sizeof(void*) + 2ull * sizeof(size_t);
  const uint64_t fanout_slice_init = substreams_init * bytes_per_sub;
  const uint64_t fanout_slice_max = substreams_max * bytes_per_sub;
  const uint64_t fanout_soa_worst =
    per_wave.fanout_soa - fanout_slice_init + fanout_slice_max;

  uint64_t total =
    (uint64_t)input_device_staging_buffers * input_staging_per_wave +
    2ull *
      (per_wave.dev_decompressed + per_wave.blosc1_meta + fanout_soa_worst) +
    nvcomp_temp_max +
    2ull * (uint64_t)samples_per_batch * sizeof(struct sample_plan);
  *out_total = total;
  return DAMACY_OK;
}

enum damacy_status
wave_pool_resolve_sizing(uint32_t max_chunks_per_wave,
                         uint32_t max_substreams_per_chunk,
                         uint8_t input_device_staging_buffers,
                         uint64_t max_gpu_memory_bytes,
                         uint64_t max_chunk_uncompressed_bytes,
                         uint32_t samples_per_batch,
                         struct wave_pool_sizing* out)
{
  const uint32_t max_substreams_per_wave = DAMACY_MAX_SUBSTREAMS_PER_WAVE(
    max_chunks_per_wave, max_substreams_per_chunk);
  const uint64_t min_per_wave = max_chunk_uncompressed_bytes;
  uint64_t total_min = 0;
  enum damacy_status s = predict_pool_total(max_chunks_per_wave,
                                            max_substreams_per_wave,
                                            input_device_staging_buffers,
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

  const uint64_t headroom = max_gpu_memory_bytes - total_min;
  const uint64_t scale =
    (uint64_t)input_device_staging_buffers + DAMACY_N_WAVES;
  const uint64_t delta_per_wave_cap = scale > 0 ? headroom / scale : 0;
  const uint64_t step = 1ull << 20; // 1 MB granularity
  uint64_t per_wave = min_per_wave + (delta_per_wave_cap / step) * step;
  if (per_wave < min_per_wave)
    per_wave = min_per_wave;
  const uint64_t useful_max =
    (uint64_t)max_chunks_per_wave * max_chunk_uncompressed_bytes;
  if (per_wave > useful_max)
    per_wave = useful_max;

  uint64_t predicted = 0;
  s = predict_pool_total(max_chunks_per_wave,
                         max_substreams_per_wave,
                         input_device_staging_buffers,
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
                           input_device_staging_buffers,
                           per_wave,
                           per_wave,
                           max_chunk_uncompressed_bytes,
                           samples_per_batch,
                           &predicted);
    if (s != DAMACY_OK)
      return s;
  }
  out->input_staging_per_wave = per_wave;
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
