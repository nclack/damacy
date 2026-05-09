#include "gpu_budget.h"

#include "assemble/assemble.h"
#include "damacy_config.h"
#include "damacy_limits.h"
#include "decoder/blosc1.h"
#include "decoder/decoder_lz4.h"
#include "decoder/decoder_memcpy.h"
#include "decoder/decoder_zstd.h"
#include "decoder/shuffle.h"
#include "planner/planner.h"

// Sum of struct-array allocs wave_init makes for blosc1 parse + assemble
// metadata, scaled to one wave. With the parse moved to the host the only
// device-resident blosc1 metadata is the totals struct used by status_reduce.
static uint64_t
wave_blosc1_meta_bytes(void)
{
  const uint64_t cap = (uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE;
  return cap * sizeof(struct assemble_chunk) + sizeof(struct blosc1_totals);
}

// One wave's nvcomp fanout SOA + memcpy/shuffle op arrays.
static uint64_t
wave_fanout_soa_bytes(uint8_t max_bpe)
{
  const uint64_t zsubs = (uint64_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;
  const uint64_t lsubs = lz4_subs_per_wave(max_bpe);
  uint64_t soa = zsubs * (2 * sizeof(void*) + 2 * sizeof(size_t)) +
                 lsubs * (2 * sizeof(void*) + 2 * sizeof(size_t));
  soa += (uint64_t)DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
         sizeof(struct gpu_memcpy_op);
  soa += 2ull * (uint64_t)DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
         sizeof(struct gpu_shuffle_op);
  return soa;
}

// nvcomp scratch (temp + actual-size + status arrays) for one wave.
// Mirrors wave_init's capacity math.
static enum damacy_status
wave_nvcomp_bytes(const struct damacy_config* cfg, uint64_t* out)
{
  const uint64_t dev_per_wave = cfg->device_buffer_bytes / 2;
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  const uint8_t max_bpe = resolve_max_bpe(cfg);
  const uint64_t zsubs = (uint64_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;
  const uint64_t lsubs = lz4_subs_per_wave(max_bpe);
  const uint64_t cap_worst =
    (uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE * runtime_chunk_cap;
  const uint64_t total_uncompressed =
    dev_per_wave < cap_worst ? dev_per_wave : cap_worst;
  uint64_t zstd_per = runtime_chunk_cap;
  if (zstd_per > total_uncompressed && total_uncompressed > 0)
    zstd_per = total_uncompressed;
  uint64_t lz4_per = zstd_per / max_bpe;
  if (lz4_per == 0)
    lz4_per = 1;
  size_t zstd_temp = 0, lz4_temp = 0;
  if (decoder_zstd_query_temp_bytes(
        zsubs, (size_t)zstd_per, (size_t)total_uncompressed, &zstd_temp))
    return DAMACY_CUDA;
  if (decoder_lz4_query_temp_bytes(
        lsubs, (size_t)lz4_per, (size_t)total_uncompressed, &lz4_temp))
    return DAMACY_CUDA;
  *out = (uint64_t)zstd_temp + zsubs * sizeof(size_t) + zsubs * sizeof(int) +
         (uint64_t)lz4_temp + lsubs * sizeof(size_t) + lsubs * sizeof(int);
  return DAMACY_OK;
}

enum damacy_status
gpu_budget_compute(const struct damacy_config* cfg, struct gpu_budget* out)
{
  const uint64_t host_per_wave = cfg->host_buffer_bytes / 2;
  const uint64_t dev_per_wave = cfg->device_buffer_bytes / 2;
  const uint8_t max_bpe = resolve_max_bpe(cfg);

  uint64_t per_wave_nvcomp = 0;
  enum damacy_status s = wave_nvcomp_bytes(cfg, &per_wave_nvcomp);
  if (s != DAMACY_OK)
    return s;

  out->dev_compressed = 2ull * host_per_wave;
  out->dev_decompressed = 2ull * dev_per_wave;
  out->dev_unshuffle_scratch = 2ull * dev_per_wave;
  out->blosc1_meta = 2ull * wave_blosc1_meta_bytes();
  out->fanout_soa = 2ull * wave_fanout_soa_bytes(max_bpe);
  out->nvcomp_temp = 2ull * per_wave_nvcomp;
  out->batch_metadata =
    2ull * (uint64_t)cfg->batch_size * sizeof(struct sample_plan);
  out->total = out->dev_compressed + out->dev_decompressed +
               out->dev_unshuffle_scratch + out->blosc1_meta + out->fanout_soa +
               out->nvcomp_temp + out->batch_metadata;
  return DAMACY_OK;
}
