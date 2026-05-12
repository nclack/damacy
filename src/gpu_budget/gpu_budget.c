#include "gpu_budget.h"

#include "damacy_config.h"
#include "planner/planner.h"
#include "wave/wave.h"

enum damacy_status
gpu_budget_compute(const struct damacy_config* cfg, struct gpu_budget* out)
{
  const uint64_t host_per_wave = cfg->host_buffer_bytes / 2;
  const uint64_t dev_per_wave = cfg->device_buffer_bytes / 2;
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);

  // Single source of truth for one wave's device-resident bytes lives
  // in wave/. Pool-level totals are 2× because the orchestrator keeps
  // two waves in flight.
  struct wave_alloc_summary per_wave = { 0 };
  enum damacy_status s = wave_predict_bytes(
    host_per_wave, dev_per_wave, runtime_chunk_cap, &per_wave);
  if (s != DAMACY_OK)
    return s;

  out->dev_compressed = 2ull * per_wave.dev_compressed;
  out->dev_decompressed = 2ull * per_wave.dev_decompressed;
  out->dev_unshuffle_scratch = 2ull * per_wave.dev_unshuffle_scratch;
  out->blosc1_meta = 2ull * per_wave.blosc1_meta;
  out->fanout_soa = 2ull * per_wave.fanout_soa;
  out->nvcomp_temp = 2ull * per_wave.nvcomp_temp;
  out->batch_metadata =
    2ull * (uint64_t)cfg->batch_size * sizeof(struct sample_plan);
  out->total = out->dev_compressed + out->dev_decompressed +
               out->dev_unshuffle_scratch + out->blosc1_meta + out->fanout_soa +
               out->nvcomp_temp + out->batch_metadata;
  return DAMACY_OK;
}
