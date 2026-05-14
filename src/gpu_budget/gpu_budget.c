#include "gpu_budget.h"

#include "damacy_config.h"
#include "log/log.h"
#include "planner/planner.h"
#include "wave/wave.h"

#include <stdlib.h>

// --- predictor ------------------------------------------------------------

enum damacy_status
gpu_budget_predict(const struct damacy_config* cfg,
                   uint64_t host_slab_per_wave,
                   uint64_t dev_decompressed_per_wave,
                   struct gpu_budget_breakdown* out)
{
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);

  // Single source of truth for one wave's device-resident bytes lives
  // in wave/. Pool-level totals are 2× because the orchestrator keeps
  // two waves in flight. The shared nvcomp scratch is queried
  // separately and counted once.
  struct wave_alloc_summary per_wave = { 0 };
  enum damacy_status s = wave_predict_bytes(
    host_slab_per_wave, dev_decompressed_per_wave, &per_wave);
  if (s != DAMACY_OK)
    return s;

  uint64_t nvcomp_temp = 0;
  s = wave_pool_shared_predict_bytes(
    dev_decompressed_per_wave, runtime_chunk_cap, &nvcomp_temp);
  if (s != DAMACY_OK)
    return s;

  out->dev_compressed = 2ull * per_wave.dev_compressed;
  out->dev_decompressed = 2ull * per_wave.dev_decompressed;
  out->blosc1_meta = 2ull * per_wave.blosc1_meta;
  out->fanout_soa = 2ull * per_wave.fanout_soa;
  out->nvcomp_temp = nvcomp_temp;
  out->batch_metadata =
    2ull * (uint64_t)cfg->batch_size * sizeof(struct sample_plan);
  out->total = out->dev_compressed + out->dev_decompressed + out->blosc1_meta +
               out->fanout_soa + out->nvcomp_temp + out->batch_metadata;
  return DAMACY_OK;
}

// --- runtime committer ---------------------------------------------------

struct gpu_budget
{
  uint64_t max_bytes;
  uint64_t committed;
};

struct gpu_budget*
gpu_budget_new(uint64_t max_bytes)
{
  struct gpu_budget* b = (struct gpu_budget*)calloc(1, sizeof(*b));
  if (!b)
    return NULL;
  b->max_bytes = max_bytes;
  b->committed = 0;
  return b;
}

void
gpu_budget_destroy(struct gpu_budget* b)
{
  free(b);
}

enum damacy_status
gpu_budget_try_commit(struct gpu_budget* b, uint64_t bytes, const char* tag)
{
  if (!b || bytes == 0)
    return DAMACY_OK;
  if (b->max_bytes > 0 && b->committed + bytes > b->max_bytes) {
    log_error("%s would exceed GPU budget: committed=%llu add=%llu cap=%llu",
              tag ? tag : "gpu_budget",
              (unsigned long long)b->committed,
              (unsigned long long)bytes,
              (unsigned long long)b->max_bytes);
    return DAMACY_OOM;
  }
  b->committed += bytes;
  return DAMACY_OK;
}

void
gpu_budget_commit(struct gpu_budget* b, uint64_t bytes)
{
  if (!b)
    return;
  b->committed += bytes;
}

void
gpu_budget_release(struct gpu_budget* b, uint64_t bytes)
{
  if (!b)
    return;
  if (bytes > b->committed)
    b->committed = 0;
  else
    b->committed -= bytes;
}

uint64_t
gpu_budget_committed(const struct gpu_budget* b)
{
  return b ? b->committed : 0;
}

uint64_t
gpu_budget_max(const struct gpu_budget* b)
{
  return b ? b->max_bytes : 0;
}

uint64_t
gpu_budget_set_committed_for_test(struct gpu_budget* b, uint64_t v)
{
  if (!b)
    return 0;
  const uint64_t prev = b->committed;
  b->committed = v;
  return prev;
}
