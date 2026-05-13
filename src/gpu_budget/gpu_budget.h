// Predicts wave-pool + per-batch GPU memory before any allocation.
// Lazy batch-output tensors (sized from the first AABB) are accounted
// separately at batch_pool_allocate time.
#pragma once

#include "damacy.h"

#include <stdint.h>

struct gpu_budget
{
  uint64_t dev_compressed;        // 2× host_slab_per_wave (H2D mirror)
  uint64_t dev_decompressed;      // 2× dev_decompressed_per_wave
  uint64_t dev_unshuffle_scratch; // 2× dev_decompressed_per_wave
  uint64_t blosc1_meta;           // 2× per-wave parse + assemble metadata
  // 2× per-wave nvcomp fanout SOA + op arrays. The fanout slice is
  // sized off DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP. The per-wave fanout
  // may grow at runtime up to DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE; the
  // wave grow path checks the delta against the configured budget
  // (Phase 5) and refuses to exceed it.
  uint64_t fanout_soa;
  // 1× (zstd_temp + actual+status), pool-shared, sized off the initial
  // floor; observe-and-grow may raise this at runtime up to the
  // structural ceiling. Same Phase-5 enforcement applies.
  uint64_t nvcomp_temp;
  uint64_t batch_metadata;        // 2× cfg.batch_size × sizeof(sample_plan)
  uint64_t total;
};

// Phase 5: per-wave host/device extents come from
// wave_pool_resolve_sizing, not the deprecated cfg buffer fields.
// max_chunk_uncompressed_bytes and batch_size still come from cfg.
enum damacy_status gpu_budget_compute(const struct damacy_config* cfg,
                                      uint64_t host_slab_per_wave,
                                      uint64_t dev_decompressed_per_wave,
                                      struct gpu_budget* out);
