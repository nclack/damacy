// Predicts wave-pool + per-batch GPU memory before any allocation.
// Lazy batch-output tensors (sized from the first AABB) are accounted
// separately at batch_pool_allocate time.
#pragma once

#include "damacy.h"

#include <stdint.h>

struct gpu_budget
{
  uint64_t dev_compressed;        // 2× host_per_wave (H2D mirror)
  uint64_t dev_decompressed;      // 2× dev_per_wave
  uint64_t dev_unshuffle_scratch; // 2× dev_per_wave
  uint64_t blosc1_meta;           // 2× per-wave parse + assemble metadata
  // 2× per-wave nvcomp fanout SOA + op arrays. The fanout slice of this
  // is sized off DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP — the per-wave
  // fanout may grow at runtime up to DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE
  // when a wave's actual substream count exceeds the initial floor. The
  // grow is not currently accounted against max_gpu_memory_bytes.
  // TODO(phase 5): track committed bytes on grow and enforce the cap.
  uint64_t fanout_soa;
  // 1× (zstd_temp + actual+status), pool-shared, sized off the initial
  // floor; observe-and-grow may raise this at runtime up to the
  // structural ceiling. Same phase-5 TODO as fanout_soa.
  uint64_t nvcomp_temp;
  uint64_t batch_metadata;        // 2× cfg.batch_size × sizeof(sample_plan)
  uint64_t total;
};

enum damacy_status gpu_budget_compute(const struct damacy_config* cfg,
                                      struct gpu_budget* out);
