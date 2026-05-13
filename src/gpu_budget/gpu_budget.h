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
  uint64_t fanout_soa;            // 2× per-wave nvcomp fanout SOA + op arrays
  uint64_t nvcomp_temp;           // 2× (zstd_temp + actual+status)
  uint64_t batch_metadata;        // 2× cfg.batch_size × sizeof(sample_plan)
  uint64_t total;
};

enum damacy_status gpu_budget_compute(const struct damacy_config* cfg,
                                      struct gpu_budget* out);
