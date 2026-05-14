// Two roles for the same memory accounting:
//
//   1. Predictor (struct gpu_budget_breakdown / gpu_budget_predict):
//      consulted at damacy_create / damacy_config_describe to size
//      wave-resident buffers before any allocation. Pure: no driver
//      calls beyond the nvcomp temp-bytes query.
//
//   2. Runtime committer (struct gpu_budget / try_commit / release):
//      one object owned by struct damacy; threaded through wave_pool,
//      batch_pool, and the observe-and-grow paths. Single source of
//      truth for "have we exceeded the cap?" and "how many bytes are
//      committed?". Replaces the open-coded checks that used to live
//      in three different modules.
#pragma once

#include "damacy.h"

#include <stdint.h>

// --- predictor ------------------------------------------------------------

struct gpu_budget_breakdown
{
  uint64_t dev_compressed;   // 2× host_slab_per_wave (H2D mirror)
  uint64_t dev_decompressed; // 2× dev_decompressed_per_wave
  uint64_t blosc1_meta;      // 2× per-wave parse + assemble metadata
  // 2× per-wave nvcomp fanout SOA + op arrays. The fanout slice is
  // sized off DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP. The per-wave fanout
  // may grow at runtime up to DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE; the
  // wave grow path checks the delta against the configured budget and
  // refuses to exceed it.
  uint64_t fanout_soa;
  // 1× (zstd_temp + actual+status), pool-shared, sized off the initial
  // floor; observe-and-grow may raise this at runtime up to the
  // structural ceiling. Same budget enforcement applies.
  uint64_t nvcomp_temp;
  uint64_t batch_metadata; // 2× cfg.batch_size × sizeof(sample_plan)
  uint64_t total;
};

// Per-wave host/device extents come from wave_pool_resolve_sizing.
// max_chunk_uncompressed_bytes and batch_size still come from cfg.
enum damacy_status
gpu_budget_predict(const struct damacy_config* cfg,
                   uint64_t host_slab_per_wave,
                   uint64_t dev_decompressed_per_wave,
                   struct gpu_budget_breakdown* out);

// --- runtime committer ---------------------------------------------------

// Tracks bytes committed against a configured cap. Single-threaded
// access only — callers serialize through scheduler_lock (or are on
// the worker tick under the same lock). No internal mutex.
struct gpu_budget;

// max_bytes is the resolved ceiling (default applied; non-zero by
// caller). Returns NULL on alloc failure.
struct gpu_budget*
gpu_budget_new(uint64_t max_bytes);

void
gpu_budget_destroy(struct gpu_budget* b);

// Add `bytes` to committed. Returns DAMACY_OK on success;
// DAMACY_OOM (with a log_error tagged by `tag`) if committed + bytes
// would exceed the cap. On OOM the committed counter is unchanged.
enum damacy_status
gpu_budget_try_commit(struct gpu_budget* b, uint64_t bytes, const char* tag);

// Unconditional add. Used at damacy_create when the resolver has
// already shown the geometry fits — calling try_commit there would
// duplicate the check.
void
gpu_budget_commit(struct gpu_budget* b, uint64_t bytes);

void
gpu_budget_release(struct gpu_budget* b, uint64_t bytes);

uint64_t
gpu_budget_committed(const struct gpu_budget* b);
uint64_t
gpu_budget_max(const struct gpu_budget* b);

// Test-only hook: overwrites committed and returns the prior value.
// Used by tests to drive the observe-and-grow OOM path without
// fabricating a workload that escapes the resolver's reservation.
uint64_t
gpu_budget_set_committed_for_test(struct gpu_budget* b, uint64_t v);
