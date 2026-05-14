// Runtime GPU-memory committer. Tracks bytes committed against a
// configured cap; single source of truth for "have we exceeded?" and
// "how many bytes are committed?" across wave_pool, batch_pool, and
// the observe-and-grow paths.
//
// The predictor that drives the cap-vs-need check at create time
// lives in wave/wave_budget.h (gpu_budget_predict +
// gpu_budget_breakdown). Splitting them avoids the cycle that would
// otherwise arise between gpu_budget (committer) and wave_budget
// (per-component byte math used by the grow paths).
#pragma once

#include "damacy.h"

#include <stdint.h>

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
