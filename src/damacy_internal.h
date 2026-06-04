#pragma once

#include "batch_pool/batch_pool.h"
#include "damacy.h"
#include "gpu_budget/gpu_budget.h"
#include "lookahead/lookahead.h"
#include "numa/numa.h"
#include "planner/planner.h"
#include "prefetch/array_meta.h"
#include "prefetch/chunk_layout.h"
#include "prefetch/prefetch_cache.h"
#include "prefetch/prefetcher.h"
#include "prefetch/shard_index.h"
#include "render_job/render_job.h"
#include "scheduler/scheduler.h"
#include "store/metadata_store_async.h"
#include "store/store.h"
#include "wave/wave_pool.h"

#include <cuda.h>
#include <stdint.h>

// Only one handle is live at a time (the orchestrator's `handle` field).
struct damacy_batch
{
  struct damacy* d;
  uint16_t slot_idx;
  uint64_t batch_id;
};

// Lock order (outer → inner): scheduler_lock → prefetcher->lock → cache->lock
// → lookahead->lock. Code holding scheduler_lock may acquire any of the
// inner locks; the reverse causes deadlock. The prefetcher worker never
// takes scheduler_lock.
struct damacy
{
  struct damacy_config cfg;
  enum damacy_status failed_status;
  uint64_t next_batch_id;
  // Push-side cursor: indexes into the stream of samples ever pushed. The
  // prefetcher consumes the same global sample_seq order independently of
  // output batch placement.
  uint64_t pushed_samples;
  uint64_t page_alignment;
  int cuda_device;
  int retained_primary_device; // -1 = caller's ctx; else release at destroy
  CUcontext retained_primary;  // pushed per-call by ctx_guard when retained
  CUcontext worker_ctx;        // pushed by the worker on its first tick

  // Resolved NUMA placement plan; node<0 means "no pinning". Filled by
  // numa_init at create-time and shared with the store's io_queue and
  // the scheduler so worker threads can pin themselves on entry.
  struct numa_resolved numa;

  // GPU memory budgeting. Single source of truth for committed/max
  // across wave_pool, batch_pool, and the observe-and-grow paths.
  struct gpu_budget* budget;

  struct store* store_host;
  struct store* store_meta;
  struct store* store_meta_latency;
  struct metadata_store_async* store_meta_async;
  struct store* store_gds;
  struct planner* planner;

  // Async metadata cache fetchers use a separate store boundary so cache
  // misses do not occupy the prefetcher worker while waiting on small I/O.
  struct array_meta_async_fetcher array_meta_async_fetcher;
  struct shard_index_async_fetcher shard_index_async_fetcher;
  struct chunk_layout_async_fetcher chunk_layout_async_fetcher;
  struct prefetch_cache* array_meta_cache;
  struct prefetch_cache* shard_index_cache;
  struct prefetch_cache* chunk_layout_cache;
  struct prefetcher* prefetcher;
  // plan_ready_prefetch pops ready sample_seq intervals here, then
  // plan_reserve copies each segment into the target batch slot.
  struct prefetcher_ready* staging;

  struct damacy_lookahead lookahead;
  struct damacy_batch_pool batch_pool;
  struct render_job_pool render_jobs;
  // Owns the 4 streams + both waves; built once in damacy_create and
  // driven directly by the orchestrator (no per-call ctx building).
  struct wave_pool wave_pool;

  struct damacy_batch handle;
  struct damacy_stats stats;

  // Worker drives the pipeline; user-thread API coordinates via scheduler_lock.
  // Created last in damacy_create, torn down first in destroy_inner.
  struct scheduler* sched;
  int worker_ctx_pushed; // worker pushes worker_ctx on first tick; no matching
                         // pop.
};

// Tight enough to react to GPU event completion before wave-boundary
// gaps dominate; cuEventQuery is cheap so the worker isn't burning a core.
#define DAMACY_POP_POLL_NS 10000

struct ctx_guard
{
  int active;
};

enum damacy_status
ctx_guard_enter(struct damacy* d, struct ctx_guard* g);

void
ctx_guard_exit(struct ctx_guard* g);

int
damacy_scheduler_step(void* arg);

enum damacy_status
plan_reserve(struct damacy* self,
             uint16_t slot_idx,
             struct prefetcher_ready* ready,
             uint32_t n_ready,
             int close_batch);

enum damacy_status
plan_run(struct damacy* self, uint16_t slot_idx, float* out_elapsed_ms);

enum damacy_status
plan_commit(struct damacy* self,
            uint16_t slot_idx,
            enum damacy_status run_status,
            float elapsed_ms,
            int* changed);

enum damacy_status
plan_ready_prefetch(struct damacy* self, int close_partial, int* changed);
