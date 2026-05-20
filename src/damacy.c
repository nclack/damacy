// damacy: streaming loader. Two batch slots and two wave slots in flight.
// A worker thread (src/scheduler) drives the pipeline; the user-thread
// API (push/pop/release/flush) coordinates via scheduler_lock + the
// scheduler's condition variable. Background I/O on the io_queue rounds
// out the threading model.

#include "damacy.h"

#include "batch_pool/batch_pool.h"
#include "damacy_config.h"
#include "damacy_stats.h"
#include "gpu_budget/gpu_budget.h"
#include "log/log.h"
#include "lookahead/lookahead.h"
#include "numa/numa.h"
#include "nvtx/nvtx.h"
#include "planner/planner.h"
#include "platform/platform.h"
#include "scheduler/scheduler.h"
#include "store/store.h"
#include "store/store_fs_gds.h"
#include "util/cuda_check.h"
#include "util/prelude.h"
#include "wave/wave_budget.h"
#include "wave/wave_pool.h"
#include "zarr/zarr_meta_cache.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_cache.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

// Internal handle returned by damacy_pop and consumed by damacy_release.
// Only one is live at a time (the orchestrator's `handle` field).
struct damacy_batch
{
  struct damacy* d;
  uint16_t slot_idx;
  uint64_t batch_id;
};

struct damacy
{
  struct damacy_config cfg;
  enum damacy_status failed_status;
  uint64_t next_batch_id;
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
  struct store* store_gds;
  struct zarr_meta_cache* meta_cache;
  struct zarr_shard_cache* shard_cache;
  struct planner* planner;

  struct damacy_lookahead lookahead;
  struct damacy_batch_pool batch_pool;
  // Owns the 4 streams + both waves; built once in damacy_create and
  // driven directly by the orchestrator (no per-call ctx building).
  struct wave_pool wave_pool;

  // Sample working set used while planning one batch.
  struct damacy_sample_slot* batch_samples;
  struct damacy_sample* batch_stage;

  struct damacy_batch handle;
  struct damacy_stats stats;

  // Worker drives the pipeline; user-thread API coordinates via scheduler_lock.
  // Created last in damacy_create, torn down first in destroy_inner.
  struct scheduler* sched;
  int worker_ctx_pushed; // worker pushes worker_ctx on first tick; no matching
                         // pop.
};

// Scheduler tick. 10 µs reacts to GPU event completion fast enough
// that wave-boundary gaps don't dominate; cuEventQuery is cheap so the
// worker isn't burning a core.
#define DAMACY_POP_POLL_NS 10000

// --- ctx guard ------------------------------------------------------------

struct ctx_guard
{
  int active;
};

static enum damacy_status
ctx_guard_enter(struct damacy* d, struct ctx_guard* g)
{
  g->active = 0;
  if (!d || d->retained_primary_device < 0)
    return DAMACY_OK;
  enum damacy_status s = DAMACY_CUDA;
  CU(Fail, cuCtxPushCurrent(d->retained_primary));
  g->active = 1;
  return DAMACY_OK;
Fail:
  return s;
}

static void
ctx_guard_exit(struct ctx_guard* g)
{
  if (g && g->active) {
    cuCtxPopCurrent(NULL);
    g->active = 0;
  }
}

// Lazy batch-output pool sizing + GPU-budget enforcement. Geometry is
// fixed by cfg->sample_shape at create-time; this only allocates the
// device buffers on first push. Idempotent.
static enum damacy_status
batch_pool_allocate(struct damacy* self)
{
  struct damacy_batch_pool* pool = &self->batch_pool;
  if (pool->allocated)
    return DAMACY_OK;

  enum damacy_status s =
    batch_pool_compute_layout(pool,
                              self->cfg.sample_shape,
                              self->cfg.sample_rank,
                              self->cfg.batch_size,
                              damacy_dtype_bpe(self->cfg.dtype));
  if (s != DAMACY_OK)
    return s;

  const uint64_t need = 2ull * pool->n_bytes;
  s = gpu_budget_try_commit(self->budget, need, "batch-output pool");
  if (s != DAMACY_OK)
    return s;

  s = batch_pool_alloc_dev(pool);
  if (s != DAMACY_OK) {
    gpu_budget_release(self->budget, need);
    return s;
  }
  return DAMACY_OK;
}

// 1 if `aabb` extents (assumed same rank as cfg) match cfg->sample_shape.
static int
sample_aabb_extents_match_cfg(const struct damacy_config* cfg,
                              const struct damacy_aabb* aabb)
{
  for (uint8_t d = 0; d < cfg->sample_rank; ++d) {
    int64_t extent = aabb->dims[d].end - aabb->dims[d].beg;
    if (extent != cfg->sample_shape[d])
      return 0;
  }
  return 1;
}

// --- planning -------------------------------------------------------------

static enum damacy_status
push_one(struct damacy* self, const struct damacy_sample* sample)
{
  if (!sample->uri)
    return DAMACY_INVAL;
  if (sample->aabb.rank == 0 || sample->aabb.rank > DAMACY_MAX_RANK)
    return DAMACY_RANK;

  struct zarr_metadata meta;
  enum damacy_status ms =
    zarr_meta_cache_get(self->meta_cache, sample->uri, &meta);
  if (ms != DAMACY_OK)
    return ms;

  if (!cast_path_supported(self->cfg.dtype, meta.dtype))
    return DAMACY_DTYPE;
  if (sample->aabb.rank != meta.rank)
    return DAMACY_RANK;
  if (sample->aabb.rank != self->cfg.sample_rank)
    return DAMACY_RANK;
  if (!sample_aabb_extents_match_cfg(&self->cfg, &sample->aabb))
    return DAMACY_INVAL;

  if (lookahead_push(&self->lookahead, sample))
    return DAMACY_OOM;
  return DAMACY_OK;
}

// --- plan: reserve [locked] → run [unlocked] → commit [locked] -------------
// run does the planner CPU work + sample_plans H2D off the
// scheduler_lock. The slot sits in BATCH_PLANNING for the window so
// pop (any_batch_in_flight) and flush (any_batch_planning) wait.

static enum damacy_status
plan_reserve(struct damacy* self, uint16_t slot_idx, uint32_t n_samples)
{
  if (n_samples == 0)
    return DAMACY_OK;
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  if (slot->state != BATCH_FREE)
    return DAMACY_INVAL;

  lookahead_drain(&self->lookahead, self->batch_samples, n_samples);

  enum damacy_status status = batch_pool_allocate(self);
  if (status != DAMACY_OK) {
    for (uint32_t i = 0; i < n_samples; ++i)
      sample_slot_clear(&self->batch_samples[i]);
    self->failed_status = status;
    return status;
  }
  for (uint32_t i = 0; i < n_samples; ++i) {
    self->batch_stage[i].uri = self->batch_samples[i].uri;
    self->batch_stage[i].aabb = self->batch_samples[i].aabb;
  }
  slot->n_samples = n_samples;
  slot->state = BATCH_PLANNING;
  return DAMACY_OK;
}

// Returns elapsed ms via *out_elapsed_ms so the metric can be recorded
// in plan_commit, which runs under scheduler_lock (stats are read by
// damacy_stats_get under the same lock).
static enum damacy_status
plan_run(struct damacy* self, uint16_t slot_idx, float* out_elapsed_ms)
{
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  CHECK(InvalidArg, slot->state == BATCH_PLANNING);
  struct planner_output plan_out = {
    .read_ops = slot->read_ops,
    .read_ops_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .chunk_plans = slot->chunk_plans,
    .chunk_plans_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .sample_plans = slot->sample_plans,
    .sample_plans_cap = self->cfg.batch_size,
    .read_op_groups = slot->read_op_groups,
    .read_op_groups_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .paths = &slot->paths,
  };
  struct platform_clock plan_clock = { 0 };
  platform_toc(&plan_clock);
  enum damacy_status status = planner_plan(self->planner,
                                           self->batch_stage,
                                           slot->n_samples,
                                           slot_idx,
                                           self->batch_pool.strides,
                                           self->batch_pool.rank,
                                           &plan_out);
  *out_elapsed_ms = platform_toc(&plan_clock) * 1000.0f;
  if (status != DAMACY_OK)
    return status;
  slot->n_chunks = plan_out.n_chunk_plans;
  slot->n_chunks_to_load = plan_out.n_chunks_to_load;
  slot->n_loads_issued = plan_out.n_loads_issued;
  slot->n_sample_plans = plan_out.n_sample_plans;
  slot->n_read_op_groups = plan_out.n_read_op_groups;
  if (plan_out.n_sample_plans > 0) {
    if (cuMemcpyHtoD(CUDPTR(slot->d_sample_plans),
                     slot->sample_plans,
                     (size_t)plan_out.n_sample_plans *
                       sizeof(struct sample_plan)) != CUDA_SUCCESS)
      return DAMACY_CUDA;
  }
  return DAMACY_OK;
InvalidArg:
  return DAMACY_INVAL;
}

// *changed (nullable; flush passes NULL): OR-set on every BATCH transition.
static enum damacy_status
plan_commit(struct damacy* self,
            uint16_t slot_idx,
            enum damacy_status run_status,
            float elapsed_ms,
            int* changed)
{
  metric_record(&self->stats.plan, elapsed_ms, 0, 0);
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  for (uint32_t i = 0; i < slot->n_samples; ++i)
    sample_slot_clear(&self->batch_samples[i]);
  if (run_status != DAMACY_OK) {
    slot->state = BATCH_FREE;
    slot->n_samples = 0;
    slot->deferred_release_pending = 0;
    self->failed_status = run_status;
    if (changed)
      *changed = 1;
    return run_status;
  }
  slot->n_chunks_dispatched = 0;
  slot->n_groups_dispatched = 0;
  slot->chunks_remaining = (int32_t)slot->n_chunks;
  slot->batch_id = self->next_batch_id++;
  slot->state = BATCH_FILLING;
  self->stats.chunks_planned += slot->n_chunks;
  self->stats.chunks_to_load += slot->n_chunks_to_load;
  self->stats.reads_issued += slot->n_loads_issued;
  if (changed)
    *changed = 1;

  if (slot->n_chunks == 0) {
    // Degenerate batch: zero the output and skip to READY. cuMemsetD8
    // runs on the legacy null stream; if a deferred release wait is
    // pending on stream_post, sync it first so the memset can't race
    // a still-in-flight consumer read.
    if (slot->deferred_release_pending) {
      cuStreamSynchronize(self->wave_pool.stream_post);
      slot->deferred_release_pending = 0;
    }
    if (cuMemsetD8(CUDPTR(slot->dev_ptr), 0, self->batch_pool.n_bytes) !=
        CUDA_SUCCESS) {
      slot->state = BATCH_FREE;
      self->failed_status = DAMACY_CUDA;
      return DAMACY_CUDA;
    }
    slot->state = BATCH_READY;
  }
  slot->deferred_release_pending = 0;
  return DAMACY_OK;
}

// Drains lookahead-planned batches into free host_slab_slots, planning
// a fresh batch when no FILLING batch has chunks left to peel. Stops
// when there are no free slots, no batches with work, and no room to
// plan more.
static enum damacy_status
kick_peel_into_free_slots(struct damacy* self, int* changed)
{
  for (;;) {
    int target_slot = find_filling_slot_with_work(&self->batch_pool);
    if (target_slot < 0) {
      int free_slot = find_free_batch_slot(&self->batch_pool);
      if (free_slot < 0)
        break;
      if (self->lookahead.size < self->cfg.batch_size)
        break;
      enum damacy_status s =
        plan_reserve(self, (uint16_t)free_slot, self->cfg.batch_size);
      if (s != DAMACY_OK)
        return s;
      scheduler_unlock(self->sched);
      float plan_ms = 0.f;
      enum damacy_status rs = plan_run(self, (uint16_t)free_slot, &plan_ms);
      scheduler_lock(self->sched);
      s = plan_commit(self, (uint16_t)free_slot, rs, plan_ms, changed);
      if (s != DAMACY_OK)
        return s;
      continue;
    }

    enum damacy_status err = DAMACY_OK;
    struct wave_pool_peel_ticket t =
      wave_pool_peel_reserve(&self->wave_pool, (uint16_t)target_slot, &err);
    if (err != DAMACY_OK)
      return err;
    if (t.slot_idx < 0)
      break;
    damacy_nvtx_range_pushf("peel/slot%d", t.slot_idx);
    scheduler_unlock(self->sched);
    struct store_event ev = wave_pool_peel_submit(&self->wave_pool, &t);
    scheduler_lock(self->sched);
    enum damacy_status s =
      wave_pool_peel_commit(&self->wave_pool, &t, ev, changed);
    damacy_nvtx_range_pop();
    if (s != DAMACY_OK)
      return s;
    if (!any_slot_free(&self->wave_pool))
      break;
  }
  return DAMACY_OK;
}

// --- scheduler ------------------------------------------------------------

// One scheduler tick, under scheduler_lock. Lazy ctx push on first call.
// *changed contract (authoritative): every transition site
// (wave_pool_advance, plan_commit, wave_pool_peel_commit) OR-sets it on
// a real state transition; the worker broadcasts iff non-zero.
static int
damacy_scheduler_step(void* arg)
{
  struct damacy* self = (struct damacy*)arg;
  if (!self->worker_ctx_pushed) {
    if (self->worker_ctx)
      cuCtxPushCurrent(self->worker_ctx);
    self->worker_ctx_pushed = 1;
  }
  self->stats.worker_steps++;
  // Wake any pop waiter so it can observe the latched error.
  if (self->failed_status != DAMACY_OK)
    return 1;

  int changed = 0;
  enum damacy_status r = wave_pool_advance(&self->wave_pool, &changed);
  if (r == DAMACY_OK && self->failed_status == DAMACY_OK)
    r = kick_peel_into_free_slots(self, &changed);
  if (r != DAMACY_OK && self->failed_status == DAMACY_OK) {
    self->failed_status = r;
    return 1;
  }
  return changed;
}

// --- public API: create / destroy ----------------------------------------

// Single teardown list shared by every destroy path. cuda_skip=1 leaks
// CUDA-owned state and skips driver calls; CPU heap is always released.
// wave_pool first: its destroy syncs streams before the downstream
// batch_pool / planner free what those streams touched.
static void
destroy_inner(struct damacy* self, int cuda_skip)
{
  if (!self)
    return;

  // Stop the worker first so its accesses retire before we free.
  scheduler_destroy(self->sched);
  self->sched = NULL;

  wave_pool_destroy(&self->wave_pool, cuda_skip);

  free(self->batch_stage);
  self->batch_stage = NULL;
  free(self->batch_samples);
  self->batch_samples = NULL;
  lookahead_destroy(&self->lookahead);
  batch_pool_destroy(&self->batch_pool, cuda_skip);

  planner_destroy(self->planner);
  self->planner = NULL;
  zarr_shard_cache_destroy(self->shard_cache);
  self->shard_cache = NULL;
  zarr_meta_cache_destroy(self->meta_cache);
  self->meta_cache = NULL;
  store_destroy(self->store_gds);
  self->store_gds = NULL;
  store_destroy(self->store_host);
  self->store_host = NULL;
  gpu_budget_destroy(self->budget);
  self->budget = NULL;
}

enum damacy_status
damacy_create(const struct damacy_config* cfg, struct damacy** out)
{
  enum damacy_status s = DAMACY_INVAL;
  struct damacy* self = NULL;
  struct ctx_guard cg = { 0 };

  CHECK_SILENT(InvalidArg, out);
  *out = NULL;

  s = validate_config(cfg);
  if (s != DAMACY_OK)
    return s;

  s = DAMACY_OOM;
  self = (struct damacy*)calloc(1, sizeof(*self));
  CHECK(Fail, self);
  self->cfg = *cfg;
  self->failed_status = DAMACY_OK;
  self->page_alignment = (uint64_t)platform_page_alignment();
  // -1 sentinel set before any goto Fail: 0 from calloc is a valid CUdevice.
  self->retained_primary_device = -1;
  self->retained_primary = NULL;
  stats_init(&self->stats);

  s = DAMACY_CUDA;
  CU(Fail, cuInit(0));

  CUcontext caller_ctx = NULL;
  CU(Fail, cuCtxGetCurrent(&caller_ctx));
  if (cfg->device >= 0) {
    if (caller_ctx) {
      CUdevice cur_dev;
      CU(Fail, cuCtxGetDevice(&cur_dev));
      if ((int)cur_dev != cfg->device) {
        s = DAMACY_INVAL;
        log_error("damacy_create: Config.device=%d but a CUcontext is "
                  "already current on device %d — likely a missing "
                  "cuCtxSetCurrent / torch.cuda.set_device(%d)",
                  cfg->device,
                  (int)cur_dev,
                  cfg->device);
        goto Fail;
      }
    }
    CUdevice dev;
    CU(Fail, cuDeviceGet(&dev, cfg->device));
    CUcontext primary = NULL;
    CU(Fail, cuDevicePrimaryCtxRetain(&primary, dev));
    self->retained_primary_device = cfg->device;
    self->retained_primary = primary;
    self->worker_ctx = primary;
    self->cuda_device = cfg->device;
  } else {
    if (!caller_ctx) {
      log_error("damacy_create: no CUcontext is current on calling thread");
      s = DAMACY_INVAL;
      goto Fail;
    }
    CUdevice dev;
    CU(Fail, cuCtxGetDevice(&dev));
    self->cuda_device = (int)dev;
    self->worker_ctx = caller_ctx;
  }

  s = ctx_guard_enter(self, &cg);
  if (s != DAMACY_OK)
    goto Fail;

  // Resolve the GPU's host-NUMA node now that we know the CUdevice. The
  // resolved plan is consumed by the wave_pool_init scope below and by
  // the store / scheduler worker threads at startup.
  {
    CUdevice cu_dev;
    if (cuCtxGetDevice(&cu_dev) == CUDA_SUCCESS) {
      numa_init(
        cfg->tuning.numa_strategy, cfg->tuning.numa_node, cu_dev, &self->numa);
    } else {
      // Should never happen — ctx_guard_enter just pushed our ctx, or
      // the caller's ctx is live. Be safe; treat as disabled.
      self->numa.node = -1;
    }
  }

  const uint64_t max_gpu = cfg->tuning.max_gpu_memory_bytes;
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  self->budget = gpu_budget_new(max_gpu);
  if (!self->budget) {
    s = DAMACY_OOM;
    goto Fail;
  }

  // Carve out the double-buffered batch-output pool before sizing
  // wave-resident buffers; the resolver is greedy, so without this
  // reservation the lazy pool has no room at first push.
  uint64_t pool_reserve = 0;
  {
    uint64_t pool_bytes = 0;
    CHECK(Fail,
          (s = resolve_sample_volume_bytes(cfg, &pool_bytes)) == DAMACY_OK);
    pool_reserve = 2ull * pool_bytes;
  }
  if (pool_reserve >= max_gpu) {
    log_error("damacy: batch-output pool reserve=%llu >= "
              "max_gpu_memory_bytes=%llu; nothing left for wave-resident "
              "buffers (sample_shape × batch_size × dtype_bpe × 2 exceeds cap)",
              (unsigned long long)pool_reserve,
              (unsigned long long)max_gpu);
    s = DAMACY_BUDGET;
    goto Fail;
  }
  const uint64_t resolver_budget = max_gpu - pool_reserve;

  const uint32_t resolved_max_chunks_per_wave =
    resolve_max_chunks_per_wave(cfg);
  const uint16_t resolved_max_substreams_per_chunk =
    resolve_max_substreams_per_chunk(cfg);
  struct wave_pool_sizing sizing = { 0 };
  s = wave_pool_resolve_sizing(resolved_max_chunks_per_wave,
                               resolved_max_substreams_per_chunk,
                               resolver_budget,
                               runtime_chunk_cap,
                               cfg->batch_size,
                               &sizing);
  if (s != DAMACY_OK)
    goto Fail;
  {
    struct gpu_budget_breakdown predicted = { 0 };
    s = gpu_budget_predict(cfg,
                           sizing.host_slab_per_wave,
                           sizing.dev_decompressed_per_wave,
                           &predicted);
    if (s != DAMACY_OK)
      goto Fail;
    gpu_budget_commit(self->budget, predicted.total);
    log_debug("damacy: resolved geometry from max_gpu_memory_bytes=%llu "
              "(pool_reserve=%llu, resolver_budget=%llu): "
              "host_slab_per_wave=%llu dev_decompressed_per_wave=%llu "
              "initial_nvcomp_temp=%llu predicted_total=%llu",
              (unsigned long long)max_gpu,
              (unsigned long long)pool_reserve,
              (unsigned long long)resolver_budget,
              (unsigned long long)sizing.host_slab_per_wave,
              (unsigned long long)sizing.dev_decompressed_per_wave,
              (unsigned long long)predicted.nvcomp_temp,
              (unsigned long long)predicted.total);
    // Resolver guarantees this fits; assert defensively in case the
    // accounting drifts. A breach here is a bug, not user input.
    if (gpu_budget_committed(self->budget) > gpu_budget_max(self->budget)) {
      log_error(
        "damacy: post-resolve drift: total=%llu cap=%llu "
        "(dev_compressed=%llu dev_decompressed=%llu "
        "blosc1_meta=%llu fanout_soa=%llu nvcomp_temp=%llu batch_meta=%llu)",
        (unsigned long long)predicted.total,
        (unsigned long long)gpu_budget_max(self->budget),
        (unsigned long long)predicted.dev_compressed,
        (unsigned long long)predicted.dev_decompressed,
        (unsigned long long)predicted.blosc1_meta,
        (unsigned long long)predicted.fanout_soa,
        (unsigned long long)predicted.nvcomp_temp,
        (unsigned long long)predicted.batch_metadata);
      s = DAMACY_BUDGET;
      goto Fail;
    }
  }

  uint8_t want_gds = resolve_enable_gds(cfg);

  s = DAMACY_OOM;

  // Sample.uri is absolute; fs store joins root+key, so empty root is a
  // pass-through.
  {
    struct store_fs_config sc = {
      .root = "",
      .nthreads = (int)cfg->tuning.n_io_threads,
      .affinity = &self->numa,
    };
    self->store_host = store_fs_create(&sc);
    CHECK(Fail, self->store_host);
  }
  if (want_gds) {
    struct store_fs_gds_config sc = {
      .root = "",
      .fd_cache_capacity = 0,
    };
    self->store_gds = store_fs_gds_create(&sc);
    if (!self->store_gds) {
      s = DAMACY_INVAL;
      goto Fail;
    }
  }

  self->meta_cache =
    zarr_meta_cache_create(self->store_host, cfg->tuning.n_zarrs_meta_cache);
  CHECK(Fail, self->meta_cache);
  self->shard_cache =
    zarr_shard_cache_create(self->store_host, cfg->tuning.n_shards_meta_cache);
  CHECK(Fail, self->shard_cache);

  for (int b = 0; b < 2; ++b)
    CHECK(Fail,
          batch_slot_init(&self->batch_pool.slots[b], cfg->batch_size) == 0);

  s = DAMACY_OOM;
  // Pin the calling thread to the GPU's NUMA node for the duration of
  // wave_pool_init so first-touch of pinned-host slabs + per-wave
  // scratch lands on the right node. Restored immediately after.
  {
    struct platform_cpu_mask saved_aff;
    numa_scope_enter(&self->numa, &saved_aff);
    int wp_rc = wave_pool_init(&self->wave_pool,
                               &self->batch_pool,
                               want_gds ? self->store_gds : self->store_host,
                               &self->stats,
                               cfg->dtype,
                               resolve_host_buffer_waves(cfg),
                               resolved_max_chunks_per_wave,
                               resolved_max_substreams_per_chunk,
                               sizing.host_slab_per_wave,
                               sizing.dev_decompressed_per_wave,
                               runtime_chunk_cap,
                               (int)want_gds,
                               cfg->debug.bypass_decode,
                               self->budget);
    numa_scope_exit(&saved_aff);
    CHECK(Fail, wp_rc == 0);
  }
  if (want_gds)
    log_info("damacy: compressed reads via cuFile / GDS (skip bulk H2D)");

  struct planner_config pcfg = {
    .meta_cache = self->meta_cache,
    .shard_cache = self->shard_cache,
    .page_alignment = self->page_alignment,
    .max_chunk_uncompressed_bytes = runtime_chunk_cap,
    .read_op_max_bytes = resolve_max_read_op_bytes(cfg),
    .max_chunks_per_wave = resolved_max_chunks_per_wave,
    .max_substreams_per_chunk = resolved_max_substreams_per_chunk,
  };
  CHECK(Fail, planner_create(&pcfg, &self->planner) == DAMACY_OK);

  CHECK(Fail,
        lookahead_init(&self->lookahead,
                       cfg->lookahead_batches * cfg->batch_size) == 0);

  self->batch_samples = (struct damacy_sample_slot*)calloc(
    cfg->batch_size, sizeof(struct damacy_sample_slot));
  CHECK(Fail, self->batch_samples);
  self->batch_stage = (struct damacy_sample*)calloc(
    cfg->batch_size, sizeof(struct damacy_sample));
  CHECK(Fail, self->batch_stage);

  self->handle.d = self;

  // Spawn the worker last — everything it touches must already exist.
  self->sched = scheduler_create(
    damacy_scheduler_step, self, DAMACY_POP_POLL_NS, &self->numa);
  if (!self->sched) {
    s = DAMACY_OOM;
    goto Fail;
  }

  ctx_guard_exit(&cg);
  *out = self;
  return DAMACY_OK;

Fail:
  if (self) {
    // destroy_inner under the pushed ctx, then pop, then release primary.
    destroy_inner(self, 0);
    ctx_guard_exit(&cg);
    if (self->retained_primary_device >= 0)
      cuDevicePrimaryCtxRelease((CUdevice)self->retained_primary_device);
    free(self);
  }
  return s;

InvalidArg:
  return DAMACY_INVAL;
}

void
damacy_destroy(struct damacy* self)
{
  if (!self)
    return;

  struct ctx_guard cg = { 0 };
  enum damacy_status gs = ctx_guard_enter(self, &cg);
  if (gs != DAMACY_OK) {
    // Ctx no longer pushable (device reset, primary released elsewhere):
    // leak CUDA state and walk the teardown list with skip=1.
    log_warn("damacy_destroy: ctx_guard_enter failed (status=%d); "
             "leaking CUDA resources",
             (int)gs);
    destroy_inner(self, 1);
  } else {
    destroy_inner(self, 0);
    ctx_guard_exit(&cg);
  }
  if (self->retained_primary_device >= 0)
    cuDevicePrimaryCtxRelease((CUdevice)self->retained_primary_device);
  free(self);
}

int
damacy_get_device(const struct damacy* d)
{
  return d ? d->cuda_device : -1;
}

// --- push -----------------------------------------------------------------

struct damacy_push_result
damacy_push(struct damacy* self, struct damacy_sample_slice samples)
{
  struct damacy_push_result r = { .unconsumed = samples, .status = DAMACY_OK };
  if (!self) {
    r.status = DAMACY_INVAL;
    return r;
  }
  if (samples.beg > samples.end) {
    r.status = DAMACY_INVAL;
    return r;
  }
  // No ctx_guard: push touches no CUDA. The lock guards lookahead +
  // meta/shape checks against the worker's plan_into_slot drain.
  scheduler_lock(self->sched);
  if (self->failed_status != DAMACY_OK) {
    r.status = DAMACY_SHUTDOWN;
    goto Done;
  }
  for (const struct damacy_sample* s = samples.beg; s != samples.end; ++s) {
    if (self->lookahead.size == self->lookahead.cap) {
      r.unconsumed.beg = s;
      r.status = DAMACY_AGAIN;
      goto Done;
    }
    enum damacy_status ps = push_one(self, s);
    if (ps != DAMACY_OK) {
      r.unconsumed.beg = s;
      r.status = ps;
      goto Done;
    }
  }
  r.unconsumed.beg = samples.end;
Done:
  scheduler_unlock(self->sched);
  return r;
}

// --- pop ------------------------------------------------------------------

enum damacy_status
damacy_pop(struct damacy* self, struct damacy_batch** out)
{
  CHECK_SILENT(InvalidArg, self);
  CHECK_SILENT(InvalidArg, out);
  *out = NULL;

  // No ctx_guard: pop only touches batch-slot state. CUDA stays on the worker.
  damacy_nvtx_range_push("damacy_pop");
  enum damacy_status r;
  scheduler_lock(self->sched);
  for (;;) {
    if (self->failed_status != DAMACY_OK) {
      r = self->failed_status;
      goto Done;
    }
    int slot_idx = find_oldest_ready_slot(&self->batch_pool);
    if (slot_idx >= 0) {
      struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
      slot->state = BATCH_HELD;
      self->handle.slot_idx = (uint16_t)slot_idx;
      self->handle.batch_id = slot->batch_id;
      self->stats.batches_emitted++;
      *out = &self->handle;
      r = DAMACY_OK;
      goto Done;
    }
    if (!any_wave_in_flight(&self->wave_pool) &&
        !any_slot_in_flight(&self->wave_pool) &&
        !any_batch_in_flight(&self->batch_pool) &&
        self->lookahead.size < self->cfg.batch_size) {
      r = DAMACY_AGAIN;
      goto Done;
    }
    struct platform_clock wait_clock = { 0 };
    platform_toc(&wait_clock);
    SCHEDULER_WAIT_DIAG(self->sched, 5000);
    metric_record(
      &self->stats.pop_wait, platform_toc(&wait_clock) * 1000.0f, 0, 0);
  }

Done:
  scheduler_unlock(self->sched);
  damacy_nvtx_range_pop();
  return r;

InvalidArg:
  return DAMACY_INVAL;
}

void
damacy_release(struct damacy* self, struct damacy_batch* b)
{
  if (!self || !b)
    return;
  if (b != &self->handle) {
    log_warn("damacy_release: foreign handle (not the active batch)");
    return;
  }
  uint16_t s = b->slot_idx;
  if (s >= 2) {
    log_warn("damacy_release: slot_idx=%u out of range", (unsigned)s);
    return;
  }
  scheduler_lock(self->sched);
  if (self->batch_pool.slots[s].state != BATCH_HELD) {
    log_warn("damacy_release: slot %u not HELD (state=%d); double release?",
             (unsigned)s,
             (int)self->batch_pool.slots[s].state);
    scheduler_unlock(self->sched);
    return;
  }
  self->batch_pool.slots[s].state = BATCH_FREE;
  self->batch_pool.slots[s].n_chunks = 0;
  self->batch_pool.slots[s].n_chunks_dispatched = 0;
  self->batch_pool.slots[s].n_groups_dispatched = 0;
  self->batch_pool.slots[s].deferred_release_pending = 0;
  scheduler_unlock(self->sched);
}

enum damacy_status
damacy_release_event(struct damacy* self, struct damacy_batch* b, void* event)
{
  // NULL event → degenerate to the immediate-release path.
  if (!event) {
    damacy_release(self, b);
    return DAMACY_OK;
  }
  if (!self || !b)
    return DAMACY_INVAL;
  if (b != &self->handle) {
    log_warn("damacy_release_event: foreign handle (not the active batch)");
    return DAMACY_INVAL;
  }
  uint16_t s = b->slot_idx;
  if (s >= 2) {
    log_warn("damacy_release_event: slot_idx=%u out of range", (unsigned)s);
    return DAMACY_INVAL;
  }

  // Push the retained-primary context so cuStreamWaitEvent / cuEventRecord
  // land on the right device when the caller is on another thread.
  struct ctx_guard cg = { 0 };
  enum damacy_status r = ctx_guard_enter(self, &cg);
  if (r != DAMACY_OK)
    return r;

  scheduler_lock(self->sched);
  struct damacy_batch_slot* slot = &self->batch_pool.slots[s];
  if (slot->state != BATCH_HELD) {
    log_warn(
      "damacy_release_event: slot %u not HELD (state=%d); double release?",
      (unsigned)s,
      (int)slot->state);
    r = DAMACY_INVAL;
    goto Done;
  }

  // Defer reuse on stream_post (where assemble writes the slot's
  // dev_ptr). stream_post is FIFO, so any subsequent kick_assemble — for
  // either slot — picks up this wait. The flag is read by plan_into_slot
  // to host-sync stream_post before its sync cuMemsetD8 (which targets the
  // legacy null stream and would otherwise race).
  if (cuStreamWaitEvent(self->wave_pool.stream_post, (CUevent)event, 0) !=
      CUDA_SUCCESS) {
    // Deferred wait couldn't be installed; fall back to immediate release
    // so the slot doesn't leak (caller would block forever in pop).
    slot->state = BATCH_FREE;
    slot->n_chunks = 0;
    slot->n_chunks_dispatched = 0;
    slot->n_groups_dispatched = 0;
    slot->deferred_release_pending = 0;
    r = DAMACY_CUDA;
    goto Done;
  }
  slot->deferred_release_pending = 1;

  slot->state = BATCH_FREE;
  slot->n_chunks = 0;
  slot->n_chunks_dispatched = 0;
  slot->n_groups_dispatched = 0;
  r = DAMACY_OK;

Done:
  scheduler_unlock(self->sched);
  ctx_guard_exit(&cg);
  return r;
}

// --- flush ----------------------------------------------------------------

enum damacy_status
damacy_flush(struct damacy* self)
{
  if (!self)
    return DAMACY_INVAL;

  // plan_into_slot below issues cuMemcpyHtoD on this thread; push the
  // retained primary so the call lands in the right context.
  struct ctx_guard cg = { 0 };
  enum damacy_status r = ctx_guard_enter(self, &cg);
  if (r != DAMACY_OK)
    return r;

  scheduler_lock(self->sched);
  if (self->failed_status != DAMACY_OK) {
    r = self->failed_status;
    goto Done;
  }

  // Worker only plans at full batch_size; flush emits the truncated tail.
  if (self->lookahead.size > 0 && self->lookahead.size < self->cfg.batch_size) {
    // Wait until a slot is FREE *and* no other plan is in progress.
    // plan_run releases the lock so a worker plan could be mid-flight;
    // both predicates are re-evaluated together under the lock on every
    // wake (cv_wait yields only inside the loop body), so we exit when
    // both hold simultaneously and the subsequent find_free_batch_slot
    // read is current.
    while ((find_free_batch_slot(&self->batch_pool) < 0 ||
            any_batch_planning(&self->batch_pool)) &&
           self->failed_status == DAMACY_OK)
      SCHEDULER_WAIT_DIAG(self->sched, 5000);
    if (self->failed_status != DAMACY_OK) {
      r = self->failed_status;
      goto Done;
    }
    int free_slot = find_free_batch_slot(&self->batch_pool);
    uint32_t n = self->lookahead.size;
    r = plan_reserve(self, (uint16_t)free_slot, n);
    if (r != DAMACY_OK)
      goto Done;
    scheduler_unlock(self->sched);
    float plan_ms = 0.f;
    enum damacy_status rs = plan_run(self, (uint16_t)free_slot, &plan_ms);
    scheduler_lock(self->sched);
    r = plan_commit(self, (uint16_t)free_slot, rs, plan_ms, NULL);
    if (r != DAMACY_OK)
      goto Done;
    self->stats.batches_truncated++;
  }

  // any_slot_in_flight catches the SLOT_PEELING window: peel_reserve has
  // already bumped n_chunks_dispatched (so find_oldest_filling_slot can
  // be -1) but peel_submit hasn't run yet, no wave exists yet — without
  // this check, flush would return while the worker still has unposted
  // IO to submit. damacy_pop's AGAIN gate keeps the same invariant.
  struct platform_clock flush_clock = { 0 };
  platform_toc(&flush_clock);
  while ((any_wave_in_flight(&self->wave_pool) ||
          any_slot_in_flight(&self->wave_pool) ||
          find_oldest_filling_slot(&self->batch_pool) >= 0 ||
          any_batch_planning(&self->batch_pool)) &&
         self->failed_status == DAMACY_OK)
    SCHEDULER_WAIT_DIAG(self->sched, 5000);
  metric_record(
    &self->stats.flush_wait, platform_toc(&flush_clock) * 1000.0f, 0, 0);
  r = self->failed_status != DAMACY_OK ? self->failed_status : DAMACY_OK;

Done:
  scheduler_unlock(self->sched);
  ctx_guard_exit(&cg);
  return r;
}

// --- batch info / stats ---------------------------------------------------

void
damacy_batch_info(const struct damacy_batch* b, struct damacy_batch_info* out)
{
  if (!out)
    return;
  memset(out, 0, sizeof(*out));
  if (!b || !b->d || b->slot_idx >= 2)
    return;
  const struct damacy* self = b->d;
  const struct damacy_batch_slot* slot = &self->batch_pool.slots[b->slot_idx];
  if (slot->state != BATCH_HELD)
    return;
  out->device_ptr = slot->dev_ptr;
  out->rank = self->batch_pool.rank;
  out->dtype = self->cfg.dtype;
  out->ready_stream = (void*)self->wave_pool.stream_decode;
  out->batch_id = slot->batch_id;
  for (uint8_t d = 0; d < self->batch_pool.rank; ++d)
    out->shape[d] = self->batch_pool.shape[d];
  // shape[0] reflects actual sample count (< batch_size for flushed partials).
  out->shape[0] = (int64_t)slot->n_samples;
}

void
damacy_stats_get(const struct damacy* self, struct damacy_stats* out)
{
  if (!out)
    return;
  if (!self) {
    memset(out, 0, sizeof(*out));
    return;
  }
  // scheduler_lock guards every metric_record write; without it the
  // struct copy below races every plan/pop_wait/flush_wait update. The
  // mutex doesn't change observable state, so the const cast is safe.
  struct damacy* m = (struct damacy*)self;
  scheduler_lock(m->sched);
  *out = m->stats;
  out->gpu_bytes_committed = gpu_budget_committed(m->budget);
  scheduler_unlock(m->sched);
  // Cache stats have their own internal mutex; safe outside scheduler_lock.
  if (m->meta_cache) {
    struct zarr_meta_cache_stats ms;
    zarr_meta_cache_stats_get(m->meta_cache, &ms);
    out->zarr_meta_hits = ms.counters.hits;
    out->zarr_meta_misses = ms.counters.misses;
  }
  if (m->shard_cache) {
    struct zarr_shard_cache_stats ss;
    zarr_shard_cache_stats_get(m->shard_cache, &ss);
    out->shard_idx_hits = ss.counters.hits;
    out->shard_idx_misses = ss.counters.misses;
  }
}

void
damacy_stats_reset(struct damacy* self)
{
  if (!self)
    return;
  stats_init(&self->stats);
}

// Test-only hook. Overwrites gpu_bytes_committed so unit tests can drive
// the observe-and-grow OOM path without having to fabricate a workload
// that escapes the resolver's worst-case reservation. Returns the prior
// value. Not declared in damacy.h — tests forward-declare it with
// extern; production code must not call it.
uint64_t
damacy_set_gpu_bytes_committed_for_test(struct damacy* self, uint64_t v)
{
  if (!self)
    return 0;
  return gpu_budget_set_committed_for_test(self->budget, v);
}

void
damacy_config_describe(const struct damacy_config* cfg)
{
  if (!cfg) {
    log_info("damacy_config_describe: NULL config");
    return;
  }
  const uint64_t max_gpu = cfg->tuning.max_gpu_memory_bytes;
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  uint64_t pool_reserve = 0;
  {
    uint64_t pool_bytes = 0;
    // resolve_sample_volume_bytes rejects rank=0 / non-positive dims; on
    // those, leave pool_reserve at 0 so describe still prints useful
    // info for the rest of the geometry.
    enum damacy_status pvs = resolve_sample_volume_bytes(cfg, &pool_bytes);
    if (pvs == DAMACY_OK)
      pool_reserve = 2ull * pool_bytes;
    else
      log_info(
        "damacy_config_describe: resolve_sample_volume_bytes failed (%s); "
        "pool_reserve=0",
        damacy_status_str(pvs));
  }
  const uint64_t resolver_budget =
    pool_reserve < max_gpu ? max_gpu - pool_reserve : 0;
  log_info("damacy_config_describe: max_gpu_memory_bytes=%llu "
           "(pool_reserve=%llu, resolver_budget=%llu, "
           "max_chunk_uncompressed_bytes=%llu, batch_size=%u)",
           (unsigned long long)max_gpu,
           (unsigned long long)pool_reserve,
           (unsigned long long)resolver_budget,
           (unsigned long long)runtime_chunk_cap,
           (unsigned)cfg->batch_size);

  struct wave_pool_sizing sizing = { 0 };
  enum damacy_status rs =
    wave_pool_resolve_sizing(resolve_max_chunks_per_wave(cfg),
                             resolve_max_substreams_per_chunk(cfg),
                             resolver_budget,
                             runtime_chunk_cap,
                             cfg->batch_size,
                             &sizing);
  if (rs != DAMACY_OK) {
    log_info("damacy_config_describe: wave_pool_resolve_sizing failed (%s)",
             damacy_status_str(rs));
    return;
  }
  struct gpu_budget_breakdown predicted = { 0 };
  if (gpu_budget_predict(cfg,
                         sizing.host_slab_per_wave,
                         sizing.dev_decompressed_per_wave,
                         &predicted) != DAMACY_OK) {
    log_info("damacy_config_describe: gpu_budget_predict failed");
    return;
  }
  log_info("damacy_config_describe: host_slab_per_wave=%llu "
           "dev_decompressed_per_wave=%llu",
           (unsigned long long)sizing.host_slab_per_wave,
           (unsigned long long)sizing.dev_decompressed_per_wave);
  log_info("damacy_config_describe: dev_compressed=%llu dev_decompressed=%llu "
           "blosc1_meta=%llu fanout_soa=%llu "
           "nvcomp_temp=%llu batch_metadata=%llu",
           (unsigned long long)predicted.dev_compressed,
           (unsigned long long)predicted.dev_decompressed,
           (unsigned long long)predicted.blosc1_meta,
           (unsigned long long)predicted.fanout_soa,
           (unsigned long long)predicted.nvcomp_temp,
           (unsigned long long)predicted.batch_metadata);
  // predicted.total is the *initial* allocation (initial fanout / decoder
  // floors); sizing.worst_case_total_bytes is the post-grow worst case
  // the resolver pre-reserved against the cap. Their difference is the
  // grow-time headroom; the cap minus the worst case is unused slack.
  const uint64_t initial_alloc = predicted.total;
  const uint64_t worst_case = sizing.worst_case_total_bytes;
  const uint64_t reserved_for_grow =
    worst_case > initial_alloc ? worst_case - initial_alloc : 0;
  const uint64_t slack = max_gpu > worst_case ? max_gpu - worst_case : 0;
  log_info("damacy_config_describe: initial_alloc=%llu reserved_for_grow=%llu "
           "worst_case_total=%llu slack=%llu cap=%llu",
           (unsigned long long)initial_alloc,
           (unsigned long long)reserved_for_grow,
           (unsigned long long)worst_case,
           (unsigned long long)slack,
           (unsigned long long)max_gpu);
}
