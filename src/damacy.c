// damacy: streaming loader. Two batch slots and two wave slots in flight.
// A worker thread (src/scheduler) drives the pipeline; the user-thread
// API (push/pop/release/flush) coordinates via scheduler_lock + the
// scheduler's condition variable. Background I/O on the io_queue and
// host parse on compute_pool round out the threading model.

#include "damacy.h"

#include "batch_pool/batch_pool.h"
#include "damacy_config.h"
#include "damacy_stats.h"
#include "gpu_budget/gpu_budget.h"
#include "log/log.h"
#include "lookahead/lookahead.h"
#include "nvtx/nvtx.h"
#include "planner/planner.h"
#include "platform/platform.h"
#include "scheduler/scheduler.h"
#include "store/store.h"
#include "threadpool/threadpool.h"
#include "util/cuda_check.h"
#include "util/prelude.h"
#include "wave/wave.h"
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

  // GPU memory budgeting. Lazy batch-output tensors are summed against
  // gpu_bytes_budget at batch_pool_allocate. 0 budget = no cap.
  uint64_t gpu_bytes_committed;
  uint64_t gpu_bytes_budget;

  struct store* store;
  struct zarr_meta_cache* meta_cache;
  struct zarr_shard_cache* shard_cache;
  struct planner* planner;

  struct threadpool* compute_pool; // host blosc1 parse

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

// Lazy batch-output pool sizing + GPU-budget enforcement. Idempotent.
static enum damacy_status
batch_pool_allocate(struct damacy* self, const struct damacy_aabb* sample_aabb)
{
  struct damacy_batch_pool* pool = &self->batch_pool;
  if (pool->allocated)
    return DAMACY_OK;

  enum damacy_status s = batch_pool_compute_layout(
    pool, sample_aabb, self->cfg.batch_size, damacy_dtype_bpe(self->cfg.dtype));
  if (s != DAMACY_OK)
    return s;

  if (self->gpu_bytes_budget > 0) {
    uint64_t need = 2ull * pool->n_bytes;
    if (self->gpu_bytes_committed + need > self->gpu_bytes_budget) {
      log_error("damacy: batch-output pool would exceed GPU budget "
                "(committed=%llu add=%llu cap=%llu n_bytes=%llu)",
                (unsigned long long)self->gpu_bytes_committed,
                (unsigned long long)need,
                (unsigned long long)self->gpu_bytes_budget,
                (unsigned long long)pool->n_bytes);
      return DAMACY_OOM;
    }
  }

  s = batch_pool_alloc_dev(pool);
  if (s != DAMACY_OK)
    return s;
  self->gpu_bytes_committed += 2ull * pool->n_bytes;
  return DAMACY_OK;
}

// --- planning -------------------------------------------------------------

static enum damacy_status
push_one(struct damacy* self, const struct damacy_sample* sample)
{
  if (!sample->uri)
    return DAMACY_INVAL;
  if (sample->aabb.rank == 0 || sample->aabb.rank > DAMACY_MAX_RANK)
    return DAMACY_RANK;

  const struct zarr_metadata* meta = NULL;
  enum damacy_status ms =
    zarr_meta_cache_get(self->meta_cache, sample->uri, &meta);
  if (ms != DAMACY_OK)
    return ms;

  if (!cast_path_supported(self->cfg.dtype, meta->dtype))
    return DAMACY_DTYPE;
  if (sample->aabb.rank != meta->rank)
    return DAMACY_RANK;
  if (self->batch_pool.allocated &&
      !sample_shape_matches_pool(&self->batch_pool, &sample->aabb))
    return DAMACY_INVAL;

  if (lookahead_push(&self->lookahead, sample))
    return DAMACY_OOM;
  return DAMACY_OK;
}

// Drains n_samples from the lookahead and transitions slot FREE→FILLING.
static enum damacy_status
plan_into_slot(struct damacy* self, uint16_t slot_idx, uint32_t n_samples)
{
  if (n_samples == 0)
    return DAMACY_OK;
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  if (slot->state != BATCH_FREE)
    return DAMACY_INVAL;

  lookahead_drain(&self->lookahead, self->batch_samples, n_samples);

  enum damacy_status status =
    batch_pool_allocate(self, &self->batch_samples[0].aabb);
  if (status != DAMACY_OK)
    goto Cleanup;

  for (uint32_t i = 0; i < n_samples; ++i) {
    self->batch_stage[i].uri = self->batch_samples[i].uri;
    self->batch_stage[i].aabb = self->batch_samples[i].aabb;
  }

  struct planner_output plan_out = {
    .read_ops = slot->read_ops,
    .read_ops_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .chunk_plans = slot->chunk_plans,
    .chunk_plans_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .sample_plans = slot->sample_plans,
    .sample_plans_cap = self->cfg.batch_size,
  };
  uint64_t plan_t0 = monotonic_ns();
  status = planner_plan(self->planner,
                        self->batch_stage,
                        n_samples,
                        slot_idx,
                        self->batch_pool.strides,
                        self->batch_pool.rank,
                        &plan_out);
  metric_record(
    &self->stats.plan, (float)((monotonic_ns() - plan_t0) / 1.0e6), 0, 0);
  if (status != DAMACY_OK)
    goto Cleanup;

  slot->n_chunks = plan_out.n_chunk_plans;
  slot->n_chunks_dispatched = 0;
  slot->chunks_remaining = (int32_t)plan_out.n_chunk_plans;
  slot->n_sample_plans = plan_out.n_sample_plans;
  slot->n_samples = n_samples;
  slot->batch_id = self->next_batch_id++;
  slot->state = BATCH_FILLING;

  // Sample plans uploaded once per batch; waves consume them alongside
  // their per-wave chunk records.
  if (plan_out.n_sample_plans > 0) {
    if (cuMemcpyHtoD(CUDPTR(slot->d_sample_plans),
                     slot->sample_plans,
                     (size_t)plan_out.n_sample_plans *
                       sizeof(struct sample_plan)) != CUDA_SUCCESS) {
      status = DAMACY_CUDA;
      goto Cleanup;
    }
  }

  // Degenerate batch: no chunks → zero the output and skip to READY.
  if (slot->n_chunks == 0) {
    if (cuMemsetD8(CUDPTR(slot->dev_ptr), 0, self->batch_pool.n_bytes) !=
        CUDA_SUCCESS) {
      status = DAMACY_CUDA;
      goto Cleanup;
    }
    slot->state = BATCH_READY;
  }

Cleanup:
  for (uint32_t i = 0; i < n_samples; ++i)
    sample_slot_clear(&self->batch_samples[i]);
  if (status != DAMACY_OK)
    self->failed_status = status;
  return status;
}

// Fills FREE wave slots; plans a fresh batch when no FILLING slot has work.
static enum damacy_status
kick_new_waves(struct damacy* self)
{
  for (;;) {
    int w = find_free_wave(&self->wave_pool);
    if (w < 0)
      break;

    int target_slot = find_filling_slot_with_work(&self->batch_pool);
    if (target_slot < 0) {
      int free_slot = find_free_batch_slot(&self->batch_pool);
      if (free_slot < 0)
        break;
      if (self->lookahead.size < self->cfg.batch_size)
        break;
      enum damacy_status s =
        plan_into_slot(self, (uint16_t)free_slot, self->cfg.batch_size);
      if (s != DAMACY_OK)
        return s;
      continue;
    }

    enum damacy_status s =
      wave_pool_peel(&self->wave_pool, (uint16_t)w, (uint16_t)target_slot);
    if (s != DAMACY_OK)
      return s;
  }
  return DAMACY_OK;
}

// --- scheduler ------------------------------------------------------------

// One scheduler tick, under scheduler_lock. Lazy ctx push on first call.
// Always returns 1 to broadcast — wakeups are cheap and pop re-checks.
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
  if (self->failed_status != DAMACY_OK)
    return 1;

  enum damacy_status r = wave_pool_advance(&self->wave_pool);
  if (r == DAMACY_OK && self->failed_status == DAMACY_OK)
    r = kick_new_waves(self);
  if (r != DAMACY_OK && self->failed_status == DAMACY_OK)
    self->failed_status = r;
  return 1;
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

  threadpool_free(self->compute_pool);
  self->compute_pool = NULL;

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
  store_destroy(self->store);
  self->store = NULL;
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

  self->compute_pool = threadpool_new((int)cfg->n_compute_threads);
  if (!self->compute_pool) {
    s = DAMACY_OOM;
    goto Fail;
  }

  // Phase 5: max_gpu_memory_bytes is the primary knob. Apply the legacy
  // default (~1 GB) if the user left it at 0. host_buffer_bytes /
  // device_buffer_bytes are deprecated; warn once if either is set,
  // then ignore the value. Log the *resolved* cap so the migration case
  // (deprecated fields set, max_gpu_memory_bytes left at 0) reports a
  // useful number rather than "current: 0 bytes".
  const uint64_t resolved_max_gpu = resolve_max_gpu_memory(cfg);
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  self->gpu_bytes_budget = resolved_max_gpu;
  if (cfg->host_buffer_bytes || cfg->device_buffer_bytes)
    log_warn("damacy: host_buffer_bytes / device_buffer_bytes are deprecated "
             "(Phase 5); ignored. Use max_gpu_memory_bytes to size the "
             "pipeline (resolved: %llu bytes).",
             (unsigned long long)resolved_max_gpu);

  // Subtract the batch-output reserve before sizing wave-resident
  // buffers; the resolver is greedy, so without this carve-out the
  // lazy pool has no room at first push.
  const uint64_t pool_reserve = cfg->batch_output_reserve_bytes;
  if (pool_reserve >= resolved_max_gpu) {
    log_error("damacy: batch_output_reserve_bytes=%llu >= "
              "max_gpu_memory_bytes=%llu; nothing left for wave-resident "
              "buffers",
              (unsigned long long)pool_reserve,
              (unsigned long long)resolved_max_gpu);
    s = DAMACY_INVAL;
    goto Fail;
  }
  const uint64_t resolver_budget = resolved_max_gpu - pool_reserve;

  // Resolve per-wave geometry from the resolver budget; rejects with
  // OOM if even the minimum (one chunk per wave) doesn't fit. Then
  // call gpu_budget_compute against the resolved geometry to seed
  // gpu_bytes_committed — the grow paths read this back when checking
  // whether a new allocation would breach the cap.
  struct wave_pool_sizing sizing = { 0 };
  s = wave_pool_resolve_sizing(
    resolver_budget, runtime_chunk_cap, cfg->batch_size, &sizing);
  if (s != DAMACY_OK)
    goto Fail;
  {
    struct gpu_budget budget = { 0 };
    s = gpu_budget_compute(cfg,
                           sizing.host_slab_per_wave,
                           sizing.dev_decompressed_per_wave,
                           &budget);
    if (s != DAMACY_OK)
      goto Fail;
    self->gpu_bytes_committed = budget.total;
    log_info("damacy: resolved geometry from max_gpu_memory_bytes=%llu "
             "(pool_reserve=%llu, resolver_budget=%llu): "
             "host_slab_per_wave=%llu dev_decompressed_per_wave=%llu "
             "initial_nvcomp_temp=%llu predicted_total=%llu",
             (unsigned long long)resolved_max_gpu,
             (unsigned long long)pool_reserve,
             (unsigned long long)resolver_budget,
             (unsigned long long)sizing.host_slab_per_wave,
             (unsigned long long)sizing.dev_decompressed_per_wave,
             (unsigned long long)budget.nvcomp_temp,
             (unsigned long long)budget.total);
    // Resolver guarantees this fits; assert defensively in case the
    // accounting drifts. A breach here is a bug, not user input.
    if (self->gpu_bytes_committed > self->gpu_bytes_budget) {
      log_error(
        "damacy: post-resolve drift: total=%llu cap=%llu "
        "(dev_compressed=%llu dev_decompressed=%llu unshuffle_scratch=%llu "
        "blosc1_meta=%llu fanout_soa=%llu nvcomp_temp=%llu batch_meta=%llu)",
        (unsigned long long)budget.total,
        (unsigned long long)self->gpu_bytes_budget,
        (unsigned long long)budget.dev_compressed,
        (unsigned long long)budget.dev_decompressed,
        (unsigned long long)budget.dev_unshuffle_scratch,
        (unsigned long long)budget.blosc1_meta,
        (unsigned long long)budget.fanout_soa,
        (unsigned long long)budget.nvcomp_temp,
        (unsigned long long)budget.batch_metadata);
      s = DAMACY_OOM;
      goto Fail;
    }
  }

  s = DAMACY_OOM;
  // Sample.uri is absolute; the fs store joins root+key, so empty root
  // turns join into a pass-through.
  struct store_fs_config sc = {
    .root = "",
    .nthreads = (int)cfg->n_io_threads,
  };
  self->store = store_fs_create(&sc);
  CHECK(Fail, self->store);

  self->meta_cache =
    zarr_meta_cache_create(self->store, cfg->n_zarrs_meta_cache);
  CHECK(Fail, self->meta_cache);
  self->shard_cache =
    zarr_shard_cache_create(self->store, cfg->n_shards_meta_cache);
  CHECK(Fail, self->shard_cache);

  struct planner_config pcfg = {
    .meta_cache = self->meta_cache,
    .shard_cache = self->shard_cache,
    .page_alignment = self->page_alignment,
    .max_chunk_uncompressed_bytes = runtime_chunk_cap,
  };
  CHECK(Fail, planner_create(&pcfg, &self->planner) == DAMACY_OK);

  for (int b = 0; b < 2; ++b)
    CHECK(Fail,
          batch_slot_init(&self->batch_pool.slots[b], cfg->batch_size) == 0);

  s = DAMACY_OOM;
  CHECK(Fail,
        wave_pool_init(&self->wave_pool,
                       &self->batch_pool,
                       self->store,
                       self->compute_pool,
                       &self->stats,
                       &self->failed_status,
                       cfg->dtype,
                       sizing.host_slab_per_wave,
                       sizing.dev_decompressed_per_wave,
                       runtime_chunk_cap,
                       resolved_max_gpu,
                       &self->gpu_bytes_committed) == 0);

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
  self->sched =
    scheduler_create(damacy_scheduler_step, self, DAMACY_POP_POLL_NS);
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
        !any_batch_in_flight(&self->batch_pool) &&
        self->lookahead.size < self->cfg.batch_size) {
      r = DAMACY_AGAIN;
      goto Done;
    }
    uint64_t wait_t0 = monotonic_ns();
    scheduler_wait(self->sched);
    metric_record(
      &self->stats.pop_wait, (float)((monotonic_ns() - wait_t0) / 1.0e6), 0, 0);
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
  scheduler_unlock(self->sched);
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
    while (find_free_batch_slot(&self->batch_pool) < 0 &&
           self->failed_status == DAMACY_OK)
      scheduler_wait(self->sched);
    if (self->failed_status != DAMACY_OK) {
      r = self->failed_status;
      goto Done;
    }
    int free_slot = find_free_batch_slot(&self->batch_pool);
    uint32_t n = self->lookahead.size;
    r = plan_into_slot(self, (uint16_t)free_slot, n);
    if (r != DAMACY_OK)
      goto Done;
    self->stats.batches_truncated++;
  }

  uint64_t flush_t0 = monotonic_ns();
  while ((any_wave_in_flight(&self->wave_pool) ||
          find_oldest_filling_slot(&self->batch_pool) >= 0) &&
         self->failed_status == DAMACY_OK)
    scheduler_wait(self->sched);
  metric_record(&self->stats.flush_wait,
                (float)((monotonic_ns() - flush_t0) / 1.0e6),
                0,
                0);
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
  *out = self->stats;
  if (self->meta_cache) {
    struct zarr_meta_cache_stats ms;
    zarr_meta_cache_stats_get(self->meta_cache, &ms);
    out->zarr_meta_hits = ms.counters.hits;
    out->zarr_meta_misses = ms.counters.misses;
  }
  if (self->shard_cache) {
    struct zarr_shard_cache_stats ss;
    zarr_shard_cache_stats_get(self->shard_cache, &ss);
    out->shard_idx_hits = ss.counters.hits;
    out->shard_idx_misses = ss.counters.misses;
  }
  out->gpu_bytes_committed = self->gpu_bytes_committed;
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
  const uint64_t prev = self->gpu_bytes_committed;
  self->gpu_bytes_committed = v;
  return prev;
}

void
damacy_config_describe(const struct damacy_config* cfg)
{
  if (!cfg) {
    log_info("damacy_config_describe: NULL config");
    return;
  }
  const uint64_t resolved_max_gpu = resolve_max_gpu_memory(cfg);
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  const uint64_t pool_reserve = cfg->batch_output_reserve_bytes;
  const uint64_t resolver_budget =
    pool_reserve < resolved_max_gpu ? resolved_max_gpu - pool_reserve : 0;
  log_info("damacy_config_describe: input max_gpu_memory_bytes=%llu "
           "(resolved=%llu, batch_output_reserve=%llu, resolver_budget=%llu, "
           "max_chunk_uncompressed_bytes=%llu, batch_size=%u)",
           (unsigned long long)cfg->max_gpu_memory_bytes,
           (unsigned long long)resolved_max_gpu,
           (unsigned long long)pool_reserve,
           (unsigned long long)resolver_budget,
           (unsigned long long)runtime_chunk_cap,
           (unsigned)cfg->batch_size);
  if (cfg->host_buffer_bytes || cfg->device_buffer_bytes)
    log_info("damacy_config_describe: host_buffer_bytes=%llu / "
             "device_buffer_bytes=%llu are deprecated and ignored",
             (unsigned long long)cfg->host_buffer_bytes,
             (unsigned long long)cfg->device_buffer_bytes);

  struct wave_pool_sizing sizing = { 0 };
  enum damacy_status rs = wave_pool_resolve_sizing(
    resolver_budget, runtime_chunk_cap, cfg->batch_size, &sizing);
  if (rs != DAMACY_OK) {
    log_info("damacy_config_describe: wave_pool_resolve_sizing failed (%s)",
             damacy_status_str(rs));
    return;
  }
  struct gpu_budget budget = { 0 };
  if (gpu_budget_compute(cfg,
                         sizing.host_slab_per_wave,
                         sizing.dev_decompressed_per_wave,
                         &budget) != DAMACY_OK) {
    log_info("damacy_config_describe: gpu_budget_compute failed");
    return;
  }
  log_info("damacy_config_describe: host_slab_per_wave=%llu "
           "dev_decompressed_per_wave=%llu",
           (unsigned long long)sizing.host_slab_per_wave,
           (unsigned long long)sizing.dev_decompressed_per_wave);
  log_info("damacy_config_describe: dev_compressed=%llu dev_decompressed=%llu "
           "unshuffle_scratch=%llu blosc1_meta=%llu fanout_soa=%llu "
           "nvcomp_temp=%llu batch_metadata=%llu",
           (unsigned long long)budget.dev_compressed,
           (unsigned long long)budget.dev_decompressed,
           (unsigned long long)budget.dev_unshuffle_scratch,
           (unsigned long long)budget.blosc1_meta,
           (unsigned long long)budget.fanout_soa,
           (unsigned long long)budget.nvcomp_temp,
           (unsigned long long)budget.batch_metadata);
  // budget.total is the *initial* allocation (initial fanout / decoder
  // floors); sizing.worst_case_total_bytes is the post-grow worst case
  // the resolver pre-reserved against the cap. Their difference is the
  // grow-time headroom; the cap minus the worst case is unused slack.
  const uint64_t initial_alloc = budget.total;
  const uint64_t worst_case = sizing.worst_case_total_bytes;
  const uint64_t reserved_for_grow =
    worst_case > initial_alloc ? worst_case - initial_alloc : 0;
  const uint64_t slack =
    resolved_max_gpu > worst_case ? resolved_max_gpu - worst_case : 0;
  log_info("damacy_config_describe: initial_alloc=%llu reserved_for_grow=%llu "
           "worst_case_total=%llu slack=%llu cap=%llu",
           (unsigned long long)initial_alloc,
           (unsigned long long)reserved_for_grow,
           (unsigned long long)worst_case,
           (unsigned long long)slack,
           (unsigned long long)resolved_max_gpu);
}
