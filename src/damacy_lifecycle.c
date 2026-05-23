#include "damacy.h"

#include "damacy_config.h"
#include "damacy_internal.h"
#include "damacy_stats.h"
#include "log/log.h"
#include "platform/platform.h"
#include "store/store_fs_gds.h"
#include "util/cuda_check.h"
#include "util/prelude.h"
#include "wave/wave_budget.h"

#include <stdlib.h>

static int
damacy_io_exec_post(struct prefetch_executor* e,
                    void (*fn)(void*),
                    void* ctx,
                    void (*ctx_free)(void*))
{
  struct damacy* d = container_of(e, struct damacy, io_exec);
  return io_queue_post(d->prefetch_io_q, fn, ctx, ctx_free);
}

// --- ctx guard ------------------------------------------------------------

enum damacy_status
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

void
ctx_guard_exit(struct ctx_guard* g)
{
  if (g && g->active) {
    cuCtxPopCurrent(NULL);
    g->active = 0;
  }
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

  // Order: stop producers (prefetcher) → drain consumers (io_queue) →
  // free resources (caches). prefetch_fetch_worker locks cache mutexes
  // on completion, so caches must outlive io_queue's pending tasks.
  prefetcher_destroy(self->prefetcher);
  self->prefetcher = NULL;
  io_queue_destroy(self->prefetch_io_q);
  self->prefetch_io_q = NULL;
  prefetch_cache_destroy(self->chunk_layout_cache);
  self->chunk_layout_cache = NULL;
  prefetch_cache_destroy(self->shard_index_cache);
  self->shard_index_cache = NULL;
  prefetch_cache_destroy(self->array_meta_cache);
  self->array_meta_cache = NULL;

  wave_pool_destroy(&self->wave_pool, cuda_skip);

  free(self->staging);
  self->staging = NULL;
  free(self->batch_stage);
  self->batch_stage = NULL;
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
  const uint32_t resolved_max_substreams_per_chunk =
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

  self->prefetch_io_q = io_queue_create(2, &self->numa);
  CHECK(Fail, self->prefetch_io_q);
  self->io_exec.post = damacy_io_exec_post;
  array_meta_fetcher_init(&self->array_meta_fetcher, self->store_host);
  {
    struct prefetch_cache_config amc_cfg = {
      .capacity = cfg->tuning.n_zarrs_meta_cache,
      .max_probe = 16,
      .ops = &array_meta_ops,
      .fetcher = &self->array_meta_fetcher.base,
      .executor = &self->io_exec,
    };
    self->array_meta_cache = prefetch_cache_create(&amc_cfg);
    CHECK(Fail, self->array_meta_cache);
  }
  shard_index_fetcher_init(
    &self->shard_index_fetcher, self->store_host, self->array_meta_cache);
  {
    struct prefetch_cache_config sic_cfg = {
      .capacity = cfg->tuning.n_shards_meta_cache,
      .max_probe = 16,
      .ops = &shard_index_ops,
      .fetcher = &self->shard_index_fetcher.base,
      .executor = &self->io_exec,
    };
    self->shard_index_cache = prefetch_cache_create(&sic_cfg);
    CHECK(Fail, self->shard_index_cache);
  }
  chunk_layout_fetcher_init(&self->chunk_layout_fetcher,
                            self->store_host,
                            self->array_meta_cache,
                            self->shard_index_cache,
                            resolved_max_substreams_per_chunk);
  {
    struct prefetch_cache_config clc_cfg = {
      .capacity = cfg->tuning.n_zarrs_meta_cache,
      .max_probe = 16,
      .ops = &chunk_layout_ops,
      .fetcher = &self->chunk_layout_fetcher.base,
      .executor = &self->io_exec,
    };
    self->chunk_layout_cache = prefetch_cache_create(&clc_cfg);
    CHECK(Fail, self->chunk_layout_cache);
  }

  for (int b = 0; b < 2; ++b)
    CHECK(Fail,
          batch_slot_init(&self->batch_pool.slots[b], cfg->batch_size) == 0);

  s = DAMACY_OOM;
  // Pin the calling thread to the GPU's NUMA node for the duration of
  // wave_pool_init so first-touch of pinned-host slabs + per-wave
  // scratch lands on the right node. Restored immediately after.
  {
    struct platform_cpu_mask saved_affinity;
    numa_scope_enter(&self->numa, &saved_affinity);
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
    numa_scope_exit(&saved_affinity);
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

  {
    struct prefetcher_config pf_cfg = {
      .lookahead = &self->lookahead,
      .array_meta_cache = self->array_meta_cache,
      .shard_index_cache = self->shard_index_cache,
      .chunk_layout_cache = self->chunk_layout_cache,
      .capacity = cfg->lookahead_batches * cfg->batch_size,
      // +4 covers the admit→release_batch transit window per batch_id.
      .batch_capacity = cfg->lookahead_batches + 4,
    };
    self->prefetcher = prefetcher_create(&pf_cfg);
    CHECK(Fail, self->prefetcher);
  }

  self->batch_stage = (struct planner_sample*)calloc(
    cfg->batch_size, sizeof(struct planner_sample));
  CHECK(Fail, self->batch_stage);
  self->staging = (struct prefetcher_ready*)calloc(
    cfg->batch_size, sizeof(struct prefetcher_ready));
  CHECK(Fail, self->staging);

  self->handle.d = self;

  // Start the prefetcher worker before the scheduler so the scheduler's
  // first tick can see ready batches.
  CHECK(Fail, prefetcher_start(self->prefetcher) == 0);

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
