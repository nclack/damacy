// damacy: streaming loader.
//
// Step 5: wave scheduler + double buffering.
//
// Two batch slots (B=2) and two wave slots (W=2) in flight. Each
// `damacy_pop` advances the wave state machine, kicks new work into
// any FREE wave slot, and either returns a READY batch or
// poll-sleeps until one becomes available. Waves do not cross batch
// boundaries (each wave belongs to exactly one batch slot); read-op
// coalescing lands in step 7.
//
// Threading: planner / scheduler / CUDA launches all run on the user
// thread inside damacy_push / damacy_pop / damacy_flush. The only
// background threads are the n_io_threads io_queue workers; each job
// is a single `pread` against an FD looked up in store_fs's per-key
// FD cache (`store_fs.c::fs_get_file`).

#include "damacy.h"

#include "assemble/assemble.h"
#include "batch_pool/batch_pool.h"
#include "damacy_config.h"
#include "damacy_internal.h"
#include "damacy_stats.h"
#include "decoder/bitshuffle.h"
#include "decoder/blosc1.h"
#include "decoder/blosc1_host.h"
#include "decoder/decoder_lz4.h"
#include "decoder/decoder_memcpy.h"
#include "decoder/decoder_zstd.h"
#include "decoder/shuffle.h"
#include "decoder/status_reduce.h"
#include "gpu_budget/gpu_budget.h"
#include "log/log.h"
#include "lookahead/lookahead.h"
#include "planner/planner.h"
#include "platform/platform.h"
#include "store/store.h"
#include "threadpool/threadpool.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "wave/wave.h"
#include "zarr/zarr_meta_cache.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_cache.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

// Poll interval inside damacy_pop's wait-loop. ~50 µs is short enough
// that the boundary between wave stages doesn't add visible latency,
// long enough that we don't burn a core spinning.
#define DAMACY_POP_POLL_NS 50000

// CUDA driver-API result check. Goes through log_error so it lands in
// the same channel as the project's CHECK macros.
#define CU(label, expr)                                                        \
  do {                                                                         \
    CUresult _r = (expr);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      const char* _msg = NULL;                                                 \
      cuGetErrorString(_r, &_msg);                                             \
      log_error("cu: %s -> %s", #expr, _msg ? _msg : "?");                     \
      goto label;                                                              \
    }                                                                          \
  } while (0)

// Driver API uses CUdeviceptr (uint64_t) for device pointers; we keep
// our struct fields typed (void*, struct assemble_chunk*, ...) and cast
// at the alloc/free/memcpy boundary. Inverse direction
// (CUdeviceptr → typed) needs the inverse cast inline.
#define CUDPTR(p) ((CUdeviceptr)(uintptr_t)(p))

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

// Validate one sample against the meta cache + cfg. Returns the appropriate
// damacy_status; on OK, the sample is pushed into the lookahead.
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

// Drain n_samples from the lookahead, plan them into the given batch slot,
// and transition it FREE→FILLING. If n_samples == 0 this is a no-op
// returning OK. Exposed via damacy_internal.h so wave/ can drive batch
// planning from kick_new_waves.
enum damacy_status
damacy_plan_into_slot(struct damacy* self,
                      uint16_t slot_idx,
                      uint32_t n_samples)
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

  // Upload sample_plans to device once per batch. Waves consume them
  // alongside their per-wave chunk records.
  if (plan_out.n_sample_plans > 0) {
    if (cuMemcpyHtoD(CUDPTR(slot->d_sample_plans),
                     slot->sample_plans,
                     (size_t)plan_out.n_sample_plans *
                       sizeof(struct sample_plan)) != CUDA_SUCCESS) {
      status = DAMACY_CUDA;
      goto Cleanup;
    }
  }

  // Degenerate batch: planner emits no chunks → output stays
  // zero-initialized; transition straight to READY after zeroing.
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

// --- public API: create / destroy ----------------------------------------

// Single teardown list shared by every destroy path. cuda_skip=1 leaks
// CUDA-owned state and skips driver calls; CPU heap is always released.
static void
destroy_inner(struct damacy* self, int cuda_skip)
{
  if (!self)
    return;

  if (!cuda_skip) {
    // NULL stream is the legacy default; the per-stream guard separates
    // "we created this" from "never set in a partially-failed create".
    CUstream* const streams[] = {
      &self->stream_compute,
      &self->stream_h2d,
      &self->stream_zstd,
      &self->stream_lz4,
    };
    for (size_t i = 0; i < countof(streams); ++i)
      if (*streams[i]) {
        cuStreamSynchronize(*streams[i]);
        cuStreamDestroy(*streams[i]);
        *streams[i] = NULL;
      }
  }

  threadpool_free(self->compute_pool);
  self->compute_pool = NULL;

  free(self->batch_stage);
  self->batch_stage = NULL;
  free(self->batch_samples);
  self->batch_samples = NULL;
  lookahead_destroy(&self->lookahead);
  for (int w = 0; w < 2; ++w)
    wave_destroy(&self->waves[w], cuda_skip);
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
  // Set before any goto Fail so the cleanup branch doesn't try to
  // release a primary we never retained (calloc gives 0, which is a
  // valid CUdevice).
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
  }

  s = ctx_guard_enter(self, &cg);
  if (s != DAMACY_OK)
    goto Fail;

  // Non-blocking so damacy doesn't force-serialize against the legacy
  // default stream that some user code still lands on.
  CU(Fail, cuStreamCreate(&self->stream_h2d, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&self->stream_compute, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&self->stream_zstd, CU_STREAM_NON_BLOCKING));
  CU(Fail, cuStreamCreate(&self->stream_lz4, CU_STREAM_NON_BLOCKING));

  self->compute_pool = threadpool_new((int)cfg->n_compute_threads);
  if (!self->compute_pool) {
    s = DAMACY_OOM;
    goto Fail;
  }

  // Predict wave-resident GPU bytes and reject early if over budget.
  // Batch-output tensors (sized from the first AABB) are checked
  // separately at batch_pool_allocate.
  self->gpu_bytes_budget = cfg->max_gpu_memory_bytes;
  {
    struct gpu_budget budget = { 0 };
    s = gpu_budget_compute(cfg, &budget);
    if (s != DAMACY_OK)
      goto Fail;
    self->gpu_bytes_committed = budget.total;
    if (self->gpu_bytes_budget > 0 &&
        self->gpu_bytes_committed > self->gpu_bytes_budget) {
      log_error(
        "damacy: GPU budget exceeded at create: total=%llu cap=%llu "
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

  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
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

  uint64_t host_per_wave = cfg->host_buffer_bytes / 2;
  uint64_t dev_per_wave = cfg->device_buffer_bytes / 2;
  s = DAMACY_INVAL;
  CHECK(Fail, host_per_wave > 0 && dev_per_wave > 0);
  s = DAMACY_OOM;
  const uint8_t max_bpe = resolve_max_bpe(cfg);
  for (int w = 0; w < 2; ++w)
    CHECK(Fail,
          wave_init(&self->waves[w],
                    host_per_wave,
                    dev_per_wave,
                    max_bpe,
                    runtime_chunk_cap) == 0);

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

  ctx_guard_exit(&cg);
  *out = self;
  return DAMACY_OK;

Fail:
  if (self) {
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
    // ctx no longer pushable (device reset, primary released elsewhere).
    // Leak CUDA-owned state and walk the same teardown list with skip=1
    // so any new resource added to destroy_inner is covered here too.
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
  if (self->failed_status != DAMACY_OK) {
    r.status = DAMACY_SHUTDOWN;
    return r;
  }
  if (samples.beg > samples.end) {
    r.status = DAMACY_INVAL;
    return r;
  }
  // No ctx_guard: SPSC enqueue + sync zarr metadata I/O only; no CUDA
  // driver calls reach this path (see issue #21).
  for (const struct damacy_sample* s = samples.beg; s != samples.end; ++s) {
    if (self->lookahead.size == self->lookahead.cap) {
      r.unconsumed.beg = s;
      r.status = DAMACY_AGAIN;
      return r;
    }
    enum damacy_status ps = push_one(self, s);
    if (ps != DAMACY_OK) {
      r.unconsumed.beg = s;
      r.status = ps;
      return r;
    }
  }
  r.unconsumed.beg = samples.end;
  return r;
}

// --- pop ------------------------------------------------------------------

enum damacy_status
damacy_pop(struct damacy* self, struct damacy_batch** out)
{
  CHECK_SILENT(InvalidArg, self);
  CHECK_SILENT(InvalidArg, out);
  *out = NULL;
  if (self->failed_status != DAMACY_OK)
    return self->failed_status;

  struct ctx_guard cg = { 0 };
  enum damacy_status r = ctx_guard_enter(self, &cg);
  if (r != DAMACY_OK)
    return r;

  for (;;) {
    r = advance_waves(self);
    if (r != DAMACY_OK)
      goto Done;
    // finalize_wave can set failed_status on a post-decode codec error
    // even when advance_waves itself returned OK; bail before handing
    // out a possibly-corrupt batch.
    if (self->failed_status != DAMACY_OK) {
      r = self->failed_status;
      goto Done;
    }
    r = kick_new_waves(self);
    if (r != DAMACY_OK)
      goto Done;

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
    if (!any_wave_in_flight(self->waves) && !any_batch_in_flight(&self->batch_pool) &&
        self->lookahead.size < self->cfg.batch_size) {
      r = DAMACY_AGAIN;
      goto Done;
    }
    // Attribute the upcoming poll to whichever stage we're blocked on.
    // Prefer compute when any wave has reached H2D/ASSEMBLE — that's
    // what we'd most likely be waiting on; fall back to IO otherwise.
    int waiting_compute = 0;
    for (int w = 0; w < 2; ++w) {
      enum wave_state ws = self->waves[w].state;
      if (ws == WAVE_H2D || ws == WAVE_ASSEMBLE) {
        waiting_compute = 1;
        break;
      }
    }
    uint64_t poll_t0 = monotonic_ns();
    platform_sleep_ns(DAMACY_POP_POLL_NS);
    float poll_ms = (float)((monotonic_ns() - poll_t0) / 1.0e6);
    metric_record(waiting_compute ? &self->stats.pop_wait_compute
                                  : &self->stats.pop_wait_io,
                  poll_ms,
                  0,
                  0);
  }

Done:
  ctx_guard_exit(&cg);
  return r;

InvalidArg:
  return DAMACY_INVAL;
}

void
damacy_release(struct damacy* self, struct damacy_batch* b)
{
  if (!self || !b || b != &self->handle)
    return;
  uint16_t s = b->slot_idx;
  if (s >= 2)
    return;
  if (self->batch_pool.slots[s].state != BATCH_HELD)
    return;
  self->batch_pool.slots[s].state = BATCH_FREE;
  self->batch_pool.slots[s].n_chunks = 0;
  self->batch_pool.slots[s].n_chunks_dispatched = 0;
}

// --- flush ----------------------------------------------------------------

enum damacy_status
damacy_flush(struct damacy* self)
{
  if (!self)
    return DAMACY_INVAL;
  if (self->failed_status != DAMACY_OK)
    return self->failed_status;

  struct ctx_guard cg = { 0 };
  enum damacy_status r = ctx_guard_enter(self, &cg);
  if (r != DAMACY_OK)
    return r;

  // Plan a partial batch from the remaining lookahead, if any.
  if (self->lookahead.size > 0 && self->lookahead.size < self->cfg.batch_size) {
    int free_slot = find_free_batch_slot(&self->batch_pool);
    if (free_slot < 0) {
      // Both slots in use; drain one wave-cycle's worth and try again.
      // For step 5 we allow one drain pass; if still no slot, return AGAIN.
      r = advance_waves(self);
      if (r != DAMACY_OK)
        goto Done;
      free_slot = find_free_batch_slot(&self->batch_pool);
      if (free_slot < 0) {
        r = DAMACY_AGAIN;
        goto Done;
      }
    }
    uint32_t n = self->lookahead.size;
    r = damacy_plan_into_slot(self, (uint16_t)free_slot, n);
    if (r != DAMACY_OK)
      goto Done;
    self->stats.batches_truncated++;
  }

  // Drain everything in flight by spinning the scheduler until no
  // FILLING slots remain.
  uint64_t flush_t0 = monotonic_ns();
  while (any_wave_in_flight(self->waves) || find_oldest_filling_slot(&self->batch_pool) >= 0) {
    r = advance_waves(self);
    if (r != DAMACY_OK)
      goto Done;
    r = kick_new_waves(self);
    if (r != DAMACY_OK)
      goto Done;
    if (any_wave_in_flight(self->waves))
      platform_sleep_ns(DAMACY_POP_POLL_NS);
  }
  metric_record(&self->stats.flush_wait,
                (float)((monotonic_ns() - flush_t0) / 1.0e6),
                0,
                0);
  r = DAMACY_OK;

Done:
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
  out->ready_stream = (void*)self->stream_compute;
  out->batch_id = slot->batch_id;
  for (uint8_t d = 0; d < self->batch_pool.rank; ++d)
    out->shape[d] = self->batch_pool.shape[d];
  // shape[0] reflects the actual sample count (== n_samples for the
  // batch, which equals cfg.batch_size for full batches and < that
  // for flushed partials).
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
