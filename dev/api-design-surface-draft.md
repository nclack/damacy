# damacy public C API — surface design

> Rewritten 2026-05-03 after the design discussion that retired the
> `Sampler` concept, added multi-zarr push-with-lookahead, and made O_DIRECT a
> first-class constraint.
>
> The implementation is C (matching the rest of the codebase). The eventual
> Python binding is planned via Rust+pyo3 or a thin C++ shim; this header
> is the canonical surface either way.

## Premises

- **No internal sampler.** Users (typically a PyTorch `Sampler`/`DataLoader`)
  push `(uri, aabb)` tuples. damacy plans, loads, decompresses, and
  assembles. A uniform-crop sampler exists *for benchmarking and tests* in
  `bench/`, not in the public API.
- **Multi-zarr.** Every push carries a `uri`. Metadata and shard indices
  are cached per-uri.
- **Push-with-lookahead.** The user pushes ahead of consumption so the
  planner has a window to coalesce reads across batches. Fixed-capacity
  queue; pushes block (or `-EAGAIN`) when full.
- **Strictly bounded resources.** Every memory pool, thread count, and
  cache size is fixed at `damacy_create`. Nothing grows during streaming.
- **O_DIRECT, page-aligned reads** from day 1 into a fixed pinned host
  staging slab. Page size is read once via `platform_page_alignment()`.
- **Two CUDA streams** (driver API): `h2d` for transfers, `compute` for
  decompress + assemble. Events synchronize across them.
- **Device binding by current context.** `damacy_create` captures the
  caller's current `CUcontext` and binds the instance to it for life;
  there is no `device_id` in the config. Multi-GPU = one `damacy*` per
  rank, in the standard DDP-one-process-per-GPU shape (see rationale).
- **Streamed (wave) assembly.** The host staging slab is *not* sized for
  a worst-case batch. The pipeline streams chunks through it in waves,
  writing each chunk into its destination output tensor as it goes. The
  per-batch output tensor is sized for one batch; the staging slab is
  sized for IO throughput.
- **Double-buffered, not configurable.** Internally damacy keeps two
  wave slots and two batch slots in flight. Deeper buffering only
  averages latency, not throughput; there is no `prefetch_depth` knob.
- **Main-thread orchestration.** All planning, CUDA launches, and event
  polling happen on the user's thread inside `damacy_push` /
  `damacy_pop` / `damacy_flush`. The only auxiliary threads are the
  `n_io_threads` IO workers. No CUDA host callbacks.
- **Fail-the-stream errors.** Any IO/decode/CUDA failure puts the
  instance into a terminal state — subsequent `pop` returns the failure,
  `push` returns `DAMACY_SHUTDOWN`. Recovery is `damacy_destroy` +
  `damacy_create`.
- **`damacy_flush` for end-of-epoch.** Drains everything currently
  planned and finalizes any partial last batch (truncated to the count
  of complete samples). Idempotent. Stream is resumable after flush.
- **Fixed-width types throughout.** `uint32_t` / `uint64_t` for sizes
  and counts; no `size_t` / `int` in the public ABI. Documented limits
  for chunk and shard sizes (see `src/limits.h`).

## Header sketch

```c
// damacy.h
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "damacy/limits.h"   // DAMACY_MAX_RANK

#ifdef __cplusplus
extern "C" {
#endif

enum damacy_dtype {
  DAMACY_U8, DAMACY_U16, DAMACY_I16, DAMACY_U32, DAMACY_F16, DAMACY_F32
};

enum damacy_status {
  DAMACY_OK        = 0,
  DAMACY_AGAIN,        // non-blocking call would block (queue full)
  DAMACY_INVAL,        // bad arguments
  DAMACY_NOTFOUND,     // uri unresolvable / not a zarr
  DAMACY_DTYPE,        // zarr dtype != configured dtype
  DAMACY_RANK,         // sample rank incompatible with zarr rank
  DAMACY_IO,           // read/open failure on a shard file
  DAMACY_DECODE,       // codec parse / decompression failure
  DAMACY_CUDA,         // driver/runtime call failed
  DAMACY_OOM,          // would exceed a configured cap
  DAMACY_SHUTDOWN,     // pipeline destroyed or draining
};

// Human-readable name for a status code; safe for log/error messages.
const char* damacy_status_str(enum damacy_status s);

// Half-open [beg, end) interval along one axis, in level-0 voxel indices.
struct damacy_interval {
  int64_t beg, end;
};

// Variable-rank AABB. Only dims[0..rank) are read; axis order matches the
// zarr's stored axis order. rank <= DAMACY_MAX_RANK.
struct damacy_aabb {
  struct damacy_interval dims[DAMACY_MAX_RANK];
  uint8_t                rank;
};

struct damacy_sample {
  const char*        uri;   // null-terminated; copied internally
  struct damacy_aabb aabb;
};

// Caller-owned slice of samples to push.
struct damacy_sample_slice {
  const struct damacy_sample* beg;
  const struct damacy_sample* end;
};

// All resource caps fixed at create-time. Nothing grows after this.
// Device is captured from the current CUcontext at damacy_create.
// Output batches are double-buffered (B=2); waves are double-buffered
// internally. Neither is configurable.
struct damacy_config {
  // Batch geometry
  uint32_t batch_size;             // samples per batch
  uint32_t lookahead_batches;      // user-push queue depth (>= 2)
  uint32_t n_io_threads;

  // Streaming buffers (split in half across two wave slots internally)
  uint64_t host_buffer_bytes;      // pinned staging; sized for IO bw
  uint64_t device_buffer_bytes;    // device decompress scratch

  // LRU caps (no FD cache; FDs are open/close per read_op)
  uint32_t n_zarrs_meta_cache;
  uint32_t n_shards_meta_cache;

  // Output dtype expected from all pushed samples; mismatched zarrs error.
  enum damacy_dtype dtype;
};

struct damacy;
struct damacy_batch;

enum damacy_status damacy_create(const struct damacy_config* cfg,
                                 struct damacy**             out);
void               damacy_destroy(struct damacy* d);

// Push as many samples as fit. The return value is the unconsumed suffix
// of the input slice plus a status:
//   OK        all samples were consumed (result.unconsumed.beg == samples.end)
//   AGAIN     the lookahead queue filled mid-slice; caller should pop a
//             batch (or wait) and retry with the returned suffix
//   INVAL     bad arguments (samples.beg > samples.end, null d, etc.)
//   NOTFOUND  could not resolve a uri; result.unconsumed.beg points at it
//   DTYPE     zarr dtype mismatch; result.unconsumed.beg points at the sample
//   RANK      sample rank incompatible with the resolved zarr's rank
// On any non-AGAIN error, the offending sample is at result.unconsumed.beg
// and was NOT consumed.
struct damacy_push_result {
  struct damacy_sample_slice unconsumed;
  enum damacy_status         status;
};
struct damacy_push_result
damacy_push(struct damacy* d, struct damacy_sample_slice samples);

// Block until the next batch is on-device-ready, in push-FIFO order.
// *out is owned by damacy until damacy_release.
enum damacy_status damacy_pop(struct damacy* d, struct damacy_batch** out);

// Return the batch's slot to the pool.
void damacy_release(struct damacy* d, struct damacy_batch* b);

// Drain anything currently planned/in-flight, finalize any partial last
// batch (truncated to the number of complete samples) and ready it for
// pop. Idempotent. Stream is resumable after flush; subsequent push
// starts a fresh batch.
enum damacy_status damacy_flush(struct damacy* d);

struct damacy_batch_info {
  void*             device_ptr;                   // dtype-typed, contiguous
  int64_t           shape[DAMACY_MAX_RANK + 1];   // [N, ...zarr axes]
  uint8_t           rank;                         // includes leading N axis
  enum damacy_dtype dtype;
  void*             ready_stream;                 // CUstream
  uint64_t          batch_id;                     // monotonic
};
// device ordinal is derivable from device_ptr via
// cuPointerGetAttribute(..., CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ...)
// or from ready_stream via cuStreamGetCtx → cuCtxGetDevice.

void damacy_batch_info(const struct damacy_batch* b,
                       struct damacy_batch_info*  out);

// Cumulative metrics. All counters are stage-cumulative; reset with
// damacy_stats_reset.
struct damacy_metric {
  const char* name;
  float       ms;            // cumulative
  float       best_ms;       // best single observation (1e30f = none)
  double      input_bytes;   // cumulative bytes consumed by stage
  double      output_bytes;  // cumulative bytes produced by stage
  uint64_t    count;
};

struct damacy_stats {
  struct damacy_metric plan;
  struct damacy_metric io;
  struct damacy_metric h2d;
  struct damacy_metric decompress;
  struct damacy_metric assemble;
  struct damacy_metric pop_wait_io;
  struct damacy_metric pop_wait_compute;
  struct damacy_metric push_backpressure;
  struct damacy_metric flush_wait;

  uint64_t zarr_meta_hits, zarr_meta_misses;
  uint64_t shard_idx_hits, shard_idx_misses;
  uint64_t batches_emitted;
  uint64_t batches_truncated;
  uint64_t waves_emitted;
};

void damacy_stats_get(const struct damacy* d, struct damacy_stats* out);
void damacy_stats_reset(struct damacy* d);

#ifdef __cplusplus
}
#endif
```

## Usage shape

```c
// CUDA context for the target device must be current before damacy_create.
struct damacy_config cfg = {
  .batch_size          = 8,
  .lookahead_batches   = 4,        // user pushes a few batches ahead
  .n_io_threads        = 8,
  .host_buffer_bytes   = 256ull << 20,   // 128 MB per wave slot
  .device_buffer_bytes = 512ull << 20,   // 256 MB per wave slot
  .n_zarrs_meta_cache  = 4096,
  .n_shards_meta_cache = 16384,
  .dtype               = DAMACY_U16,
};
struct damacy* d = NULL;
if (damacy_create(&cfg, &d) != DAMACY_OK) { /* handle */ }

// Helper: drain a slice through push, popping when the queue is full.
static void
push_all(struct damacy* d, struct damacy_sample_slice s,
         void (*consume)(struct damacy*))
{
  while (s.beg != s.end) {
    struct damacy_push_result r = damacy_push(d, s);
    s = r.unconsumed;
    if (r.status == DAMACY_OK)    return;
    if (r.status == DAMACY_AGAIN) { consume(d); continue; }
    /* otherwise: log r.status, skip the bad sample, advance, etc. */
    fprintf(stderr, "push: %s\n", damacy_status_str(r.status));
    s.beg += 1;   // skip and continue, or break
  }
}

static void train_one(struct damacy* d) {
  struct damacy_batch* b = NULL;
  damacy_pop(d, &b);
  struct damacy_batch_info info; damacy_batch_info(b, &info);
  train_step(&info);   // makes its compute stream wait on info.ready_stream
  damacy_release(d, b);
}

// Prime the lookahead. user_fill_slice fills `buf` with up to N samples.
struct damacy_sample buf[256];
for (int primed = 0; primed < cfg.lookahead_batches * cfg.batch_size; ) {
  size_t n = user_fill_slice(buf, sizeof buf / sizeof *buf);
  push_all(d, (struct damacy_sample_slice){ buf, buf + n }, train_one);
  primed += (int)n;
}

// Steady state.
for (int step = 0; step < N_STEPS; ++step) {
  train_one(d);
  size_t n = user_fill_slice(buf, cfg.batch_size);
  push_all(d, (struct damacy_sample_slice){ buf, buf + n }, train_one);
}

damacy_destroy(d);
```

## Python binding shape (later)

A thin pyo3/pybind layer turns `damacy_pop` + `damacy_batch_info` into a
DLPack v1.0 capsule whose stream field carries `ready_stream`, so JAX/PyTorch
consumers don't need a host sync. The capsule deleter calls `damacy_release`.

## Why these choices

- **Push-with-lookahead, not pull-with-callback.** The user's sampler is
  the source of truth for ordering; damacy never invents samples or
  reorders pushed ones across batches. Lookahead exists *within* the
  planner as cross-batch coalescing room, not in the consumer-visible
  order.
- **No `Sampler` class.** The original draft's `UniformCropSampler` belongs
  in user code (or `bench/`). Embedding it would either be too restrictive
  for real workloads or balloon into a parallel sampler library.
- **`damacy_release` is explicit, not RAII-via-shared_ptr.** Same reason
  C; no language assumes destructor semantics.
- **Pool-managed `damacy_batch`.** Caller-allocated output tensors would
  force the user to plan `prefetch_depth × batch_nbytes` and break the
  pipeline's freedom to recycle slots; the handle gives the runtime control.
- **`ready_stream` exposed.** Overlapping decode with training is the entire
  point; users must be able to `cuStreamWaitEvent` instead of host-syncing.
- **Bounded `damacy_config`.** Production trainers want predictable RSS
  and no surprise allocations during training. The config struct is the
  whole resource contract.
- **No `device_id` field.** `damacy_create` captures the calling thread's
  current `CUcontext`; the IO and scheduler threads `cuCtxSetCurrent` to
  that context as needed. Same pattern as cuBLAS/cuDNN/nvCOMP. The device
  ordinal is derivable from the returned `device_ptr` (or `ready_stream`)
  if a consumer needs it for routing.
- **Slice push, not single-sample push.** Per-sample call overhead is the
  obvious bottleneck for a streaming API once batch size grows or samples
  are tiny. `damacy_push` takes a `damacy_sample_slice` and returns the
  unconsumed suffix + status, so callers amortize call overhead and so
  partial consumption (queue full mid-slice, validation error on one
  sample) is expressible without losing the cursor. Internally the URI is
  copied so the caller can free the slice immediately after return.
- **Typed `damacy_status` enum, not POSIX errnos.** Returned by every
  fallible call; `damacy_status_str` gives a human-readable name. Future
  bindings (Python via pyo3/pybind) translate the enum to native
  exceptions cleanly, and we avoid overloading `errno` semantics for
  library-specific conditions like `DAMACY_DTYPE` or `DAMACY_RANK`.

## Multi-GPU shape (v1: one damacy per rank)

The supported pattern is the standard PyTorch DDP layout: **one OS process
per GPU**. Each rank establishes a CUDA context for its device, calls
`damacy_create` once, and pushes/pops on that single damacy. The library
never multiplexes batches across devices.

This works because under DDP each rank's data loader only ever needs to
produce batches for one GPU; cross-rank coordination (gradient all-reduce)
is NCCL's job, not the data loader's. The user's `Sampler` (typically
`DistributedSampler`) handles cross-rank dataset partitioning *before*
samples are pushed.

Single-process multi-GPU (legacy `nn.DataParallel`, some model-parallel
inference servers) is out of scope for v1. The escape hatch is to
construct multiple `damacy*` instances, one per device. A future v2 could
let one instance fan batches across devices, but it would change output
pool partitioning, scheduler structure, and the per-context binding rule
above; not worth the cost for a workload most users don't have.

## Open / deferred

- **Mip selection.** Auto from requested aabb resolution per project memory;
  selection logic lives in the planner and is not exposed in the surface.
  Out of scope for v1 — we plan at level 0 only.
- **Mixed dtype across zarrs.** Out of scope; the config's `dtype` is
  enforced uniformly. A future `damacy_subgroup` could partition pushes.
- **Read-config (interpolation/boundary/value-transform).** Deferred; the
  per-axis "read sampler" naming is parked per project memory. v1 returns
  the raw stored data; transforms happen post-pop.
- **GDS (cufile).** Deferred. Host-pinned + cudaMemcpyAsync is the v1
  path. The internal `host_buffer` becomes a `device_buffer` later without
  surface changes.
