# Internal architecture for damacy

> Rewritten 2026-05-03 (round 2) after critique. Streaming append-only
> plan queue; double-buffered waves and batches; main-thread-driven (no
> internal orchestrator thread, no CUDA host callbacks); parallel IO via
> a chucky-style `io_queue`; cuEvents for CUDA sync; fail-the-stream
> error model; `damacy_flush`.

## Premises

- **Append-only streaming plan.** The planner is not "per batch." It
  appends `chunk_plan` records to a single FIFO queue tagged with their
  `batch_pool_slot`. The wave scheduler walks that queue.
- **Wave = unit of coalescing and double-buffering.** A wave is the
  largest set of `chunk_plan`s whose page-aligned reads fit in one
  wave-sized host slab. Waves are coalesced before IO is issued. Two
  wave slots are kept in flight at all times. A wave may cross a batch
  boundary (it may hold the last few chunks of batch K and the first
  several of batch K+1).
- **Two batch slots in flight.** `prefetch_depth` is fixed at 2 (one
  being filled by waves, one waiting for `damacy_pop`). More buffering
  averages latency without raising throughput.
- **Main-thread orchestration.** All planning, scheduling, CUDA kicks,
  and event polling happen on the user's thread inside `damacy_push` /
  `damacy_pop` / `damacy_flush`. No internal orchestrator thread, no
  CUDA host callbacks. CUDA is synchronized via cuEvents that the
  orchestrator polls/syncs explicitly.
- **IO via `io_queue`.** A `io_queue` (chucky-style: post(fn, ctx),
  record() → event, event_wait, is_shutdown) handles the only true
  parallelism damacy needs. IO workers issue one `pread` per
  `read_op` against an FD looked up in store_fs's per-key FD cache
  (the design originally specified `open`/`close` per read; the
  implementation evolved to cache FDs — see "FD cache" note below).
- **Two CUDA streams** (driver API): `h2d` and `compute`. Decompress
  and assemble share `compute` so we don't need an event between them.
- **Fail-the-stream.** Any IO/decode/CUDA error puts the pipeline into
  a terminal failed state. Subsequent `damacy_pop` returns the failure
  status; subsequent `damacy_push` returns `DAMACY_SHUTDOWN`. Recovery
  requires destroy + recreate.
- **Per-key FD cache (lives in `store_fs`).** `fs_get_file` does a
  linear scan over a `(key → platform_file*)` slot array under a
  mutex; on miss it `open`s and inserts. FDs are held for the
  lifetime of the store and closed in `fs_destroy`. The original
  design said "no FD cache" (cheap because the inode is hot once the
  shard index is cached); profiling moved this to a cache. There is
  no eviction yet — `ulimit -n` becomes the implicit cap.

## Components

```
              user thread (calls push / pop / flush / release)
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Lookahead queue         │  cap: lookahead_batches × batch_size
                     │ (samples awaiting plan) │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ Planner                 │  consumes lookahead, appends to:
                     │  AABB → chunks          │
                     └────────────┬────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────────────┐
                     │ Plan queue (append-only ring)    │  cap: bounded
                     │  chunk_plan records, tagged with │
                     │  batch_pool_slot                 │
                     └────────────┬─────────────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────────────┐
                     │ Wave scheduler                   │  per-wave: peel a prefix
                     │  coalesces page-aligned reads    │  of plan queue that fits
                     │  builds wave's read_ops[]        │  in a wave slab
                     └─────┬─────────────┬──────────────┘
                           │             │
                  ┌────────▼───┐    ┌────▼────────────┐
                  │ Caches:    │    │ Wave slot 0/1   │
                  │  zarr meta │    │  host slab      │
                  │  shard idx │    │  dev compressed │
                  │  (LRUs)    │    │  dev decompress │
                  └────────────┘    │  meta buffers   │
                                    └────┬────────────┘
                                         │
                                         ▼
                          io_queue (n_io_threads)
                              open / pread / close
                                         │   io_event
                                         ▼
                                   h2d stream
                          cuMemcpyHtoDAsync  (slab + meta)
                                         │   cuEvent
                                         ▼
                                  compute stream
                          nvCOMP zstd batched decompress
                                         │
                                         ▼
                          assemble kernel → output_pool[slot]
                                         │   cuEvent (per-wave)
                                         ▼
                          orchestrator decrements per-batch
                          chunks_remaining; on zero, batch
                          slot transitions to READY
                                         │
                                         ▼
                           damacy_pop returns batch handle
```

## Threading model

Three thread roles, no others:

| Role | Count | Does |
|---|---|---|
| User thread | any | calls public API |
| io_queue worker | `n_io_threads` | `pread` per `read_op` (FDs cached per key in `store_fs`) |
| (no orchestrator thread) | 0 | — |

All planner / scheduler / CUDA-launch / event-poll work happens on the
user thread inside the entered API call. Concretely:

- `damacy_push(slice)`: copies samples into the lookahead queue (under a
  mutex); runs the planner if the plan queue has space; returns. Does
  not touch CUDA.
- `damacy_pop(out)`:
  1. Advance: poll io_events and cuEvents for waves in flight; transition
     wave states; on assemble-done, decrement chunks_remaining for the
     affected batch slots; mark batches READY when their counter hits zero.
  2. Kick: for any free wave slot, peel the next wave from the plan queue
     (running planner first if plan queue is dry and lookahead has work),
     build coalesced `read_op`s, post to io_queue; advance
     io_done→h2d_kick, h2d_done→decompress_kick, decompress_done→assemble_kick
     (all on the appropriate stream).
  3. Return: if a batch is READY, hand it back. Otherwise wait — see
     "Waiting without callbacks" below — and return to step 1.
- `damacy_flush()`: inserts a flush marker in the plan queue, then runs the
  pop loop until no work remains in flight and any partial batch (if the
  marker forced a truncation) has transitioned to READY. Idempotent: if
  nothing is in flight and the lookahead is empty, returns immediately.
- `damacy_release(b)`: returns the slot to the pool (under a mutex) and
  signals so a blocked `damacy_pop` can proceed.

CUDA context binding: `damacy_create` captures the caller's current
`CUcontext`. The user thread is assumed to keep that context current
during all subsequent calls; we `cuCtxGetCurrent` and assert in debug
builds. (We don't `cuCtxSetCurrent` from inside the API; that would
race with the user's own context manipulation.)

### Waiting without callbacks

`damacy_pop` may need to block until *either* an io_event fires or a
cuEvent completes. Without host callbacks we have two practical options:

1. **Brief poll loop** (v1): cuEventQuery + io_event poll + `platform_sleep_ns`
   for ~10–100 µs. Trivial; adds tens of µs latency at the boundaries
   between stages.
2. **Single-fd notify** (later if needed): a pipe or eventfd that both io
   workers and a CUDA-stream-callback shim signal; orchestrator does
   `poll()` / `read()`. Adds a callback (one) but only as a wakeup, not
   for any logic.

Start with (1). Revisit only if profiling shows poll-induced stalls.

## Data shapes

```c
// Page-aligned IO operation. Multiple chunk_plans may share one read_op
// after coalescing; in the un-coalesced case it's 1:1.
struct read_op {
  uint32_t shard_id;          // index into wave's interned-paths table
  uint64_t file_offset;       // multiple of platform_page_alignment()
  uint32_t nbytes;            // page-multiple; <= DAMACY_MAX_READ_BYTES
  uint64_t dst_buf_offset;    // page-aligned offset into wave's host slab
};

// One chunk's full plan: where it lives on disk, where it lands in the
// output. AoS, no parallel-array indirection. Carries the rank-erased
// AABBs that the assemble kernel reads directly.
struct chunk_plan {
  uint32_t           read_op_idx;       // which read brought this chunk
  uint32_t           offset_in_read;    // chunk start = host_slab + read.dst_buf_offset + this
  uint32_t           compressed_nbytes; // chunk's raw bytes
  uint32_t           decompressed_nbytes;
  uint32_t           dev_decompressed_offset;  // arena offset into wave's decompress slab
  uint16_t           batch_pool_slot;   // 0 or 1 (prefetch_depth = 2)
  uint8_t            zarr_rank;         // rank of src.dims, src_strides
  uint8_t            _pad;
  struct damacy_aabb src;               // intersection in chunk-local frame (rank=zarr_rank)
  struct damacy_aabb dst;               // intersection in [N, ...] of output (rank=zarr_rank+1)
  int64_t            src_strides[DAMACY_MAX_RANK];  // elements; from zarr chunk shape
};

// Wave bundle: what's in-flight together.
struct wave {
  // CPU side (planner-owned, valid until wave is freed):
  const char**         interned_paths;     // shard_id → path
  uint32_t             n_interned_paths;
  struct read_op*      read_ops;
  uint32_t             n_read_ops;
  struct chunk_plan*   chunk_plans;
  uint32_t             n_chunk_plans;
  // Buffers (slot-owned; reused across waves in this slot):
  void*                host_slab;          // page-aligned pinned, host_buffer_bytes/2
  uint64_t             host_used_bytes;    // sum of read_ops nbytes
  void*                dev_compressed;     // mirrors host_slab on device
  void*                dev_decompressed;   // device_buffer_bytes/2; arena allocator
  void*                dev_meta;           // dev copy of read_ops + chunk_plans for kernels
  // CUDA events (one per stream stage):
  void*                io_done_event;      // signaled by orchestrator after io_event_wait
  void*                h2d_event;          // recorded after cuMemcpyHtoDAsync
  void*                compute_event;      // recorded after assemble kernel
  // State:
  enum wave_state      state;              // FREE/PLANNED/IO/H2D/DECOMP/ASSEMBLE/DONE
};
```

`compressed_nbytes` and `decompressed_nbytes` are `uint32_t`. We
deliberately bound chunk size: `DAMACY_MAX_CHUNK_BYTES = 4 GB` (asserted
in the planner). Target chunks are ~1 MB.

`file_offset` is `uint64_t`. We deliberately bound shard size:
`DAMACY_MAX_SHARD_BYTES = 64 TB` (only 46 bits used; asserted at shard
index parse time). Target shards are ~1 GB.

`host_buffer_bytes` and `device_buffer_bytes` from the config are
split in half: each wave slot owns one half.

## Wave lifecycle (state machine)

```
            (FREE)
              │  scheduler picks next plan-queue prefix
              ▼
           PLANNED   (read_ops + chunk_plans built, host slab claimed)
              │  io_queue_post all read_ops; record final io_event
              ▼
              IO     (waiting for io_event_wait)
              │  io done; cuMemcpyHtoDAsync on h2d stream; record event
              ▼
             H2D     (waiting for h2d cuEvent)
              │  cuStreamWaitEvent(compute, h2d); kick decompress
              ▼
            DECOMP   (waiting; no event between decomp and assemble)
              │  kick assemble kernel; record compute_event
              ▼
           ASSEMBLE  (waiting for compute_event)
              │  decrement per-batch chunks_remaining; mark READY batches
              ▼
            DONE → FREE
```

State transitions are advanced inside `damacy_pop` (and `damacy_flush`).
A wave's transition `IO → H2D` requires both that io is complete *and*
the h2d stream has capacity (always true with 2 wave slots).

## Backpressure rules (where things stall)

| Producer | Consumer | Stall when... | Visible as |
|---|---|---|---|
| `damacy_push` | lookahead queue | lookahead full | returns `DAMACY_AGAIN`; user pops or waits |
| Planner | plan queue | plan queue full OR no batch_pool_slot for the next batch | planner pauses; lookahead may fill behind it (→ AGAIN to push) |
| Wave scheduler | wave slots | both wave slots in flight | nothing kicked this `pop`; pop polls and retries |
| Wave H2D | h2d stream | (n/a; serial within slot) | — |
| Decompress | compute stream | (n/a; serial within slot) | — |
| Output pool | batch slots | both READY but unread, or filling slot has chunks_remaining > 0 | scheduler can't allocate a new `batch_pool_slot` for the planner |

Plan queue capacity: `plan_queue_capacity = 2 × max_chunks_per_batch ×
batch_size` (enough to hold both in-flight batches' worth of chunks
plus a wave's worth of look-ahead). Bounded; configurable internally,
not exposed.

A new `batch_pool_slot` is assigned by the planner when it begins
emitting chunks for a new batch. It transitions: FREE → FILLING →
READY → HELD_BY_USER → FREE (after `damacy_release`).

## Buffer ownership

| Buffer | Sized as | Owner | Lifetime |
|---|---|---|---|
| Lookahead queue | `lookahead_batches × batch_size × sizeof(sample)` | damacy | for instance |
| Plan queue | `plan_queue_capacity × sizeof(chunk_plan)` | damacy | for instance |
| Host wave slab × 2 | `host_buffer_bytes / 2` each, page-aligned pinned | wave slot | reused per wave assigned to slot |
| Dev compressed × 2 | mirrors host slab | wave slot | same |
| Dev decompress × 2 | `device_buffer_bytes / 2` each | wave slot | arena reset per wave |
| Dev meta × 2 | small (read_ops + chunk_plans copy) | wave slot | rebuilt per wave |
| Output batch tensor × 2 | `batch_nbytes` each | output pool | reused after release |

## flush() semantics

```c
enum damacy_status damacy_flush(struct damacy* d);
```

- Inserts a **flush marker** at the tail of the plan queue. The marker
  carries the current batch_pool_slot's `batch_id` and the chunks emitted
  so far.
- Runs the same orchestration loop as `damacy_pop`. When the wave
  scheduler reaches the marker:
  - If the marked batch has `chunks_remaining > 0` and at least one chunk:
    truncate it. The output tensor's effective `shape[0]` is set to the
    number of samples that were complete (the assembler clamps writes;
    samples that had no chunks land as zero, but the truncation makes
    them invisible — `shape[0]` reflects only complete samples).
  - The next sample pushed after the marker starts a fresh batch.
- Blocks until all in-flight waves complete and all flushed batches
  transition to READY.
- Idempotent: if the lookahead and plan queues are empty and no waves
  are in flight, returns immediately.
- Stream is resumable: after `flush` returns, `push`/`pop` work normally.

`damacy_destroy` does **not** flush. It marks the instance shut down
(causing pending `damacy_pop` to return `DAMACY_SHUTDOWN`), tells the
io_queue to shut down (`io_queue_destroy` waits for in-flight reads),
synchronizes both CUDA streams (`cuStreamSynchronize`), then releases
all buffers.

## Error model

- Every failure path sets `instance.failed_status` and `instance.failed`.
- `damacy_push`: returns `DAMACY_SHUTDOWN` if failed.
- `damacy_pop`: returns `failed_status` if failed (drains in-flight waves
  first if possible; otherwise abandons them).
- `damacy_flush`: returns `failed_status` if failed.
- `damacy_release`: best-effort; never fails.
- IO failures (open/pread errno) → `DAMACY_IO`.
- Decompress failures (nvCOMP status) → `DAMACY_DECODE`.
- CUDA driver failures (any non-`CUDA_SUCCESS`) → `DAMACY_CUDA`.
- Cross-batch contamination: a failing wave fails *all* batches whose
  chunks were in that wave. Once the stream is in failed state, it stays
  there.

## Bounded resources (fixed at create)

| Resource | Source | Notes |
|---|---|---|
| io_queue threads | `n_io_threads` | one `io_queue` instance |
| Lookahead samples | `lookahead_batches × batch_size` | backpressure point |
| Plan queue chunks | `2 × max_chunks_per_batch × batch_size` | derived; not exposed |
| Output batch tensors | `2 × batch_nbytes` | always double-buffered |
| Pinned host wave slabs | `host_buffer_bytes` | half per slot |
| Device compressed scratch | `host_buffer_bytes` | mirrors host slabs |
| Device decompress scratch | `device_buffer_bytes` | half per slot, arena |
| Device wave-meta scratch | small fixed (per slot) | read_ops + chunk_plans |
| Zarr metadata LRU | `n_zarrs_meta_cache` | small per-entry |
| Shard index LRU | `n_shards_meta_cache` | each ~ chunks_per_shard × 16B |
| FDs | per-key in `store_fs`, no eviction | bounded by `ulimit -n` |

## Metrics

Inspired by chucky's `stream_metric` / `stream_metrics`:

```c
struct damacy_metric {
  const char* name;
  float       ms;            // cumulative
  float       best_ms;       // best single observation (1e30f = none)
  double      input_bytes;   // cumulative bytes consumed by stage
  double      output_bytes;  // cumulative bytes produced by stage
  uint64_t    count;
};

struct damacy_stats {
  // Pipeline stages (per-wave timings except where noted)
  struct damacy_metric plan;        // CPU planner work
  struct damacy_metric io;          // io_queue read time
  struct damacy_metric h2d;         // cuMemcpyHtoDAsync wave slab
  struct damacy_metric decompress;  // nvCOMP batched decompress
  struct damacy_metric assemble;    // gather kernel

  // Orchestration stalls (wall-clock the user thread is blocked in pop)
  struct damacy_metric pop_wait_io;       // poll spin while waiting on io_event
  struct damacy_metric pop_wait_compute;  // sync on compute_event
  struct damacy_metric push_backpressure; // time spent returning AGAIN
  struct damacy_metric flush_wait;        // total time inside damacy_flush

  // Cache effectiveness
  uint64_t zarr_meta_hits, zarr_meta_misses;
  uint64_t shard_idx_hits, shard_idx_misses;

  // Counters
  uint64_t batches_emitted;
  uint64_t batches_truncated;   // by flush
  uint64_t waves_emitted;
};

void damacy_stats_get(const struct damacy* d, struct damacy_stats* out);
void damacy_stats_reset(struct damacy* d);
```

## Position statements

- **Append-only plan queue, not per-batch arrays.** Coalescing across
  batches needs a continuous queue; plan-per-batch would force a
  re-merge step. The append model also makes flush a simple sentinel.
- **Two LRUs share one implementation.** Generic robin-hood + intrusive
  LRU list, parameterized by key/value. No deletion: insert evicts the
  longest-probing entry when probe distance exceeds the cap; same cap
  bounds worst-case lookup. Refcount-pinning so an in-use shard index
  isn't evicted underfoot.
- **Wave-and-batch double buffering, no deeper.** B=2 W=2 is sufficient
  to overlap IO+compute with user training when either side is the
  limit. Deeper buffering only averages latency.
- **Main-thread orchestration.** Avoids CUDA host callbacks (and the
  internal-driver-thread restrictions they impose), avoids an
  orchestrator thread (and the queues / locks / wakeups it would need),
  and gives a single sequential narrative for what the pipeline does.
  Cost: poll-sleeps in the wait path. Acceptable v1 trade.
- **Per-key FD cache.** The original design called for no FD cache
  (per-`read_op` `open`/`close` is cheap once the inode is hot, and
  it dodges `ulimit -n`). The implementation now caches FDs in
  `store_fs` because the syscall + path-resolution savings showed up
  in profiling. No eviction yet — revisit if `ulimit -n` becomes a
  problem against very large datasets.
- **`io_queue` for IO parallelism.** Reuses chucky's pattern: post jobs,
  record an event, wait on the event. Maps cleanly onto the wave's
  IO-done synchronization point.
- **Two CUDA streams (`h2d`, `compute`).** Decompress and assemble
  share `compute` so we don't need an inter-stage event between them.
- **Fail-the-stream, not fail-the-batch.** With waves crossing batch
  boundaries, a failed read can't be cleanly "blamed" on one batch.
  Easier to put the stream in a terminal failure state and require
  destroy+recreate to recover. Fits typical training loops (a sample
  failure is usually a setup bug worth aborting on).
- **Mip selection.** Out of scope for v1 internals — planner reads at
  level 0 only. Auto-mip lands later as a planner pre-pass.

## Build order

1. **Public C API surface.** Header + opaque `damacy*` + stubs that
   return `DAMACY_OK` / `DAMACY_AGAIN` skeletally. Lets `bench/main.c`
   switch to the real interface immediately.
2. **Generic LRU** (`lru_table`) + zarr metadata cache + shard index
   cache. Pure CPU; testable.
3. **Planner**: samples → `chunk_plan[]`. Page-aligned `read_op`s; *no*
   coalescing, *no* waves. One read per chunk, padded to page boundary.
   Unit-testable on synthetic AABBs.
4. **Naive end-to-end (single wave per batch).** Single wave slot,
   single batch slot, no overlap; io_queue + cuda streams used but each
   stage runs to completion before the next. Real assemble kernel
   writing to the output tensor. Correct, measurable.
5. **Wave scheduler + double buffering (W=2, B=2).** Pull-prefix from
   plan queue; orchestration state machine in `damacy_pop`; cuEvents +
   io_events for sync.
6. **`damacy_flush` + `damacy_stats`.** Flush marker, idempotent
   semantics, metrics collection.
7. **Coalescing in the planner / scheduler** (merge adjacent
   `read_op`s that cover the same page-aligned region within a wave).
8. **IO thread tuning.** `io_queue` size; backpressure on the queue.

Steps 1–4 produce a correct, slow, simple end-to-end system you can
profile. Step 5 adds the overlap. Step 6 adds the user-visible drain
primitive and the visibility to tune. Step 7 is the IO win once the
shape is stable.
