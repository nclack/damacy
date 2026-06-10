# Metadata prefetch

On networked storage, the per-read latency on small metadata
objects (zarr.json, shard indices, per-array chunk_layout
probes) is the throughput ceiling — not the chunk reads
themselves. A batch can't be planned until its metadata is
resolved, and serialized metadata reads at 50–200 ms each
starve the GPU even when chunk IO is fully parallel.

In-scope metadata is host-side and per-array or per-shard.
Per-chunk concerns (notably blosc1 `bstarts`) are decode-side
and not part of this design.

This document describes the layer that hides that latency:
an eager, multi-stage prefetcher feeding a set of pinning LRU
caches.

**Scope.** This is the *metadata* side. Chunk prefetching
(into pinned host / GPU memory, possibly via GDS) is a
separable concern with different memory and stream
orchestration — covered elsewhere.

## Concepts

Two names; nothing else needs to be introduced.

- **`prefetch_cache`** — the data structure. Generic over
  key and value type. One instance per phase: array
  metadata (zarr.json), shard indices, per-array
  `chunk_layout` (blosc block layout probed from a
  chunk's first 16 bytes). Each instance owns its own
  LRU, pending queue, ordinal watermark, and depth knobs.
- **`prefetcher`** — the orchestrator. A single component
  that consumes the lookahead stream of upcoming samples,
  enumerates the keys each sample needs at each stage, and
  pushes requests through the caches in dependency order.

The chunk planner consumes from the final cache stage — it
asks for already-resolved metadata when materializing a batch
plan. It does not enumerate keys or interact with the
prefetcher directly.

## prefetch_cache

A pinning LRU with request deduplication and async fill.

### State per entry

```
{ pending | ready | error{code} }
```

`error` is terminal; it folds in both IO failure and
upstream-cancellation (the stage that would have produced
this entry's key failed, so the entry will never become
ready). Consumers check one state, see one outcome.

### Handle

```
typedef struct { uint32_t slot; uint32_t generation; } prefetch_handle_t;
```

8 bytes, no allocation. `generation` guards slot reuse if
the entry is evicted before the handle is consumed. In
normal operation the watermark only advances after consumers
are done; the guard catches ordering bugs.

### API shape

```
prefetch_handle_t prefetch_cache_request(c, key, batch_id, &gate);
const blob_t     *prefetch_cache_try_get(c, h);
void              prefetch_cache_advance_watermark(c, batch_id);
```

`request` returns a handle. The entry's ordinal range is
widened to include `batch_id`; if it was a miss, a fetch is
enqueued on the injected executor and `gate.pending` is
incremented (see *Readiness gate* below). `try_get` returns
the blob pointer for `ready` entries, NULL for `pending`,
and surfaces the error code for `error`.

There is no per-handle release. `advance_watermark(w)` is
called by the scheduler when batch `w−1` is fully planned
and its metadata is no longer needed; entries with
`max_batch_id < w` become eligible for eviction.

### Ordinal-range pinning, sized to never saturate

The cache is driven by a monotonic batch ordinal stamped on
each sample by lookahead. Each entry holds a `[min_batch_id,
max_batch_id]` range covering every batch that has requested
the key; on every `request(c, key, batch_id, …)` the cache
widens the range. Eviction is gated on a global watermark:
an entry is evictable iff `max_batch_id < watermark`. The
scheduler advances the watermark when a batch leaves the
planner queue (see *Integration*).

This replaces explicit refcount pinning. Consumers never
call `release`; the prefetcher's lead time effectively pins
entries until the scheduler proves they're done. The
monotonic ordinal makes "done" a single integer compare
instead of a multi-handle refcount.

Each cache pins every sample seq in `[watermark, pushed_samples)`
(an entry stays pinned while `max_owner_id >= watermark`). Two
bounds size that pinned set:

- **Push back-pressure** bounds `pushed_samples −
  next_consume_seq <= lookahead_samples`
  (`damacy_push` stalls with `DAMACY_AGAIN` at the floor).
- **The watermark lags `next_consume_seq`.** `next_consume_seq`
  advances when the planner *takes* a ready wave
  (`prefetcher_take_ready_wave`), but the watermark only advances
  later, at `plan_commit`, once a batch is sealed. The consumed-
  but-not-yet-committed samples sit in the staging window, capped
  at `2 * samples_per_batch` (`planning_capacity_locked`). So
  `next_consume_seq − watermark <= 2 * samples_per_batch`.

Adding the two terms, the worst-case simultaneously-pinned set per
cache is `lookahead_samples + 2 * samples_per_batch` keys (times
the per-sample shard footprint for shard_index). The earlier
design claimed `lookahead_samples` alone; that ignored the staging
lag and under-sized the floors — distinct-URI workloads with deep
lookahead could exceed `lookahead_samples` pinned keys and saturate
the cache.

`damacy_config` validation enforces a floor so each cache can hold
its entire worst-case in-flight set, including the lag term:

```
n_array_meta_cache   >= lookahead_samples + 2*samples_per_batch
n_chunk_layout_cache >= lookahead_samples + 2*samples_per_batch
n_shard_index_cache  >= (lookahead_samples + 2*samples_per_batch)
                        * max_shards_per_sample
```

Each floor violation is rejected at `damacy_create` with
`DAMACY_INVAL` and an actionable message that names the knob,
the observed vs. required value, and the concrete fix (which
knob to raise and to what minimum). The cache sizes remain
explicit tuning knobs — useful as reuse/hit-rate levers above
the floor.

`max_shards_per_sample` is the one footprint not knowable at
config time (shard geometry is only read at runtime), so it is
declared as an explicit, validated (`> 0`) tuning knob. It does
double duty: it sizes the shard_index floor above, and at
runtime the prefetcher rejects any sample whose AABB intersects
more shards than declared (`DAMACY_INVAL`, with a message that
reports the observed vs. allowed shard count). That runtime cap
is what makes the declared bound sound — without it an
under-declared value could still overrun the cache.

With the floors enforced, the oldest in-flight sample can
always be admitted in normal operation. As **defense in depth**,
the cache still degrades gracefully if "every entry pinned" is
ever reached anyway — a residual overshoot, a future refactor, or
a caller that bypassed validation. `prefetch_cache_request_result`
returns `DAMACY_AGAIN` (logging a warning that names the cache
knob) rather than `abort()`-ing the process. The prefetcher treats
that as back-pressure: it leaves the request un-admitted (or holds
the in-progress sample in its current stage) and retries on a later
tick, once the scheduler advances the watermark and frees a pin.
Shard enumeration resumes from a per-slot cursor so a mid-stream
stall never re-issues — and thus never double-counts — an already-
issued request's gate.

### Readiness gate

For a consumer with many handles (a batch with N metadata
keys), per-handle polling is wasteful. Each `request` takes
a `prefetch_gate` — a single atomic word packing a pending
count with an error flag:

```
struct prefetch_gate { _Atomic uint64_t state; };  // [error:1 | pending:63]
```

The cache atomically increments `pending` on miss; the IO
thread decrements on fetch completion and sets the `error`
bit on failure. The consumer checks one atomic load to
know whether the whole group is ready *and* whether any
handle errored.

The planner uses one gate per batch; per scheduler tick it
loads each pending batch's gate. Zero pending → batch
graduates into planning. Error bit set → batch fails fast.

The gate is granularity-agnostic. The current planner uses
batch-grouped gates; a future wave-shaped planner can
assemble gates per wave or per sample with no change to
the cache.

### Executor

The cache does not perform IO directly. It is constructed
with an injected executor — today the existing `io_queue`
threadpool, swappable for a dedicated thread or a mock for
test isolation. The cache holds a `fetch_fn(key) → blob`
callback that knows how to read and parse one entry of
that phase's content.

## prefetcher

One component, running on a dedicated thread that
blocking-pops samples from lookahead. For each popped
sample, it advances a small state machine through the
phases:

```
sample → request(array_meta_cache, key)
       → on ready: request(shard_index_cache, derived_keys)
       → on ready: request(blosc_extras_cache, derived_keys)
       → mark sample ready-for-planner
```

Each stage's keys are derived from the prior stage's
resolved values; there is no shared enumeration logic
between the prefetcher and the planner. The prefetcher
advances reactively — when a cache entry becomes ready,
it derives the next stage's keys for the samples that
were waiting on it.

### Lead time

Prefetch depth is governed by per-cache pin capacity, not
by GPU memory or wave count. With small metadata objects
and a few hundred MiB of host RAM, lead times of 10k–100k
samples are reasonable — enough to hide multi-second
storage tails on networked backends.

### Error propagation

A failed fetch transitions the entry to `error{code}`.
The prefetcher, when it sees an upstream error for a
sample, marks any dependent stages for that sample as
cancelled — and the cancellation surfaces through the
same `error` state. Downstream consumers (eventually the
chunk planner) see a single error transition and fail the
batch cleanly.

### Sized to fit, with a back-pressure safety net

The cache-size floors (see *Ordinal-range pinning, sized to never
saturate*) guarantee every cache can hold the whole in-flight
working set plus the staging lag, so under a validated config
admission is never refused for lack of an evictable slot, and the
prefetcher never stalls a stage in normal operation.

If saturation is ever hit anyway, the cache returns `DAMACY_AGAIN`
and the prefetcher applies back-pressure: an array_meta stall holds
the popped sample and replays it next tick; a shard_index /
chunk_layout stall keeps the slot in its current stage and retries.
Both clear once the watermark advances. This is a recoverable
degradation, not a crash.

The single thing that could violate the shard_index sizing —
a sample that intersects more shards than `max_shards_per_sample`
— is rejected up front in `advance_from_meta`: when the
`sample_shard_iterator` count exceeds the declared bound the
sample fails with `DAMACY_INVAL` and a message reporting the
observed vs. allowed shard count and how to raise the bound
(and the matching `n_shard_index_cache`). A configuration that
previously hit the saturation path now either validates and
runs, or fails fast at `damacy_create`.

## Integration

```
lookahead ──► prefetcher ──► prefetch_cache (array meta)
  (stamps                │                ▲
   batch_id)             ├──────────────► prefetch_cache (shard index)
                         │                ▲
                         └──────────────► prefetch_cache (chunk_layout)
                                          ▲
                            executor (io_queue) ──┘

planner    ◄── try_get(handle) + gate poll
scheduler  ──► advance_watermark(batch_id+1) on plan success
```

Lookahead stamps a monotonic `batch_id` on push. The
prefetcher carries it through every `request` call so each
cache can maintain ordinal ranges.

`planner_plan` accepts the batch's `prefetch_gate*`. On
each scheduler tick it loads the gate; non-zero pending
returns `DAMACY_PENDING` (the scheduler retries next tick).
Zero pending dereferences each handle via `try_get` (now
guaranteed `ready`) and produces the chunk plan. On
successful planning, the scheduler broadcasts
`advance_watermark(batch_id + 1)` to all three caches.

## Open knobs

- Per-cache capacity and in-flight depth. Likely
  workload-dependent; expose as config.
- Executor choice per cache. Default to the shared
  `io_queue`; allow override (e.g. a dedicated thread)
  for backends where metadata and chunk reads contend.
- Fetch unit granularity. The `fetch_fn` for one phase
  may legitimately submit a single read; for another it
  may batch (a directory listing populating many shard
  index entries at once). The cache primitive does not
  constrain this — the callback decides.
- Watermark advance point. Today the scheduler advances
  after a successful plan, freeing metadata as soon as
  the batch enters the IO/decode pipeline. A safer
  variant would advance on wave-pool slot release —
  larger pin window, smaller eviction risk.

## What this does not address

- Chunk-side prefetching into pinned host or GPU
  buffers. Same orchestration shape might apply, but
  the memory model (refcount against a GPU budget, not
  host RAM) and the stream coordination are
  meaningfully different. Treated as a separate design.
- Per-chunk blosc1 sub-stream offsets (`bstarts`). These
  are per-chunk and only needed on the device for decode;
  they belong inside the decode pipeline as a kernel
  scheduled ahead of the nvcomp kernels, not as host-side
  metadata prefetch. The per-array `chunk_layout` probed
  here is a different artifact — uniform across an array's
  chunks, parsed host-side once.
- Wave-shaped planner coalesce. The planner is batch-
  shaped today and stays so; the `prefetch_gate` is
  granularity-agnostic, so a later restructure that
  coalesces at wave granularity (a sub- or super-batch
  unit) is unblocked.
- Cross-process / cross-GPU sharing of the metadata
  caches. The injectable executor and opaque cache
  surface preserve this as a future option (shmem or
  a resolver daemon) without requiring it now.

## Plan

Single PR. All three caches, the prefetcher thread, and the
planner `DAMACY_PENDING` wiring land together so the cache
primitive is validated against the real planner — not a mock.

### Work items

1. **`src/prefetch/prefetch_cache.{h,c}`** — the primitive.
   - Built on `util/lru.h` with the eviction predicate
     swapped from refcount to `max_batch_id < watermark`.
   - State machine `{pending | ready | error{code}}`,
     ordinal-range tracking, `prefetch_gate`, injected
     `fetch_fn`, injected executor.
   - Unit tests: state transitions, ordinal eviction,
     dedup, gate count + error packing, sticky errors,
     admission rejection.

2. **`src/prefetch/prefetcher.{h,c}`** — orchestrator.
   - Dedicated thread; blocking-pop from lookahead;
     per-sample state machine across the three stages.
   - Parking table for samples waiting on stage-N
     completion. Worker completion callbacks wake parked
     samples and post their next-stage request.

3. **Three cache instances** wired into
   `damacy_create` / `damacy_destroy`:
   - `array_meta_cache` — `fetch_fn` reads `zarr.json`,
     parses (today's `zarr_meta_cache_get` body).
   - `shard_index_cache` — `fetch_fn` reads the tail index
     of a shard.
   - `chunk_layout_cache` — `fetch_fn` probes the
     first-chunk blosc header (today's
     `zarr_chunk_layout_probe`).

4. **Removed** — content migrates into `fetch_fn`s:
   - `src/zarr/zarr_meta_cache.{h,c}`
   - `src/zarr/zarr_shard_cache.{h,c}`
   - `meta_entry.layout` / `layout_probed` fields.

5. **`src/lookahead/`** — extend:
   - Stamp monotonic `batch_id` on push.
   - Cond-var signal on push so the prefetcher can
     blocking-pop.

6. **`src/zarr/sample_shard_iterator.{h,c}`** — extract
   the sample-AABB → unique-shard-coord enumeration that
   currently lives inline in `planner.c` (`chunk_range` at
   line 73 plus the dedup loop around line 494). POD
   iterator (`_init` / `_next` per the project convention).
   Planner uses it as its outer loop; prefetcher uses it to
   enumerate stage-2 (shard_index) keys. Single source of
   truth for "which shards does this sample touch."
   Stages 1 and 3 key on the array URI directly — no
   enumeration helper.

7. **`src/planner/planner.c`** — synchronous cache calls
   replaced with handle dereferences via `try_get`.
   `planner_plan` accepts the per-batch `prefetch_gate*`;
   returns `DAMACY_PENDING` when `gate.pending > 0`.

8. **`src/scheduler/scheduler.c`** — pending-batch retry
   loop. On plan success, broadcast
   `advance_watermark(c, batch_id + 1)` to all three caches.

### Tests

- Unit (per cache): state machine, ordinal eviction,
  dedup, gate counting, sticky errors, admission rejection.
- Integration: cold-start batch returns `DAMACY_PENDING`
  from the planner; prefetcher warms; subsequent tick
  plans successfully. Underrun: induce a slow `fetch_fn`,
  observe repeated `DAMACY_PENDING` plus a
  planner-starvation counter incrementing.
