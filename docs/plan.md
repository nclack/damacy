# damacy implementation plan & status

> Living document — update as steps complete and conventions land.
> Design is in `api-design-surface-draft.md` (public API) and
> `api-design-internals-draft.md` (architecture). The build order
> originates in the internals doc; this file tracks current state and
> records implementation decisions made along the way.

## Build-order status

`git log --oneline` is authoritative; this is a quick map.

- [x] **1. Public C API surface + stubs** — `0779aec stubbed public api`.
  Files: `src/damacy.h`, `src/damacy.c`, `src/damacy_status.c`,
  `src/limits.h` (extended). `bench/main.c` ported to push/pop/flush.
- [x] **2. Generic LRU + zarr metadata + shard index caches.**
  - `f3a5401 log: stderr logger + prelude check macros` —
    `src/log/log.{h,c}`, `src/util/prelude.h`.
  - `0b41ff2 lru: pinned cache + hash helpers` (initial linear-probe
    impl) → `c4ad8ba lru: robin-hood + factored structs` (rewrite).
    Files: `src/util/lru.{h,c}`, `src/util/hash.h`, `tests/test_lru.c`.
  - `fab4801 zarr: meta cache (lru-backed)` —
    `src/zarr_meta_cache.{h,c}`, `tests/test_zarr_meta_cache.c`.
  - `9e380c2 zarr: shard index cache` —
    `src/zarr_shard_cache.{h,c}`, `tests/test_zarr_shard_cache.c`.
- [x] **3. Planner.** `7f4fc5c planner: page-aligned chunk plans` —
  `src/planner.{h,c}`, `tests/test_planner.c`. `struct read_op` and
  `struct chunk_plan` shapes defined here; wave-scheduler fields
  (`dst_buf_offset`, `dev_decompressed_offset`) reserved but zeroed.
- [x] **4. Naive end-to-end (single wave per batch).** Synchronous
  push → plan → store_read → cuMemcpyHtoDAsync → nvCOMP zstd
  decompress → assemble kernel → cudaStreamSynchronize. Validated
  byte-for-byte against zarr-python on a `gen_dataset.py` fixture.
  Files: `src/assemble.{h,cu}`, rewritten `src/damacy.c`,
  `bench/main.c` driving real samples. New: `damacy_config.store_root`
  (resolves sample.uri against a single fs-backed store; multi-store
  resolution lands later). v1 caps inside damacy.c:
  `DAMACY_MAX_CHUNKS_PER_BATCH = 256`,
  `DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES = 1 MB` — the product gates
  nvCOMP's temp scratch via `cudaMalloc`, so they can't be raised
  casually. Output tensor is allocated once at the first batch from
  the first sample's AABB; subsequent samples must match the same
  per-axis shape. Single batch slot, no overlap; double-buffering
  lands in step 5.
- [x] **5. Wave scheduler + double buffering (W=2, B=2).** Two batch
  slots in flight, two wave slots overlapping IO + H2D + decompress +
  assemble. `damacy_pop` is a state-machine loop (advance → kick →
  return) with `platform_sleep_ns(50µs)` poll-waits between event
  checks; no internal threads beyond the existing `n_io_threads`.
  Waves do not cross batch boundaries (read-op coalescing is step 7).
  Files: rewritten `src/damacy.c` (`struct damacy_wave`,
  `struct damacy_batch_slot[2]`, two streams `stream_h2d` /
  `stream_compute`, per-wave `struct decoder` — see gotcha below).
  New API surface: non-blocking `io_event_query` /
  `store_event_query` for poll-without-block of pending reads.
  Planner change: `read_op.shard_path` is inlined (cap
  `DAMACY_MAX_PATH = 224`) so plan entries survive across batches
  without dangling pointers into planner-owned storage.
- [ ] **6. `damacy_flush` + `damacy_stats`.** Flush marker, idempotent
  semantics, cumulative metrics collection.
- [ ] **7. Coalescing in the planner / scheduler.** Merge adjacent
  `read_op`s within a wave (planner-side; the IO and decompress paths
  are unchanged).
- [ ] **8. IO thread tuning.** `io_queue` size, backpressure on the
  queue.

Steps 1–4 produce a correct, slow, simple end-to-end system you can
profile. Step 5 adds the overlap. Step 6 adds the user-visible drain
primitive and the visibility to tune. Step 7 is the IO win once the
shape is stable.

## Implementation conventions landed in steps 2–3

These aren't visible from the design docs' "premises" section — they're
concrete style/contract decisions made during implementation. New code
should follow them.

### Code style

- **Headers contain declarations only.** No `static inline` function
  bodies, no `inline` definitions. If a helper needs a function, the
  function lives in a `.c` file with its own library target. Macros
  are fine in headers — they're not implementation code.
- **`self` is the receiver pointer name.** Methods on a struct take
  the canonical pointer parameter as `self`, e.g. `lru_get(struct lru*
  self, ...)`. Improves readability and makes "this is a method on the
  receiver" structurally obvious.
- **Variable naming scales with scope and includes units.**
  Short loop counters (`i`, `d`) and tightly-scoped temporaries are
  fine. Anything that survives across multiple lines or out of a
  block uses a descriptive name. Quantities encode units in the name:
  `chunk_size_bytes`, `n_inner_per_shard`, `decompressed_n_bytes`,
  `aligned_file_offset`. No `c`/`l`/`p`/`e`/`s` for objects with a
  lifetime longer than three lines.
- **Logging + checks split between `util/prelude.h` and `log/log.h`.**
  `prelude.h` provides `CHECK(label, expr)` / `CHECK_SILENT(label, expr)`
  / `CHECK_MUL_OVERFLOW` plus `countof` / `container_of`. CHECK
  expansions reference `log_error`, so any `.c` that uses CHECK includes
  `log/log.h` directly. Parse-only modules that only need `countof`
  (e.g. `zarr_metadata.c`) include prelude alone, no logger dependency.
  CHECK replaces `if (!cond) return ...` for precondition checks and
  `if (!ptr) goto Fail` for resource-acquisition results; errors funnel
  through a single CamelCase `Error:` / `Fail:` / `Cleanup:` /
  `Invalid:` / `CudaFail:` / `InvalidArg:` / `Rank:` label that frees
  partial state and returns the right status. `prelude.h` is private
  — never include from a header. See `src/util/lru.c::lru_create` and
  `src/damacy.c::damacy_create` for canonical patterns.
- **CUDA-runtime check macro `CU(label, expr)`.** Defined locally per
  CUDA-using `.c` (`src/damacy.c`, `src/decoder.c`) — wraps a
  `cudaError_t`-returning call, logs via `log_error`, gotos `label`.
  Same shape as `CHECK`. `src/decoder.c` also defines `NV(label, expr)`
  for `nvcompStatus_t`-returning calls.
- **Public methods check `self` consistently.** Every public entry
  point begins with `CHECK_SILENT(InvalidArg, self)` (or a similarly
  labeled equivalent) before any dereference. A missing check on one
  method is a bug, not a "tighter precondition".
- **NULL-safe destroy functions, no caller-side `if(p) destroy(p)`.**
  Every `*_destroy` / `*_free` checks `if (!self) return;` (or
  `CHECK_SILENT(Out, self)`) at the top so callers can call them
  unconditionally. `cudaFree(NULL)` and `cudaFreeHost(NULL)` are
  documented no-ops — drop the guards. `cudaStreamDestroy(NULL)`,
  `cudaEventDestroy(NULL)`, and `cudaStreamSynchronize(NULL)` are
  NOT no-ops (NULL = the legacy default stream); keep guards.
  When a create function's `Fail:` block has multi-line teardown,
  factor a static `*_free_partial` helper that does the guarded
  container deref (see `src/store_fs.c::store_fs_free_partial`,
  `src/threadpool/threadpool.c::threadpool_free_partial`,
  `src/io_queue/io_queue.posix.c::io_queue_free_partial`).
- **Many-arg functions: group invariants into a context struct.** When
  a function takes ≥5 args and several are constant across the call
  site's loop, fold them into a per-call `struct foo_ctx` and pass
  `const struct foo_ctx*`. See `planner.c::emit_chunk` and its
  `struct emit_ctx`.
- **POSIX is fair game.** `strdup`, `pthread_*`, `mkdtemp`,
  `clock_gettime` etc. are used directly. Don't introduce wrapper
  functions claiming "portability" without a concrete non-POSIX target.
- **Designated initializers.** `*self = (struct foo){ .a = ..., .b =
  ... }` instead of field-by-field assignment chains. Omitted fields
  are zero-initialized — don't spell out `= 0` for trailing fields.
- **Factored sub-structs.** Top-level structs are grouped by
  responsibility. e.g. `struct lru` contains `lru_index`, `lru_list`,
  `lru_freelist`, `lru_counters` so helpers operate on `struct lru*`
  but the field grouping is visible at a glance.
- **Fixed-width types in headers and storage.** `uint32_t` / `uint64_t` /
  `int64_t` / `int32_t` everywhere. `size_t` only for `<string.h>`-style
  locals.

### LRU contract (`src/util/lru.h`)

- `struct lru_entry*` is the public handle; pointer is **stable** for
  the entry's lifetime (slots never move; only the index reorders).
- Both `lru_get` and `lru_put` take an explicit `probe_key` argument.
  The LRU never interprets keys; collisions are disambiguated via the
  caller's `ops.eq(value, probe_key, user)`.
- **Pinning** (`lru_entry_acquire` / `lru_entry_release`): pinned
  entries (refcount > 0) are immovable. The robin-hood walk skips them
  rather than swap. Backshift on evict skips pinned (gaps may persist
  past pinned cells). Replacement on a pinned key returns NULL +
  increments `put_failures`. Lookup walks the full `max_probe` (no
  early-terminate on EMPTY) because pin-aware insertion can leave
  gaps.
- Caches expose pinning lazily: v1's `zarr_meta_cache` /
  `zarr_shard_cache` don't yet wrap acquire/release; add when the
  scheduler needs concurrent reads through a cache.

### Cache value lifetime

- `zarr_meta_cache_get` and `zarr_shard_cache_get` return pointers
  owned by the cache, valid until the entry is evicted. v1 has *no
  pinning at the cache layer* — callers must not retain pointers
  across other cache calls. Adding acquire/release wrappers is
  straightforward when needed.

### Planner contract (`src/planner.h`)

- `planner_plan` writes into caller-supplied buffers (struct
  `planner_output` with capacities). Returns `DAMACY_OOM` if either
  the read_ops or chunk_plans buffer fills mid-plan.
- `read_op.shard_path` strings are **planner-owned**, valid until the
  next `planner_plan` call. (Within one call, multiple chunks in the
  same shard share a path string.)
- Empty chunks (sentinel `offset == ZARR_SHARD_EMPTY_OFFSET`) are
  silently skipped — the assembler treats the corresponding output
  region as zero, so callers must zero-initialize the output tensor.
- `chunk_plan.dst.dims[0]` is the leading sample-index axis; the
  remaining `meta->rank` dims are the intersection in sample-local
  coordinates. `chunk_plan.src` is in chunk-local coordinates.
  `src_strides` are in elements (not bytes).
- Wave-scheduler fields (`read_op.dst_buf_offset`,
  `chunk_plan.dev_decompressed_offset`) are zeroed by the planner;
  scheduler fills them in at step 5.

### Test fixture pattern

Shared helpers live in `tests/expect.h` (`EXPECT` / `RUN` macros, both
lowering onto the project logger) and `tests/fixture.h` (file-system
test scaffolding: `fixture_write_file`, `fixture_write_synthetic_shard`,
`fixture_write_zarr`, `fixture_rm_tree`, plus the LE encoders). All
test executables include expect.h; the fs-backed integration tests
(test_zarr_meta_cache, test_zarr_shard_cache, test_planner,
test_damacy) include fixture.h.

`main` is a flat list of `RUN(test_foo)` calls — no table-driven loops,
no continuing past failures. RUN fast-fails the executable on the first
failing test so cascading failures don't mask the root cause.

Two fs fixture flavors:

- **Synthetic shards** (`fixture_write_synthetic_shard`) for cache and
  planner tests where mechanic-level control over the shard's offset
  table matters more than real chunk content. `mkdtemp` for the root,
  a per-test `MINIMAL_ZARR_JSON` literal at `<root>/<uri>/zarr.json`,
  hand-built shard footers at `<root>/<uri>/c/<a>/<b>/...`, `rm -rf`
  via `system()` for cleanup. See `tests/test_planner.c::fixture_init`.

- **Real zstd zarrs** (`fixture_write_zarr`) for end-to-end damacy
  tests where the nvCOMP decompress path must run on real bytes. The
  helper shells out to `tests/write_zarr.py` (uv-script with PEP-723
  inline deps, depends on `zarr>=3` + `numpy`); content is a
  deterministic row-major linearization plus a `--offset` so multi-zarr
  tests can distinguish sources. See `tests/test_damacy.c`. Requires
  `uv` on PATH (provided by `flake.nix` devShell).

## Patterns surfaced by step 5

- **Per-wave nvCOMP decoder.** `decoder_decompress_batch` populates
  internal pinned-host fanout arrays synchronously, then queues a
  `cudaMemcpyAsync` for them. With two waves sharing one decoder, the
  second wave's call overwrites those host arrays *before* the first
  wave's queued copy has actually run — the in-flight DMA reads the
  wrong pointers and the kernel decompresses garbage. Each wave gets
  its own `struct decoder*` (allocated in `wave_init`, freed in
  `wave_destroy`). Pattern applies to anything that owns pinned-host
  scratch consumed by a queued copy: don't share across in-flight
  pipeline stages without explicit sync.
- **Inline-string struct fields for cross-call lifetime.**
  `struct read_op.shard_path` was a planner-owned `const char*`,
  freed on each `planner_plan` call. Once the wave scheduler holds
  read_ops across batches that's a use-after-free. Inlining the
  string (`char shard_path[DAMACY_MAX_PATH]`) decouples lifetime; the
  cap is a hard fail (`DAMACY_OOM`) at planner emit time.
- **Wave state machine + poll-sleep waits.** Three observable stages
  (`WAVE_IO`, `WAVE_H2D`, `WAVE_ASSEMBLE`) with `cudaEventQuery` or
  `store_event_query` driving transitions. `cudaEventQuery` returns
  `cudaErrorNotReady` for "not yet" (which is *not* a failure) —
  treat that distinct from real errors. The pop loop sleeps
  `DAMACY_POP_POLL_NS = 50µs` between advance/kick passes when no
  batch is READY but work is in flight; cheap and non-spinning.
- **Lazy output allocation, both slots up front.** Output shape is
  established at the first sample's AABB. `batch_pool_allocate`
  allocates `cfg.batch_size × sample_volume × bpe` for *both* batch
  slots on that first call, so subsequent slot reuse only resets
  state, not the device pointer.
- **Slot-reuse data hazard.** Reused batch slots inherit whatever
  data the previous batch wrote. If the planner skips an empty
  chunk, the corresponding output region keeps the old data. The
  current code zeroes the dev_ptr only for the degenerate
  zero-chunk case (where assemble doesn't run at all) — partial
  empty-chunk batches in zarrs with sentinels would surface this.
  Not exercised by any test today (`gen_dataset.py` produces dense
  arrays); revisit if it becomes a real workload.

## Patterns from review

Two review passes against steps 1–3 produced concrete cleanups; the
durable conventions they surfaced are documented above (Code style,
LRU contract, Planner contract, Test fixture pattern). One additional
pattern worth calling out:

- **Partial-batch operations must surface failure, not pretend success.**
  When submitting N items and item k fails, return a sentinel the caller
  can detect, and drain any in-flight work first so it doesn't touch
  caller buffers after return. See `store_fs.c::fs_submit` (sentinel:
  `seq == 0`).

`git log` is authoritative for the per-change history.

## What I'd revisit

- **Cache pinning plumbing.** Step 5 (wave scheduler) likely surfaces
  the need for `zarr_*_cache_acquire/release` so scheduler-held shard
  indices can't be evicted underfoot.
- **`DAMACY_RANK` vs `DAMACY_INVAL`.** I split RANK out as its own
  status; if it never fires usefully, fold back into INVAL.
- **`src/util/lru.c` size**. ~500 LOC of carefully-factored code; could
  split helpers (list / freelist / index) into separate `.c` files if
  it grows again.
- **`store_root` config field.** v1 binds an instance to one filesystem
  root. The surface design wants per-uri resolution (`s3://...`,
  multiple roots, etc.); the wave scheduler is in now and the
  multi-backend story is the natural next surface change.
- **Empty-chunk handling on slot reuse.** See "Slot-reuse data
  hazard" above. Cheapest fix is to `cudaMemsetAsync(slot->dev_ptr,
  0, n_bytes, stream_compute)` at the start of each batch's first
  wave; cost is one full-tensor write per batch.
- **Cross-batch waves.** Step 5 keeps each wave inside one batch
  slot. The design doc envisions waves spanning batch boundaries
  (reading the last chunks of K alongside the first of K+1, fusing
  the IO). Step 7 (coalescing) is the natural place to revisit since
  it touches the same merging logic.
- **`DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES = 1 MB` cap.** Tight enough
  to fit nvCOMP's scratch on a 8GB-class laptop GPU; will bite as
  soon as someone configures larger inner chunks. Move it (and
  `DAMACY_MAX_CHUNKS_PER_BATCH`) onto `damacy_config` once we know
  the right knob shape.
