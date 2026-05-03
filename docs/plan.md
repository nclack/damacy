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
- [ ] **5. Wave scheduler + double buffering (W=2, B=2).** Append-only
  plan queue + chunk-granular scheduler in `damacy_pop`; cuEvents +
  io_events for sync; main-thread orchestration with poll-sleep waits.
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
  Errors funnel through a single `error:` (or `invalid:`,
  `precondition:`, etc.) label that frees partial state and returns the
  right `damacy_status`. CHECK replaces `if (!cond) return ...` for
  precondition checks. `prelude.h` is private — never include from a
  header. See `src/util/lru.c::lru_create` for the canonical pattern.
- **Public methods check `self` consistently.** Every public entry
  point begins with `CHECK_SILENT(Fail, self)` (or its label-named
  equivalent) before any dereference. A missing check on one method is
  a bug, not a "tighter precondition".
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
`fixture_rm_tree`, plus the LE encoders). All test executables include
expect.h; the fs-backed integration tests (test_zarr_meta_cache,
test_zarr_shard_cache, test_planner) include fixture.h.

`main` is a flat list of `RUN(test_foo)` calls — no table-driven loops,
no continuing past failures. RUN fast-fails the executable on the first
failing test so cascading failures don't mask the root cause.

The fs fixture pattern: `mkdtemp` for the root; a per-test
`MINIMAL_ZARR_JSON` literal (sharded zstd, default chunk_key_encoding,
end index_location) at `<root>/<uri>/zarr.json`; synthetic shard files
at `<root>/<uri>/c/<a>/<b>/...` with hand-built footers (offset/nbytes
pairs + CRC32C); `rm -rf` via `system()` for cleanup.

See `tests/test_planner.c::fixture_init` for the canonical setup.

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
  multiple roots, etc.); revisit once the wave scheduler is in and a
  multi-backend story is needed.
- **No on-disk smoke test yet.** End-to-end is exercised via
  `bench/main.c --root <gen_dataset.py output> --peek` (manually
  validated against zarr-python). A self-contained test requires
  real zstd-compressed payloads in the test fixture (libzstd in the
  flake); deferred.
- **`DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES = 1 MB` cap.** Tight enough
  to fit nvCOMP's scratch on a 8GB-class laptop GPU; will bite as
  soon as someone configures larger inner chunks. Move it (and
  `DAMACY_MAX_CHUNKS_PER_BATCH`) onto `damacy_config` once we know
  the right knob shape.
