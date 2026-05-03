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
- [ ] **4. Naive end-to-end (single wave per batch).** Pulls in CUDA +
  `io_queue` + the assemble kernel. Single wave slot, single batch
  slot, no overlap. First measurable build of the streaming pipeline.
  Reuses existing `src/decoder.{h,c}` for nvCOMP zstd decompress.
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

- **Logging + checks via `util/prelude.h`.** `.c` files include
  `util/prelude.h` for `CHECK(label, expr)` / `CHECK_SILENT(label, expr)` /
  `log_*` macros (`log_error` etc.). Errors funnel through a single
  `error:` label that frees partial state and returns the right
  `damacy_status`. `prelude.h` is private — never include from a header.
  See `src/util/lru.c::lru_create` for the canonical pattern.
- **Designated initializers.** `*p = (struct foo){ .a = ..., .b = ... }`
  instead of field-by-field assignment chains. Omitted fields are
  zero-initialized — don't spell out `= 0` for trailing fields.
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

Cache and planner integration tests build a tmpdir-rooted fs store:

- `mkdtemp` for the root.
- A small `MINIMAL_ZARR_JSON` literal (sharded zstd, default
  chunk_key_encoding, end index_location) at `<root>/<uri>/zarr.json`.
- Synthetic shard files at `<root>/<uri>/c/<a>/<b>/...` with
  hand-built footers (offset/nbytes pairs + CRC32C).
- `rm -rf` via `system()` for cleanup.

See `tests/test_planner.c::fixture_init` for the canonical setup.

## What I'd revisit

- **Cache pinning plumbing.** Step 5 (wave scheduler) likely surfaces
  the need for `zarr_*_cache_acquire/release` so scheduler-held shard
  indices can't be evicted underfoot.
- **`DAMACY_RANK` vs `DAMACY_INVAL`.** I split RANK out as its own
  status; if it never fires usefully, fold back into INVAL.
- **Zarr meta and shard caches share boilerplate** (str_dup, build path,
  cache wrapper). Worth extracting if a third cache appears (e.g., a
  resolver cache for multi-store URIs).
- **`src/util/lru.c` size**. ~500 LOC of carefully-factored code; could
  split helpers (list / freelist / index) into separate `.c` files if
  it grows again.
