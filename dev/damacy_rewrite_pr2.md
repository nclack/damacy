# PR-2 fine plan: planner consumes prefetch handles + legacy delete

PR-1 left the planner using legacy `zarr_meta_cache_get` / `zarr_shard_cache_get`
while the prefetcher pinned the same data in `array_meta_cache` /
`shard_index_cache`. PR-2 collapses that duplication: planner reads directly
from prefetch handles, legacy caches are deleted, config + stats are
reshaped.

Branch off `damacy-rewrite` (PR #121 base). Targets `worktree-prefetch`
once merged via the PR stack, eventually `main` after PR #120 lands.

## Data flow today (post-PR-1)

```
push → lookahead → prefetcher worker → 3 prefetch caches (pinned by batch)
                                        ↓
                                    plan_reserve (pops ready slot)
                                        ↓
                                    batch_stage[] = (uri, aabb) only
                                        ↓
                                    planner_plan(samples)
                                        ↓
                              zarr_meta_cache_get(uri)      ← duplicate fetch
                              zarr_shard_cache_get(uri,…)   ← duplicate fetch
```

## Data flow target (post-PR-2)

```
push → lookahead → prefetcher worker → 3 prefetch caches (pinned by batch)
                                        ↓
                                    plan_reserve (pops ready slot)
                                        ↓
                            batch_stage[] = struct planner_sample
                                  { uri, aabb, h_meta, h_shards*, n_shards, h_layout }
                                        ↓
                                    planner_plan(planner_samples)
                                        ↓
                            prefetch_cache_try_get(amc, h_meta)
                            prefetch_cache_try_get(sic, h_shards[i])
                            (no copies; pin lifetimes managed by batch gate)
```

## Per-sample input type (new)

Internal-only — kept out of `damacy.h`. Goes in `src/planner/planner.h` next
to `planner_output`:

```c
struct planner_sample
{
  const char* uri;
  struct damacy_aabb aabb;
  struct prefetch_handle h_meta;
  struct prefetch_handle* h_shards;  // n_shards entries
  uint32_t n_shards;
  struct prefetch_handle h_layout;   // PR-2 ignores it (decoder territory)
};
```

`planner_plan` signature change:

```c
enum damacy_status planner_plan(struct planner* p,
                                const struct planner_sample* samples,
                                uint32_t n_samples,
                                uint16_t batch_pool_slot,
                                const int64_t* dst_strides,
                                uint8_t dst_full_rank,
                                struct planner_output* out);
```

`struct damacy_sample` (public API) is unchanged. The handle-carrying type
is internal because PR-3 will not need to expose it.

## planner_config field churn

Currently:
```c
struct planner_config {
  struct zarr_meta_cache* meta_cache;
  struct zarr_shard_cache* shard_cache;
  uint64_t page_alignment;
  // ...
};
```

After PR-2:
```c
struct planner_config {
  struct prefetch_cache* array_meta_cache;
  struct prefetch_cache* shard_index_cache;
  uint64_t page_alignment;
  // ...
};
```

The chunk_layout cache is not threaded through the planner — layout
consumption is the decoder's concern, and the prefetch handle for it is
already pinned via the batch gate.

## planner.c body changes

Per-sample loop in `planner_plan` (currently planner.c:385+):

```c
// PR-2: meta now read from the prefetch cache by handle.
const struct zarr_metadata* meta = prefetch_cache_try_get(
  self->cfg.array_meta_cache, sample->h_meta);
if (!meta) {
  // h_meta is in ERROR. Query for the err code and propagate.
  int err = 0;
  enum prefetch_state st = prefetch_cache_query(
    self->cfg.array_meta_cache, sample->h_meta, NULL, &err);
  (void)st;  // must be ERROR if try_get returned NULL on a non-NULL value type
  status = err ? (enum damacy_status)err : DAMACY_INVAL;
  goto Cleanup;
}
```

Per-shard loop (currently planner.c:499–540):

```c
while (sample_shard_iterator_next(&shard_it, shard_coord)) {
  struct prefetch_handle h = sample->h_shards[shard_idx_in_sample++];

  const struct shard_index_value* sv = prefetch_cache_try_get(
    self->cfg.shard_index_cache, h);

  if (!sv) {
    // ERROR slot — read the err code; treat NOTFOUND as missing shard
    // (fill), surface anything else.
    int err = 0;
    prefetch_cache_query(self->cfg.shard_index_cache, h, NULL, &err);
    if (err == DAMACY_NOTFOUND) {
      ctx.shard_missing = 1;
      ctx.shard_entries = NULL;
      ctx.n_shard_entries = 0;
      ctx.interned_path = NULL;
    } else {
      status = err ? (enum damacy_status)err : DAMACY_DECODE;
      goto Cleanup;
    }
  } else {
    ctx.shard_entries = sv->entries;
    ctx.n_shard_entries = (uint32_t)sv->n_entries;
    ctx.shard_missing = 0;
    // interned_path: build via zarr_shard_path_build like today; intern via
    // self->paths. No pin to release.
  }

  emit_shard(...);  // existing helper, unchanged
}
```

**`active_pin` is gone.** The planner no longer holds pins — pins on
shard_index / array_meta are owned by the prefetcher's batch entry and
released when `prefetcher_release_batch` runs at `damacy_release` time
(via the existing PR-1 watermark path).

**Note about chunk_layout (`h_layout`):** out of planner scope. The decoder
currently re-parses the blosc1 header from the chunk bytes; that stays
untouched. Threading `h_layout` to the decoder is a separate refactor.

## Prefetcher tolerance for missing shards

Today `advance_from_shard` (prefetcher.c) fails the whole slot if any
shard handle errors. After PR-2 the planner has to see per-shard error
state, so the prefetcher must let missing shards through:

```c
// In advance_from_shard, replace the unconditional bail-on-ERROR with:
for (uint32_t i = 0; i < s->n_shards; ++i) {
  int err = 0;
  enum prefetch_state st = prefetch_cache_query(
    self->shard_index_cache, s->h_shards[i], NULL, &err);
  if (st == PREFETCH_STATE_PENDING)
    return;
  if (st == PREFETCH_STATE_ERROR && err != DAMACY_NOTFOUND) {
    fail_slot(p, s, err);
    return;
  }
  // PENDING_OK or ERROR-NOTFOUND: continue. The slot reaches READY with
  // some h_shards entries in their respective cache's ERROR state; the
  // planner reads each via try_get and distinguishes fill vs real.
}
// All non-fatal: transition to chunk_layout stage.
s->h_layout = prefetch_cache_request(...);
s->state = PREFETCHER_PENDING_CHUNK_LAYOUT;
```

Adds one test in `tests/test_prefetcher.c`: a sample whose shard file is
absent reaches `PREFETCHER_READY` (not ERROR); the per-shard `h_shards[i]`
is queryable and returns ERROR/NOTFOUND.

## damacy_plan.c hand-off

`batch_stage` switches type:

```c
// damacy_internal.h
struct planner_sample* batch_stage;   // was: struct damacy_sample*
```

`plan_reserve` transfer loop (currently damacy_plan.c:90–94):

```c
for (uint32_t i = 0; i < n_samples; ++i) {
  self->batch_stage[i].uri = self->staging[i].uri;
  self->batch_stage[i].aabb = self->staging[i].aabb;
  self->batch_stage[i].h_meta = self->staging[i].h_meta;
  self->batch_stage[i].h_shards = self->staging[i].h_shards;
  self->batch_stage[i].n_shards = self->staging[i].n_shards;
  self->batch_stage[i].h_layout = self->staging[i].h_layout;
  // Steal: prefetcher_ready_free in plan_commit only frees the staging
  // entry's uri (NULL'd) and h_shards array allocation. The handles
  // themselves are bare ints (slot + generation); copying them is fine.
  self->staging[i].uri = NULL;
  self->staging[i].h_shards = NULL;
  self->staging[i].n_shards = 0;
}
```

Both `uri` and `h_shards` get NULL'd post-transfer; `prefetcher_ready_free`
must accept both being NULL (already does for uri; verify for h_shards).

`plan_commit` keeps the existing `prefetcher_ready_free(&self->staging[i])`
loop — it now only frees per-staging-entry allocations (none, since uri
and h_shards moved to batch_stage). The batch_stage entries' h_shards
arrays are freed when the batch is released — wire that in
`damacy_release` / `damacy_release_event` or at the end of `plan_run`.

**Open question:** lifetime of `batch_stage[i].h_shards` after plan_run.
The planner only reads it during plan; once the batch reaches BATCH_FILLING
the planner is done with it. Free at end of plan_run, or carry until
batch_release.

**Recommend:** free at end of plan_run — simpler lifetime, no risk of
holding the prefetcher's gate longer than needed.

## Plan_reserve NOTFOUND-suppression removal

PR-1's commit `88a38e1` added a `DAMACY_NOTFOUND` suppression in
plan_reserve so the legacy planner could route missing chunks through
its own fill path. PR-2 makes the prefetch-driven planner the authority,
so the suppression must go: a slot-level ERROR (which now only happens
for non-NOTFOUND errors, since the prefetcher tolerates missing shards)
should be surfaced.

Specifically, change:
```c
if (self->staging[i].state == PREFETCHER_ERROR &&
    self->staging[i].err_code != DAMACY_NOTFOUND) {
```
back to:
```c
if (self->staging[i].state == PREFETCHER_ERROR) {
```

## Function placement (no file moves this PR)

PR-2 is in-place: the file split from PR-1 stays as is. Edits land in:

- `src/planner/planner.h` — `planner_config` field swap, `planner_sample`
  struct, `planner_plan` signature.
- `src/planner/planner.c` — internal cache reads through `prefetch_cache_try_get`.
- `src/damacy_internal.h` — `batch_stage` type swap.
- `src/damacy_lifecycle.c` — drop `meta_cache` / `shard_cache` construction
  + destroy; wire `planner_config` with the prefetch caches; `batch_stage`
  calloc with the new element size.
- `src/damacy_plan.c` — handle transfer in plan_reserve, free in plan_run,
  NOTFOUND-suppression removal.
- `src/prefetch/prefetcher.c` — `advance_from_shard` NOTFOUND tolerance.
- `src/CMakeLists.txt` — drop `zarr_meta_cache` and `zarr_shard_cache` from
  the damacy LINKS list once the planner stops needing them.

Deleted:
- `src/zarr/zarr_meta_cache.h` + `.c`
- `src/zarr/zarr_shard_cache.h` + `.c`
- `src/zarr/zarr_meta_cache` CMake target
- `src/zarr/zarr_shard_cache` CMake target
- `tests/test_zarr_meta_cache.c`
- `tests/test_zarr_shard_cache.c`
- `tests/test_zarr_cache_threading.c` (it tests the legacy modules)

## damacy_stats reshape

Today (`damacy.h:317`):

```c
struct damacy_stats {
  // ...
  uint64_t zarr_meta_hits;
  uint64_t zarr_meta_misses;
  uint64_t shard_idx_hits;
  uint64_t shard_idx_misses;
  // ...
};
```

After PR-2:

```c
struct damacy_stats {
  // ...
  struct {
    uint64_t hits;
    uint64_t misses;
  } array_meta, shard_index, chunk_layout;
  // ...
};
```

This is a public-ABI break — Python bindings, bench JSON, and
`damacy_stats_get` all update in the same commit.

## damacy_config rename + new field

Today:
```c
uint32_t n_zarrs_meta_cache;
uint32_t n_shards_meta_cache;
```

After PR-2:
```c
uint32_t n_array_meta_cache;
uint32_t n_shard_index_cache;
uint32_t n_chunk_layout_cache;
```

Updates: `src/damacy.h`, `src/damacy_config.c` (validator + describe),
`src/damacy_lifecycle.c` (prefetch_cache capacities), Python binding
(`python/src/_damacy.pyx` or equivalent), `bench/main.c` if it sets
these knobs, `dev/devlog.md`? — no, the user owns that.

## Commit sequence

1. **planner: accept planner_sample input** — pure structural change.
   `planner_sample` type added, `planner_plan` signature updated, internals
   still call `zarr_meta_cache_get` / `zarr_shard_cache_get` using the
   `uri` field. `damacy_plan.c` builds `planner_sample[]` from staging
   (handles populated but ignored). Tests that call `planner_plan` directly
   (`test_planner`?) construct `planner_sample` arrays. Build + tests pass.

2. **prefetcher: tolerate missing shards** — `advance_from_shard` lets
   NOTFOUND-shard errors through to PREFETCHER_READY. New
   `test_prefetcher` case for the missing-shard path. Existing tests
   (test_damacy with `test_missing_shard_fills`) still pass via PR-1's
   NOTFOUND suppression in plan_reserve.

3. **planner: read array_meta from prefetch handle** — replace
   `zarr_meta_cache_get` with `prefetch_cache_try_get` + state query.
   `planner_config` gains `array_meta_cache`, keeps `meta_cache` dead-but-
   present (set to NULL by callers). Build + tests pass.

4. **damacy: drop zarr_meta_cache** — remove `meta_cache` field from
   `struct damacy`, lifecycle wiring, `planner_config.meta_cache` field.
   Delete `src/zarr/zarr_meta_cache.{h,c}`, drop CMake target and library
   dep. Delete `tests/test_zarr_meta_cache.c`. Build + tests pass.

5. **planner: read shard_index from prefetch handle** — replace
   `zarr_shard_cache_get` with per-shard `prefetch_cache_try_get` + state
   query (NOTFOUND → fill). Remove `active_pin` plumbing. Remove the
   PR-1 NOTFOUND suppression in `damacy_plan.c::plan_reserve`. Build +
   tests pass.

6. **damacy: drop zarr_shard_cache** — symmetric to step 4. Drop
   `shard_cache` field, lifecycle wiring, planner_config field, delete
   sources + tests + CMake target. Build + tests pass.

7. **damacy: config rename + chunk_layout knob** —
   `n_zarrs_meta_cache` → `n_array_meta_cache`,
   `n_shards_meta_cache` → `n_shard_index_cache`,
   add `n_chunk_layout_cache`. Python binding + bench update in same
   commit. C tests pass; Python tests: see below.

8. **damacy: stats shape from prefetch caches** — rename + restructure
   the cache-hit fields. Python binding + bench update. Single commit
   so the public ABI shifts atomically.

Each commit builds + tests cleanly. Steps 4 + 6 require deleting test
files; CMake test enumeration drops them automatically (`add_test`
calls).

## Test impact

| Test | PR-2 impact |
|------|-------------|
| `test_planner.c` | `planner_plan` signature change → fixtures construct `planner_sample` arrays instead of `damacy_sample`. Commit 1 churn. |
| `test_zarr_meta_cache.c` | Deleted in commit 4. |
| `test_zarr_shard_cache.c` | Deleted in commit 6. |
| `test_zarr_cache_threading.c` | Deleted in commit 6 (it tests legacy modules). |
| `test_damacy*.c` | No source change. Behavior identical (planner reads same data, different source). |
| `test_prefetcher.c` | One new test in commit 2 (missing-shard tolerance). |
| `python/tests/test_damacy.py` | The `stats` dict keys change in commit 8 (e.g. `zarr_meta_hits` → `array_meta_hits`). Fix in same commit. The 3 already-skip'd push-validation tests stay skipped — PR-3 territory. |
| `bench/main.c` | Config knob names update in commit 7; stats JSON keys update in commit 8. |

## Open questions / risks

1. **`active_pin` removal scope.** Today's planner threads
   `zarr_shard_pin* active_pin` through `chunk_range`, `emit_shard`, and
   the cleanup goto path. Removing it touches several call sites — keep
   them in commit 5 to avoid mixing concerns.

2. **`prefetch_cache_try_get` returning NULL on success.** PR-1's
   commit `88a38e1` made `chunk_layout_fetch` return success/NULL for
   non-blosc codecs. For the shard_index cache that ambiguity doesn't
   exist (NULL ⇒ ERROR), but a future codec change could re-introduce it.
   Keep the `prefetch_cache_query` re-query as the canonical state probe.

3. **`h_shards` lifetime.** The prefetcher_ready's `h_shards` array is
   `malloc`'d by `advance_from_meta` (prefetcher.c). PR-1's plan_commit
   `prefetcher_ready_free` frees it. PR-2 transfers the pointer to
   `batch_stage`, NULL'ing it in staging. The free moves to end-of-plan_run.
   Verify `prefetcher_ready_free` is safe with `h_shards == NULL`.

4. **`prefetch_handle` size.** It's currently `{slot, generation}` = 16
   bytes (?). With `batch_size` up to `DAMACY_MAX_BATCH_SIZE` and shards
   per sample varying, the worst-case `h_shards` arrays are bounded by
   the prefetcher's own caps. The `batch_stage` array doesn't bloat
   significantly.

5. **PR ordering vs the capacity-overflow subagent.** The subagent works
   on `worktree-prefetch` (PR #120). Its diff touches
   `src/prefetch/prefetcher.c`, same file PR-2 commit 2 modifies.
   Resolution path: rebase whichever lands later. The conflict surface
   is narrow (admit_locked vs advance_from_shard).

6. **Stats ABI break tolerance.** The Python binding exposes the stats
   dict by field name. Renaming `zarr_meta_hits` → `array_meta.hits`
   breaks any external script that reads it. There are no external
   consumers we know of; the bench tool is the only in-tree one.

7. **n_zarrs_meta_cache deprecation.** Could keep both names for one
   release with the old as a deprecated alias. Probably overkill —
   damacy isn't shipping yet and the config is internal-experimental.
   Drop cleanly in commit 7.

## Out of scope (PR-3)

- Python test rewrites for push-vs-pop error placement (the 3 already
  `pytest.skip`'d tests).
- Doc updates: `api.md` (new stats names + new config names),
  `distributed.md`, `troubleshooting.md`.
- Devlog entry — user-owned.
- Threading `h_layout` to the decoder so the decoder can skip the blosc1
  header re-parse. Separate refactor; layout still re-parsed at decode.

## What's not in this plan

- Wire-format changes to the prefetch layer beyond the NOTFOUND-shard
  tolerance. Anything else the prefetcher needs is a follow-up.
- Changes to the wave scheduler, batch_pool, or assemble. Untouched.
