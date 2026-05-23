# Plan: damacy.c rewrite for the prefetcher pipeline

## 1. File layout

Split the current ~1177-line monolith into 5 files. All share one private
header `src/damacy_internal.h` that defines `struct damacy` and the
`struct damacy_batch` handle (today, those are file-static in damacy.c;
once split, multiple TUs need the layout).

- `src/damacy_internal.h` (new)
  - `struct damacy` (today: damacy.c:43-87)
  - `struct damacy_batch` (today: damacy.c:36-41)
  - `struct ctx_guard` + prototypes (today: damacy.c:96-122)
  - `#define DAMACY_POP_POLL_NS 10000` (today: damacy.c:92)
  - extern prototypes for the helpers each .c needs

- `src/damacy_lifecycle.c` (new — ~350 lines)
  - `damacy_create`, `damacy_destroy`, `damacy_get_device`, `damacy_config_describe`
  - `destroy_inner`, `ctx_guard_enter/exit`, `batch_pool_allocate`
  - All the prefetch-cache construction + `prefetcher_create/start/stop/destroy`

- `src/damacy_push.c` (new — ~80 lines)
  - `damacy_push`
  - `next_batch_id_` cursor used by push (assigns `batch_id` per-sample via
    `lookahead_push_with_batch`); see §3

- `src/damacy_plan.c` (new — ~250 lines)
  - `plan_reserve` / `plan_run` / `plan_commit`
  - The new `assemble_batch_from_prefetch` (replaces `push_one` + the
    lookahead_drain → batch_stage path); see §4
  - `find_filling_slot_with_work`/`find_free_batch_slot` callsites stay
    together here

- `src/damacy_pop.c` (new — ~250 lines)
  - `damacy_pop`, `damacy_release`, `damacy_release_event`, `damacy_flush`
  - `damacy_batch_info`, `damacy_stats_get`, `damacy_stats_reset`
  - `damacy_set_gpu_bytes_committed_for_test`

- `src/damacy_scheduler.c` (new — ~80 lines)
  - `damacy_scheduler_step` + `kick_peel_into_free_slots`
  - Owns the "ingest prefetch_ready → plan → peel" orchestration

`src/CMakeLists.txt` swap: today `add_src_lib(damacy SOURCES damacy.c …)`
at one entry — change to
`SOURCES damacy_lifecycle.c damacy_push.c damacy_plan.c damacy_pop.c damacy_scheduler.c damacy_internal.h`.
Drop `zarr_meta_cache` and `zarr_shard_cache` from the link list (see
CMakeLists.txt:209, :239) and add `prefetch` + `prefetcher`.

---

## 2. damacy_create / damacy_destroy

### Keep (no change):
- Validation block: `validate_config(cfg)` (damacy.c:463)
- CUDA context capture / `cuDevicePrimaryCtxRetain` block (damacy.c:478-516)
- `numa_init` (damacy.c:525-535)
- `gpu_budget_new`, `pool_reserve` carve-out (damacy.c:537-563)
- `wave_pool_resolve_sizing` + predicted-budget commit (damacy.c:566-609)
- `resolve_enable_gds` + `store_host`/`store_gds` construction (damacy.c:611-636)
- `batch_slot_init`, `wave_pool_init` under `numa_scope_enter/exit` (damacy.c:654-679)
- `lookahead_init` (damacy.c:683-685)
- `batch_samples` / `batch_stage` alloc — but `batch_stage` repurposes (see §4):
  rename to `batch_samples_pinned` (a `damacy_sample[]` of consumed
  prefetcher_ready samples kept alive across plan_run); `damacy_sample_slot`
  array goes away entirely
- `self->handle.d = self`
- `scheduler_create` last (damacy.c:697)
- The teardown `destroy_inner` body's ordering (scheduler → wave_pool →
  batch_stage → lookahead → batch_pool → planner → stores → budget)

### Replace:
- damacy.c:638-643 — `zarr_meta_cache_create` + `zarr_shard_cache_create` blocks.
  Replace with three `prefetch_cache_create` calls:
  - `self->amc = prefetch_cache_create(&{ .capacity=cfg->tuning.n_array_meta_cache, .ops=&array_meta_ops, .fetcher=&self->amf.base, .executor=&self->io_exec });`
  - `self->sic` with `shard_index_ops` + `shard_index_fetcher_init(&self->sif, self->store_host, self->amc)`
  - `self->clc` with `chunk_layout_ops` + `chunk_layout_fetcher_init(&self->clf, self->store_host, self->amc, self->sic)`
  - Where do the fetchers live? Embed by value in `struct damacy`:
    `struct array_meta_fetcher amf; struct shard_index_fetcher sif; struct chunk_layout_fetcher clf;`
    and one `struct prefetch_executor io_exec` shim wrapping `io_queue_post`
    on `self->store_host`'s queue (or expose it from store_fs; see Open Question §11).
- damacy.c:645-652 — `planner_config` no longer takes `meta_cache`/`shard_cache`.
  New shape (see §3): `planner_config { .page_alignment, .max_chunk_uncompressed_bytes, .read_op_max_bytes };`
  Plus planner_plan signature change.
- Add
  `self->pf = prefetcher_create(&{ .lookahead=&self->lookahead, .array_meta_cache=self->amc, .shard_index_cache=self->sic, .chunk_layout_cache=self->clc, .capacity = cfg->lookahead_batches * cfg->batch_size, .batch_capacity = cfg->lookahead_batches + DAMACY_N_BATCH_SLOTS });`
  and `prefetcher_start(self->pf)` *before* the scheduler starts, so the
  prefetcher is already draining the lookahead when the scheduler ticks.

### destroy_inner ordering:
- Step the prefetcher off first (or rely on `prefetcher_destroy` calling
  `prefetcher_stop` internally — see prefetcher.c:293-308). Order:
  `scheduler_destroy` → `prefetcher_destroy(self->pf)` (this signals
  lookahead stop) → existing wave_pool_destroy and below →
  `prefetch_cache_destroy` for clc, sic, amc (reverse dependency order:
  clc depends on sic and amc; sic depends on amc).

### `damacy_config_describe`:
No change needed — it doesn't touch caches.

---

## 3. damacy_push (NEW SHAPE)

**Behavioral change:** push no longer touches the meta cache. Validations
that survive at push:
- `samples.beg > samples.end` → DAMACY_INVAL
- `!self` → DAMACY_INVAL
- `self->failed_status != DAMACY_OK` → DAMACY_SHUTDOWN
- per-sample: `sample->uri != NULL` → DAMACY_INVAL on null
- per-sample: `sample->aabb.rank in [1, DAMACY_MAX_RANK]` → DAMACY_RANK
- per-sample: `sample->aabb.rank == self->cfg.sample_rank` → DAMACY_RANK
  (purely from cfg, no zarr_metadata needed)
- per-sample: `sample_aabb_extents_match_cfg` → DAMACY_INVAL (purely from cfg)
- lookahead full → DAMACY_AGAIN

**Validations that move to pop:** any error that today comes from
`zarr_meta_cache_get` (DAMACY_NOTFOUND, DAMACY_DECODE, DAMACY_IO) or
`cast_path_supported` against the *resolved* zarr meta (DAMACY_DTYPE) or
`meta.rank != sample.rank` (DAMACY_RANK from the *zarr*, not the cfg).

**New body shape:**
```c
struct damacy_push_result damacy_push(struct damacy* self, struct damacy_sample_slice samples) {
  // Pre-flight cfg-only checks per sample, then:
  // batch_id = ceil_div(next_pushed_count, batch_size);
  // lookahead_push_with_batch(&self->lookahead, sample, batch_id);
  // No scheduler_lock — lookahead has its own mutex.
}
```

Drop `push_one` (damacy.c:171-197) entirely. The `next_pushed_count`
cursor + the batch_id math live in `struct damacy`
(e.g. `uint64_t pushed_samples; uint64_t pushed_batch_id;`), guarded by a
new fine-grained `push_lock` (separate from scheduler_lock to avoid the
mid-flight scheduler stall today's push_one causes).

---

## 4. plan_reserve / plan_run / plan_commit

The current flow (damacy.c:204-330) is:
1. `lookahead_drain` → `batch_samples` (uri-copy)
2. copy to `batch_stage` (uri pointer + aabb)
3. `planner_plan(planner, batch_stage, n, slot_idx, …)` — synchronously
   calls `zarr_meta_cache_get` and `zarr_shard_cache_get` inside.

**New flow:**

The lookahead is *not* drained by the worker anymore — the prefetcher
owns it (prefetcher.c:244-249). The worker instead pops *ready*
prefetcher slots.

- **plan_reserve (under scheduler_lock):**
  - Requires at least `batch_size` prefetcher_ready slots for the head
    batch_id. Use a pre-collection buffer `self->staging[batch_size]` of
    `struct prefetcher_ready`.
  - `prefetcher_pop_ready(self->pf, &staging[i])` until `n == batch_size`.
    If any `staging[i].state == PREFETCHER_ERROR`, latch
    `self->failed_status = error_from_ready(&staging[i])` (need a small
    mapping helper — the prefetcher_ready doesn't currently carry the
    damacy_status; see Risk §11) and bail.
  - For each ready, copy `uri` + `aabb` into `self->batch_samples_pinned[i]`
    so planner_plan can read them; transfer ownership of `staging[i].uri`
    to the batch slot (since planner_plan needs valid strings until
    plan_commit). Store the staging handles in the slot (slot owns
    `h_meta`, `h_shards[]`, `h_layout` until release — see §6).
  - `batch_pool_allocate` as today.
  - `slot->state = BATCH_PLANNING`.

- **plan_run (off-lock):**
  - Calls `planner_plan(planner, batch_samples_pinned, n, slot_idx, strides, rank, &plan_out)`.
    Planner internally dereferences the handles via
    `prefetch_cache_try_get` instead of `zarr_meta_cache_get` /
    `zarr_shard_cache_get`. (Planner refactor is in scope of this PR;
    see §10.)
  - Same `cuMemcpyHtoD` of sample_plans as today.

- **plan_commit (under scheduler_lock):**
  - Same as today except:
    - `sample_slot_clear(&self->batch_samples[i])` becomes
      `prefetcher_ready_free(&staging[i])` (release the uri + h_shards
      alloc; the cache *entries* stay pinned until watermark advance —
      see §5).
    - Same `chunks==0` degenerate-batch path.
    - Same `chunks_planned`, `chunks_to_load`, `reads_issued` accounting.

The planner change (planner.c:381-595) is a separate concern, but the
damacy facade must pass the per-batch `gate` to planner_plan (or skip
the gate check if `prefetcher_pop_ready` already proves ready state —
the prefetcher only returns slots in READY/ERROR). Simpler: skip the
gate, rely on prefetcher_pop_ready as the readiness signal. The gate is
then an internal prefetcher detail.

---

## 5. damacy_scheduler_step

**Current** (damacy.c:389-412):
1. Push worker ctx lazily
2. `wave_pool_advance`
3. `kick_peel_into_free_slots` which interleaves `plan_reserve/run/commit`
   with peel.

**New:**
1. Push worker ctx lazily (unchanged).
2. `wave_pool_advance` (unchanged).
3. `kick_peel_into_free_slots` — same structure but the "plan a new batch"
   branch (damacy.c:343-358) replaces `self->lookahead.size < self->cfg.batch_size`
   gate with "do we have batch_size ready prefetcher slots?". Use a
   non-blocking check that scans the prefetcher's internal slot table —
   exposed via a new helper `prefetcher_ready_count(self->pf)` (add to
   prefetcher.h) OR pre-pop into `self->staging[]` greedily during the
   gate check.
4. **Watermark advance:** after a successful `plan_commit` for batch_id B,
   call `prefetcher_advance_watermark(self->pf, B + 1)`. This is the
   project's chosen safer "advance on plan success" path (per the
   metadata_prefetch.md "Watermark advance point" open knob §). The
   dev/metadata_prefetch.md plan §8 says scheduler does this.
5. **Batch gate release:** after watermark advance, also call
   `prefetcher_release_batch(self->pf, B)` to free the per-batch gate
   entry — but only if the planner gate-poll path is wired (skip if we
   elide the gate check).

`damacy_scheduler_step` no longer needs to do anything for the
prefetcher worker — the prefetcher has its own thread
(prefetcher.c:235-253). The scheduler is purely the wave + plan
orchestrator.

---

## 6. damacy_pop / damacy_release / damacy_release_event / damacy_flush

### damacy_pop:
**Idle-AGAIN predicate** (damacy.c:822-825) needs an extra term. Today:
```
!any_wave_in_flight && !any_slot_in_flight && !any_batch_in_flight && lookahead.size < batch_size
```
New:
```
!any_wave_in_flight && !any_slot_in_flight && !any_batch_in_flight
  && lookahead_size(&self->lookahead) == 0
  && prefetcher_in_flight_count(self->pf) == 0  // new helper
  && prefetcher_ready_count(self->pf) < self->cfg.batch_size
```
`prefetcher_in_flight_count` can read `prefetcher_stats.in_flight`
(already in prefetcher_stats_get, prefetcher.h:40). Add a non-locking
convenience accessor to avoid two locks per pop.

**Error surfacing:** Since push no longer surfaces NOTFOUND/DTYPE/RANK,
those statuses now reach the user at `damacy_pop`. Today's pop just
returns `self->failed_status`; the scheduler latches it from
`kick_peel_into_free_slots` errors. Add a path where `plan_reserve`
translates `staging[i].state == PREFETCHER_ERROR` into
`self->failed_status` *with the appropriate damacy_status* — which
requires the prefetcher_ready to carry an error code (see Risk §11).

### damacy_release / damacy_release_event:
**No change.** They touch only batch-pool slot state. Note: `slot->n_chunks`
and friends are zeroed on release; that's enough to invalidate the
staging pointers held by the slot. Just make sure the slot stores any
`h_meta`/`h_shards`/`h_layout` it captured in plan_reserve, and either:
- (a) the watermark-advance path proves to evict them eventually
  (current plan), OR
- (b) release explicitly drops references (would require a per-handle
  release API on prefetch_cache, which the doc explicitly avoids — *do
  not add this*).

The slot doesn't need to call anything per-handle; watermark advance
covers it.

### damacy_flush:
The synchronous-tail path (damacy.c:966-994) currently calls
`plan_reserve` with the *current* lookahead size when it's < batch_size.
New shape:
1. Signal "no more pushes" — but `lookahead_signal_stop` is for prefetcher
   shutdown, not flush. Keep the lookahead live; just don't push.
2. **Wait** for the prefetcher to fully resolve every queued sample:
   poll `lookahead_size == 0 && prefetcher_in_flight_count == 0 && prefetcher_ready_count > 0`.
   Use `SCHEDULER_WAIT_DIAG` for the wait loop.
3. Once all are ready, drain `prefetcher_pop_ready` for whatever count
   remains (1..batch_size-1 — the truncated tail), feed plan_reserve with
   the truncated count, plan_run/commit as today.
4. The existing in-flight-wait loop (damacy.c:1003-1010) stays.

A new wrinkle: the prefetcher can return PREFETCHER_ERROR for a tail
sample. In that case flush returns the latched status, same as pop.

---

## 7. Test migration

### Tests that must change — `push` no longer validates URI / dtype / rank-vs-zarr:

- `tests/test_damacy.c` — no `DAMACY_NOTFOUND` expectations from push today;
  OK as-is *except*:
  - `test_lookahead_backpressure` (damacy.c:460-490) is fine — DAMACY_AGAIN
    still flows.
  - `test_sample_shape_mismatch_rejected` in `test_damacy_caps.c:332-360`
    is fine — that's a cfg-only check that stays at push.
- `tests/test_damacy_caps.c` — `test_resolver_minimum_one_chunk` is fine;
  cfg-only.
- `python/tests/test_damacy.py`:
  - `test_unknown_uri_raises_notfound` (line 226) — currently asserts
    `d.push(...)` raises `NotFound`. Now NotFound surfaces from `pop()`
    instead. Rewrite: push a bad URI, then expect `NotFound` from the
    first `pop()`.
  - `test_unsupported_src_dtype_raises_dtype_mismatch` (line 234) — same
    shape, surface DtypeMismatch at pop.
  - `test_push_error_drops_offending_iterator` (line 345) — this one
    assumes push synchronously raises on `bad`, then a prior `good`
    survives. New behavior: push won't reject `bad`; pop will. Rewrite:
    push `[good]`, then push `[bad, good]`, pop yields `good` first
    batch, then pop raises NotFound. The "trailing good is dropped"
    semantics need to be reconsidered — under the new shape both `good`
    samples may have completed prefetch, so pop yields three good
    batches and one error. **This is a real behavioral change to call
    out in the PR description.**
- `python/tests/test_native_internals.py:60` — single `DamacyError` check;
  safe but verify the status code path still surfaces correctly.

### Tests unaffected:
- `test_array_meta.c`, `test_shard_index.c`, `test_chunk_layout_cache.c`,
  `test_prefetch_cache.c`, `test_prefetcher.c` — already test the new
  layer; no change.
- `test_planner.c` — needs *the planner refactor's* test updates; out of
  scope for this damacy.c rewrite if we split the planner-side change
  into a separate PR (see §10).

### Tests to delete (if the planner migration is in this PR):
- `tests/test_zarr_meta_cache.c`, `tests/test_zarr_shard_cache.c`,
  `tests/test_zarr_cache_threading.c` — the modules they exercise are
  being deleted (per dev/metadata_prefetch.md §"Removed").

### CMakeLists.txt (tests/):
Drop the three deleted test targets; drop link deps on `zarr_meta_cache`
/ `zarr_shard_cache` from any remaining target.

---

## 8. Python binding

`python/damacy/_api.c`:
- Line 702: `Py_BuildValue` keyword `"n_zarrs_meta_cache"` and line 779
  `.n_zarrs_meta_cache = n_zarrs_meta` need rename (e.g.
  `n_array_meta_cache`) if we're touching the C config struct in this PR
  (see §10 — recommend yes).
- Line 921: `damacy_push` call — no signature change; just be aware the
  error-status mix returned has narrowed (push won't return
  NOTFOUND/DTYPE/RANK anymore). The `raise_status` path still works for
  the few remaining (INVAL, RANK from cfg, SHUTDOWN, AGAIN); just update
  the comment at line 928 ("any other status NOTFOUND/DTYPE/RANK…raises")
  to reflect the new reality.
- Line 1025-1027: `st.zarr_meta_hits`, `st.shard_idx_hits` — these stats
  fields go away or get renamed. Suggest a new `damacy_stats` shape with
  three `prefetch_cache_stats` blocks (amc, sic, clc), each with its own
  hits/misses/size/capacity. The Python dict keys become e.g.
  `array_meta_hits`, `shard_index_hits`, `chunk_layout_hits`.

`python/damacy/__init__.py`:
- Line 453-454, 496-497, 518-519, 573-574, 985-986: `n_zarrs_meta_cache` /
  `n_shards_meta_cache` config field → `n_array_meta_cache` /
  `n_shard_index_cache` (+ new `n_chunk_layout_cache`).
- Line 646-676: `zarr_meta_hits` / `shard_idx_hits` stats fields → mirror
  the C struct change.
- Docstring updates in `Config` (line 440-490): clarify push no longer
  validates URIs; errors surface at pop.

`python/damacy/_native.pyi`:
- Line 132: same rename.

---

## 9. Docs

- `docs/distributed.md:122` — update the LRU-cap table row from
  `n_zarrs_meta_cache, n_shards_meta_cache` to the new three-cache names.
- `docs/api.md` — review the `Pipeline.push` error documentation (push no
  longer raises NotFound/DtypeMismatch; those move to pop). The "Status
  codes" section needs the OK/AGAIN/INVAL/RANK/SHUTDOWN list for push
  and the larger set for pop documented separately.
- `docs/index.md` — no change expected (skim to confirm).
- `docs/prefetch.md` — about the *consumer-side* async pattern, not the
  metadata prefetcher; unchanged.
- `docs/budget.md` — unchanged.
- `docs/troubleshooting.md` — search for "NotFound" / "DtypeMismatch"
  advice that may reference push; rewrite to reference pop.
- `dev/metadata_prefetch.md` — this is the design doc; mark §"Plan" as
  landed.
- `README.md` — verify no example shows push raising NotFound.

`bench/`:
- `bench/main.c:118-119, 295-296, 391-394, 412-414, 832-833` — rename +
  add a third knob.
- `bench/scenario.py:85-86` — same.
- `bench/scenarios/*.json` — keep old keys readable via aliasing in
  `bench/main.c` for one release, then deprecate (or hard-rename and
  edit all six JSONs in one go). Smaller PR + grace period = aliasing.

---

## 10. Suggested PR sequence

Three PRs:

**PR-1 (this rewrite, structural):** the damacy.c → 5-file split, the
push behavior change, the prefetcher wiring inside `damacy_create`, the
new plan flow consuming `prefetcher_pop_ready`. **Keeps**
`zarr_meta_cache` + `zarr_shard_cache` around as transitive deps used
inside the planner — the planner still does sync cache_get. damacy.c
constructs both the legacy caches *and* the new prefetch_caches, feeding
the planner the legacy ones and the prefetcher the new ones. The
prefetcher's caches become a "shadow" that holds metadata pinned ahead
of the planner; the planner re-fetches synchronously but every hit
short-circuits in microseconds because the file is already in the OS
page cache.
  - **Why this works:** removes the synchronous push-time validation
    cleanly. The planner's slowness becomes asymptotically zero because
    the prefetcher walks the same files first.
  - **Drawback:** double-bookkeeping during the transition.

**PR-2 (planner migration):** Rewire `planner.c` to take
`prefetch_handle h_meta` + `prefetch_handle h_shards[]` +
`prefetch_handle h_layout` per-sample (carried in a new field on
`damacy_sample` for the internal path, or in a side struct passed
alongside). Drop `zarr_meta_cache_get` / `zarr_shard_cache_get` calls.
Delete `src/zarr/zarr_meta_cache.{h,c}` + `src/zarr/zarr_shard_cache.{h,c}`.
Rename `n_zarrs_meta_cache` → `n_array_meta_cache` (config + python +
bench + docs in one shot). Add `n_chunk_layout_cache` knob.

**PR-3 (Python error semantics + tests):** rewrite the three Python tests
in test_damacy.py that depend on push-time validation. Update docs to
clarify the new push contract. Add a "push error surfaces at pop"
example in `docs/api.md`.

Keeping PR-1 minimal (no planner change) is the bigger win — it's the
riskier orchestration change, and isolating the planner refactor makes
both reviewable.

---

## 11. Risks and open questions

1. **Prefetcher_ready doesn't carry a damacy_status on error.**
   prefetcher.h:52-62 has only `state` (READY or ERROR) and the handles.
   The actual error code lives in the per-handle `out_err` field set by
   the fetcher, retrievable via `prefetch_cache_query`. Two options:
   - (a) `damacy_plan.c::plan_reserve` re-queries each handle on ERROR to
     derive the status. Doable but ugly.
   - (b) Extend `struct prefetcher_ready` with `int err_code;` populated
     by the prefetcher when it transitions slot → ERROR. Cleaner;
     one-line addition. **Recommend (b); flag for prefetcher author.**

2. **Per-batch sample push for `batch_id`.** Current `lookahead_push`
   calls `lookahead_push_with_batch(la, sample, 0)` (lookahead.c:95). For
   prefetcher dedup + watermark to work, `damacy_push` must compute
   `batch_id = floor(pushed_samples / batch_size)` and push with that
   ordinal. Need to also handle the truncated-flush case: which
   `batch_id` does a partial tail get? Same `floor` math; the
   prefetcher's batch-gate entry will have <batch_size samples but
   that's fine.

3. **prefetcher capacity sizing.** `capacity = lookahead_batches * batch_size`
   is the in-flight ceiling. `batch_capacity` controls the gate table;
   setting it to `lookahead_batches + N_BATCH_SLOTS` allows up to that
   many in-flight batch_ids before `prefetcher_advance_watermark`
   reclaims. If user lookahead_batches is very small, this can saturate
   — verify with `test_lookahead_backpressure`.

4. **Executor wiring.** prefetch_cache_config takes a
   `struct prefetch_executor*`. Today no shim wraps `io_queue_post` for
   damacy. Need a small adapter in damacy_lifecycle.c (or factor it to
   `src/prefetch/io_queue_executor.{h,c}`) that holds a `struct io_queue*`
   pointer and forwards `post` to `io_queue_post`. The io_queue lives
   inside `store_fs`; either expose `store_fs_get_io_queue()` or have
   store_fs own the executor adapter too.

5. **`damacy_release_event`'s ctx_guard.** Push the retained primary in
   `damacy_release_event` for `cuStreamWaitEvent` (damacy.c:898-900).
   This still applies post-refactor; just verify after the file split.

6. **`damacy_flush` semantics with prefetcher in flight.** If the user
   calls `damacy_flush()` immediately after push without ever calling
   pop, the prefetcher may still be parking samples on the second or
   third stage. flush has to wait for the prefetcher to finish resolving
   the *current* lookahead before it can plan the tail. Use
   `prefetcher_drain(self->pf)` (prefetcher.h:49) — already provided
   exactly for this case. **Verify** that prefetcher_drain returns once
   *every queued sample* is in READY or ERROR.

7. **`damacy_destroy` cancellation.** prefetcher_destroy → prefetcher_stop
   signals lookahead. But the worker may be mid-`advance_all` holding
   the prefetcher lock when shutdown begins. Look at prefetcher.c:248-253
   — the worker sleeps for 1 ms between advances, so the shutdown is
   bounded. The destroy order is fine.

8. **Stats struct compat.** `damacy_stats` (damacy.h:312-351) has
   `zarr_meta_hits`, `shard_idx_hits` as uint64. If we keep these field
   names but populate them from the new caches (amc + sic respectively),
   there's no ABI break. Easier than renaming. Drop
   `n_zarrs_meta_cache`/`n_shards_meta_cache` config fields though —
   those have to rename because the *count* of caches changed (now 3,
   not 2).

9. **`n_zarrs_meta_cache` field deletion timing.** PR-1 keeps the field
   (legacy cache still uses it). PR-2 deletes it. Or merge with new
   fields, breaking ABI once. Pick one — recommend a hard break in PR-2
   since this is pre-1.0.

10. **Worker context interaction.** The new prefetcher thread doesn't
    need a CUcontext — it's pure CPU + io_queue. Skip ctx_guard in
    prefetcher worker_fn. Confirmed by reading prefetcher.c:235-253 — no
    CUDA calls.

11. **Open: does `prefetcher_pop_ready` need ordering by batch_id?**
    Today it returns *any* terminal slot (prefetcher.c:373-389 walks the
    slot array linearly). The planner builds one batch from `batch_size`
    samples; if those `batch_size` ready slots span batch_ids 0 *and* 1,
    you've mixed batches. Need either: (a)
    `prefetcher_pop_ready_for_batch(p, batch_id, &out)`, or (b) per-slot
    ordering by batch_id, or (c) verify that prefetcher slots assigned
    via `lookahead_pop_blocking` are FIFO. Looking at
    prefetcher.c:241-249, `lookahead_try_pop` is FIFO and
    `find_free_slot_locked` walks linearly, so slot index ≠ batch order.
    **This is the highest-risk open question** — file a clarification
    ticket / read prefetcher tests for ordering guarantees before PR-1
    lands.

12. **Doctests.** `python/tests/test_doctests.py` runs doctests embedded
    in `__init__.py` — review the docstrings that mention push
    validation, esp. the error hierarchy comment at `_api.c:1106`.

---

### Critical Files for Implementation
- src/damacy.c
- src/prefetch/prefetcher.h
- src/planner/planner.c
- python/damacy/__init__.py
- src/CMakeLists.txt
