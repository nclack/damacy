# PR-1 fine plan: damacy.c file split + prefetcher wiring

Supersedes the high-level layout in `damacy_rewrite_plan.md §1` with
line-level mappings. PR-1 keeps legacy `zarr_meta_cache` +
`zarr_shard_cache` alive for the planner; PR-2 deletes them and rewires
planner to consume prefetch handles.

## Function placement (current damacy.c → target file)

| Lines | Symbol | Target file | Visibility |
|------:|--------|-------------|------------|
| 36–41 | `struct damacy_batch` | `damacy_internal.h` | private |
| 43–87 | `struct damacy` | `damacy_internal.h` | private |
| 92 | `#define DAMACY_POP_POLL_NS` | `damacy_internal.h` | private |
| 96–99 | `struct ctx_guard` | `damacy_internal.h` | private |
| 101–113 | `ctx_guard_enter` | `damacy_lifecycle.c` | proto in internal |
| 115–122 | `ctx_guard_exit` | `damacy_lifecycle.c` | proto in internal |
| 127–155 | `batch_pool_allocate` | `damacy_plan.c` | proto in internal |
| 157–167 | `sample_aabb_extents_match_cfg` | `damacy_push.c` | static |
| 171–197 | `push_one` | DELETED — body inlined into `damacy_push` | — |
| 204–229 | `plan_reserve` | `damacy_plan.c` | proto in internal |
| 234–277 | `plan_run` | `damacy_plan.c` | proto in internal |
| 280–330 | `plan_commit` | `damacy_plan.c` | proto in internal |
| 336–383 | `kick_peel_into_free_slots` | `damacy_scheduler.c` | static |
| 389–412 | `damacy_scheduler_step` | `damacy_scheduler.c` | proto in internal |
| 420–451 | `destroy_inner` | `damacy_lifecycle.c` | static |
| 453–733 | `damacy_create` | `damacy_lifecycle.c` | extern |
| 735–757 | `damacy_destroy` | `damacy_lifecycle.c` | extern |
| 759–763 | `damacy_get_device` | `damacy_lifecycle.c` | extern |
| 767–803 | `damacy_push` | `damacy_push.c` | extern |
| 807–855 | `damacy_pop` | `damacy_pop.c` | extern |
| 857–885 | `damacy_release` | `damacy_pop.c` | extern |
| 887–954 | `damacy_release_event` | `damacy_pop.c` | extern |
| 958–1029 | `damacy_flush` | `damacy_pop.c` | extern |
| 1033–1054 | `damacy_batch_info` | `damacy_pop.c` | extern |
| 1056–1086 | `damacy_stats_get` | `damacy_pop.c` | extern |
| 1088–1094 | `damacy_stats_reset` | `damacy_pop.c` | extern |
| 1101–1107 | `damacy_set_gpu_bytes_committed_for_test` | `damacy_lifecycle.c` | extern (test hook) |
| 1109–1193 | `damacy_config_describe` | `damacy_lifecycle.c` | extern |

Total: 5 new .c files + 1 new internal .h, replacing 1 monolithic .c.

## `damacy_internal.h` content

```c
#pragma once
#include "damacy.h"
#include "damacy_config.h"
#include "damacy_stats.h"
#include "batch_pool/batch_pool.h"
#include "gpu_budget/gpu_budget.h"
#include "lookahead/lookahead.h"
#include "numa/numa.h"
#include "planner/planner.h"
#include "prefetch/array_meta.h"
#include "prefetch/chunk_layout.h"
#include "prefetch/prefetch_cache.h"
#include "prefetch/prefetcher.h"
#include "prefetch/shard_index.h"
#include "scheduler/scheduler.h"
#include "store/store.h"
#include "wave/wave_pool.h"
#include "zarr/zarr_meta_cache.h"     // PR-2 removes
#include "zarr/zarr_shard_cache.h"    // PR-2 removes
#include <cuda.h>

struct damacy_batch { ... };  // unchanged from current
struct damacy {
  /* unchanged head fields */
  struct damacy_config cfg;
  enum damacy_status failed_status;
  uint64_t next_batch_id;          // existing — for batch slots
  uint64_t pushed_samples;         // NEW — push-side cursor (see push reshape)
  uint64_t pushed_batch_id;        // NEW — push-side current batch_id
  /* ...all existing CUDA / NUMA / budget / store fields... */

  /* legacy caches retained for planner in PR-1 (deleted in PR-2) */
  struct zarr_meta_cache* meta_cache;
  struct zarr_shard_cache* shard_cache;

  /* NEW prefetch caches + fetchers (3 of each) */
  struct array_meta_fetcher amf;
  struct shard_index_fetcher sif;
  struct chunk_layout_fetcher clf;
  struct prefetch_cache* amc;
  struct prefetch_cache* sic;
  struct prefetch_cache* clc;
  struct prefetch_executor io_exec;  // shim wrapping io_queue_post
  struct prefetcher* pf;

  /* unchanged tail */
  struct planner* planner;
  struct damacy_lookahead lookahead;
  /* batch_samples / batch_stage: drop batch_samples (was a
     damacy_sample_slot[]); rename batch_stage to batch_samples_pinned
     (a damacy_sample[] of consumed prefetcher_ready samples held
     across plan_run). Also keep struct prefetcher_ready
     staging[batch_size] for plan_reserve's pop scratchpad. */
  struct damacy_sample* batch_samples_pinned;
  struct prefetcher_ready* staging;
  /* ...rest unchanged... */
};

#define DAMACY_POP_POLL_NS 10000
struct ctx_guard { int active; };

/* lifecycle-private */
enum damacy_status ctx_guard_enter(struct damacy*, struct ctx_guard*);
void ctx_guard_exit(struct ctx_guard*);

/* plan-private */
enum damacy_status batch_pool_allocate(struct damacy*);
enum damacy_status plan_reserve(struct damacy*, uint16_t slot, uint32_t n);
enum damacy_status plan_run(struct damacy*, uint16_t slot, float* out_ms);
enum damacy_status plan_commit(struct damacy*, uint16_t slot,
                               enum damacy_status, float, int* changed);

/* scheduler entry — passed to scheduler_create by lifecycle */
int damacy_scheduler_step(void* arg);
```

## New helper: dedicated io_queue + executor adapter

The store vtable does not expose its io_queue, and `struct store_fs` is
private to `store/store_fs.h`. Rather than punch a hole in the store
API, give the prefetcher its own dedicated `io_queue` for metadata work.
Sized small (2 threads) since metadata reads are kilobytes; the bulk
chunk I/O stays on the store's own pool, isolated from metadata
head-of-line blocking.

`struct damacy` gains:
- `struct io_queue* prefetch_io_q;` — owned by damacy
- `struct prefetch_executor io_exec;` — vtable instance

Construction in damacy_create (after `numa_init`, before
`prefetcher_create`):
```c
self->prefetch_io_q = io_queue_create(2, &self->numa);
CHECK(Fail, self->prefetch_io_q);
self->io_exec.post = damacy_io_exec_post;
```

Adapter (in damacy_lifecycle.c):
```c
static int damacy_io_exec_post(struct prefetch_executor* e, void (*fn)(void*),
                               void* ctx, void (*ctx_free)(void*)) {
  struct damacy* d = container_of(e, struct damacy, io_exec);
  return io_queue_post(d->prefetch_io_q, fn, ctx, ctx_free);
}
```

Signatures verified compatible: `prefetch_executor.post(self, fn, ctx,
ctx_free) → int` and `io_queue_post(queue, fn, ctx, ctx_free) → int`
share types and ownership semantics (return 0 → callee owns ctx).

Teardown: `io_queue_destroy(self->prefetch_io_q)` after the prefetch
caches are destroyed (the caches' inflight workers must finish before
the queue dies). Order in destroy_inner:
1. prefetcher_destroy (joins worker, signals lookahead stop)
2. prefetch_cache_destroy ×3 (drains inflight fetchers)
3. io_queue_destroy(prefetch_io_q)
4. legacy zarr caches, stores, etc.

**Drops the planned `store_fs_io_queue` accessor commit** — no longer
needed. The original 10-commit sequence is now 9 commits.

## damacy_create rewiring (lifecycle.c)

Replace at current `damacy.c:646–651`:
```c
self->meta_cache = zarr_meta_cache_create(self->store_host, cfg->tuning.n_zarrs_meta_cache);
self->shard_cache = zarr_shard_cache_create(self->store_host, cfg->tuning.n_shards_meta_cache);
```
With:
```c
/* legacy caches stay for the planner */
self->meta_cache = zarr_meta_cache_create(self->store_host, cfg->tuning.n_zarrs_meta_cache);
self->shard_cache = zarr_shard_cache_create(self->store_host, cfg->tuning.n_shards_meta_cache);

/* new prefetch caches + fetchers */
self->io_exec.post = damacy_io_exec_post;
array_meta_fetcher_init(&self->amf, self->store_host);
shard_index_fetcher_init(&self->sif, self->store_host, /* amc set below */ NULL);
chunk_layout_fetcher_init(&self->clf, self->store_host, NULL, NULL,
                          resolved_max_substreams_per_chunk);

self->amc = prefetch_cache_create(&(struct prefetch_cache_config){
  .capacity = cfg->tuning.n_zarrs_meta_cache,
  .max_probe = 16,
  .ops = &array_meta_ops,
  .fetcher = &self->amf.base,
  .executor = &self->io_exec,
});
/* re-init sif/clf with the real cache pointers — fetcher_init zeroes the
   struct so we have to either: (a) split fetcher_init into _init + _wire_caches,
   or (b) construct caches first then call fetcher_init. (b) requires the
   chunk_layout_fetcher's array_meta_cache/shard_index_cache fields to be
   patched post-creation. Pick (b) — small static struct reset is cheap. */
shard_index_fetcher_init(&self->sif, self->store_host, self->amc);
self->sic = prefetch_cache_create(...);  /* shard_index_ops, &self->sif.base, ... */
chunk_layout_fetcher_init(&self->clf, self->store_host, self->amc, self->sic,
                          resolved_max_substreams_per_chunk);
self->clc = prefetch_cache_create(...);  /* chunk_layout_ops, &self->clf.base, ... */
```

After scheduler_create (current line 709), add `prefetcher_create` +
`prefetcher_start`:
```c
self->pf = prefetcher_create(&(struct prefetcher_config){
  .lookahead = &self->lookahead,
  .array_meta_cache = self->amc,
  .shard_index_cache = self->sic,
  .chunk_layout_cache = self->clc,
  .capacity = cfg->lookahead_batches * cfg->batch_size,
  .batch_capacity = cfg->lookahead_batches + 2,  // DAMACY_N_BATCH_SLOTS
});
prefetcher_start(self->pf);  /* must come before scheduler kicks */
```

Order: prefetcher_create *before* scheduler_create so the worker can
admit samples the instant `damacy_push` lands one. Actually — neither
order matters until push, but matching destroy order (scheduler →
prefetcher → caches) is cleanest if create mirrors it reversed.

## damacy_destroy / destroy_inner rewiring

Insert after `scheduler_destroy` (current line 427) and before
`wave_pool_destroy` (line 430):
```c
prefetcher_destroy(self->pf);  /* signals lookahead stop, joins worker */
self->pf = NULL;
```

Insert after `zarr_meta_cache_destroy` chain (current 441–444) — or
better, replace with prefetch cache teardown in reverse dependency
order:
```c
prefetch_cache_destroy(self->clc); self->clc = NULL;
prefetch_cache_destroy(self->sic); self->sic = NULL;
prefetch_cache_destroy(self->amc); self->amc = NULL;
zarr_shard_cache_destroy(self->shard_cache); self->shard_cache = NULL;  // PR-1 only
zarr_meta_cache_destroy(self->meta_cache); self->meta_cache = NULL;     // PR-1 only
```

## damacy_push reshape (push.c)

New body (replaces current 767–803):
```c
struct damacy_push_result damacy_push(struct damacy* self,
                                      struct damacy_sample_slice samples) {
  struct damacy_push_result r = { .unconsumed = samples, .status = DAMACY_OK };
  if (!self || samples.beg > samples.end) { r.status = DAMACY_INVAL; return r; }

  scheduler_lock(self->sched);
  if (self->failed_status != DAMACY_OK) { r.status = DAMACY_SHUTDOWN; goto Done; }

  for (const struct damacy_sample* s = samples.beg; s != samples.end; ++s) {
    /* cfg-only validations that survive at push */
    if (!s->uri)                                          { r.status = DAMACY_INVAL; r.unconsumed.beg = s; goto Done; }
    if (s->aabb.rank == 0 || s->aabb.rank > DAMACY_MAX_RANK) { r.status = DAMACY_RANK; r.unconsumed.beg = s; goto Done; }
    if (s->aabb.rank != self->cfg.sample_rank)            { r.status = DAMACY_RANK; r.unconsumed.beg = s; goto Done; }
    if (!sample_aabb_extents_match_cfg(&self->cfg, &s->aabb)) { r.status = DAMACY_INVAL; r.unconsumed.beg = s; goto Done; }

    if (self->lookahead.size == self->lookahead.cap)      { r.status = DAMACY_AGAIN; r.unconsumed.beg = s; goto Done; }

    uint64_t batch_id = self->pushed_samples / self->cfg.batch_size;
    if (lookahead_push_with_batch(&self->lookahead, s, batch_id)) {
      r.status = DAMACY_OOM; r.unconsumed.beg = s; goto Done;
    }
    self->pushed_samples++;
    self->pushed_batch_id = batch_id;
  }
  r.unconsumed.beg = samples.end;
Done:
  scheduler_unlock(self->sched);
  return r;
}
```

**Dropped from push:** zarr_meta_cache_get, cast_path_supported, the
sample-rank-vs-zarr-rank check. Those errors now surface at `damacy_pop`.

**Lock note:** push still takes scheduler_lock — lookahead_push has its
own mutex, but the `pushed_samples` cursor + `failed_status` read need
exclusion from the scheduler. Could be relaxed in PR-2 to a fine-grained
push_lock, but holding scheduler_lock matches current behavior for the
mid-flight stall risk (which is the same).

## damacy_plan rewrite (plan.c)

`plan_reserve` (current 204–229) replaces lookahead_drain with a
prefetcher pop loop, all under scheduler_lock:

```c
static enum damacy_status plan_reserve(struct damacy* self, uint16_t slot_idx,
                                       uint32_t n_samples) {
  if (n_samples == 0) return DAMACY_OK;
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  if (slot->state != BATCH_FREE) return DAMACY_INVAL;

  uint64_t head_batch_id = self->next_batch_id;  /* see note below */
  for (uint32_t i = 0; i < n_samples; ++i) {
    if (!prefetcher_pop_ready_for_batch(self->pf, head_batch_id, &self->staging[i])) {
      /* shouldn't happen — scheduler's gate function already verified
         availability. If it does, return AGAIN and rewind. */
      for (uint32_t j = 0; j < i; ++j)
        prefetcher_ready_free(&self->staging[j]);
      return DAMACY_AGAIN;
    }
    if (self->staging[i].state == PREFETCHER_ERROR) {
      /* latch error and bail; flush slot, free remaining */
      enum damacy_status es = self->staging[i].err_code
                              ? (enum damacy_status)self->staging[i].err_code
                              : DAMACY_INVAL;
      for (uint32_t j = 0; j <= i; ++j)
        prefetcher_ready_free(&self->staging[j]);
      self->failed_status = es;
      return es;
    }
  }

  enum damacy_status status = batch_pool_allocate(self);
  if (status != DAMACY_OK) {
    for (uint32_t i = 0; i < n_samples; ++i)
      prefetcher_ready_free(&self->staging[i]);
    self->failed_status = status;
    return status;
  }
  for (uint32_t i = 0; i < n_samples; ++i) {
    self->batch_samples_pinned[i].uri = self->staging[i].uri;     /* transfer */
    self->batch_samples_pinned[i].aabb = self->staging[i].aabb;
    /* h_meta / h_shards / h_layout: PR-1 doesn't use them
       (planner still uses legacy cache_get). PR-2 carries them through. */
  }
  slot->n_samples = n_samples;
  slot->state = BATCH_PLANNING;
  return DAMACY_OK;
}
```

**Important:** `head_batch_id`. The scheduler picks "the next batch to
plan"; that's `self->next_batch_id` (incremented in plan_commit when
state becomes BATCH_FILLING). Two scheduler scenarios:
- pf has batch_size ready slots for batch_id N → pop them, plan.
- pf has slots for some N, M > N, but < batch_size for N → wait (no plan).
- pf has full batch_size for N + partial for M → plan N only.

The "is N's batch ready" predicate becomes a new helper (see scheduler.c).

`plan_run` (current 234–277): unchanged except the `self->batch_stage`
reference becomes `self->batch_samples_pinned`.

`plan_commit` (current 280–330): one change at line 289–290 — replace
`sample_slot_clear(&self->batch_samples[i])` with
`prefetcher_ready_free(&self->staging[i])`. The `batch_samples_pinned`
URIs are transferred but the staging entry's `h_shards` allocation needs
freeing (`prefetcher_ready_free` does both — uri free + h_shards free).
After release, the `batch_samples_pinned[i].uri` becomes dangling — so
either: (a) `batch_samples_pinned[i].uri` is also strdup'd at transfer,
or (b) ownership is split (uri to `batch_samples_pinned`, h_shards array
stays on staging and is freed here).

Choose (b): in plan_reserve, set `self->staging[i].uri = NULL` after the
transfer; then `prefetcher_ready_free` doesn't double-free. Simpler than
double-allocating.

Additional in plan_commit on success: after `self->next_batch_id++`,
call `prefetcher_advance_watermark(self->pf, self->next_batch_id)`. (The
watermark is exclusive — entries with `max_batch_id < watermark` evict.
So advancing to next_batch_id means batch IDs 0..next-1 are unpinned.
For PR-1 with planner still using legacy caches, this is purely a
hygiene call — won't affect correctness.)

## damacy_scheduler rewrite (scheduler.c)

`kick_peel_into_free_slots` (current 336–383): change the "is a fresh
batch plannable?" gate at line 345 from
```c
if (self->lookahead.size < self->cfg.batch_size) break;
```
to
```c
if (!prefetcher_batch_full_ready(self->pf, self->next_batch_id,
                                  self->cfg.batch_size)) break;
```

`prefetcher_batch_full_ready(p, batch_id, n)` is a NEW prefetcher API:
returns 1 iff p has ≥ n slots in terminal state (READY or ERROR) for
this batch_id. Implementation: linear scan over slots (acceptable for
PR-1 sizes).

Add this to `prefetch/prefetcher.h` + `.c` + a test. Belongs in this PR
(it's specifically for the rewrite consumer).

Alternative: greedy pre-pop into `self->staging` inside the gate check,
return slots to the prefetcher if count < batch_size (push back not
supported). The pre-pop approach loses ordering safety, so use the
new `_full_ready` query instead.

`damacy_scheduler_step` (current 389–412): no logic change. Still calls
`kick_peel_into_free_slots`; the prefetcher worker runs independently.

## damacy_pop / flush / stats rewrite (pop.c)

`damacy_pop` idle-AGAIN gate (current 834–840) widens to include
prefetcher state:
```c
if (!any_wave_in_flight(&self->wave_pool) &&
    !any_slot_in_flight(&self->wave_pool) &&
    !any_batch_in_flight(&self->batch_pool) &&
    lookahead_size(&self->lookahead) == 0 &&
    prefetcher_in_flight(self->pf) == 0 &&
    !prefetcher_has_ready(self->pf)) {
  r = DAMACY_AGAIN;
  goto Done;
}
```

Two new prefetcher accessors: `prefetcher_in_flight(p)` and
`prefetcher_has_ready(p)`. The first reads `prefetcher_stats.in_flight`
without a separate lock acquire; the second scans for any
READY/ERROR slot. Both small additions; same PR as the rewrite.

`damacy_flush` (current 958–1029): the body becomes harder because the
prefetcher is now the gatekeeper for the tail. New steps after the
existing failed_status check:

1. Wait until lookahead drained AND prefetcher has resolved every
   in-flight sample:
   ```c
   while (lookahead_size(&self->lookahead) > 0 ||
          prefetcher_in_flight(self->pf) > 0) {
     if (self->failed_status != DAMACY_OK) { r = self->failed_status; goto Done; }
     SCHEDULER_WAIT_DIAG(self->sched, 5000);
   }
   ```
2. If `prefetcher_has_ready(self->pf)`, the tail samples are ready.
   Pre-count how many: `n_tail = prefetcher_ready_count_for_batch(self->pf, head_id)`.
   New API — see below.
3. If `n_tail > 0 && n_tail < batch_size`, plan the truncated tail:
   `plan_reserve(self, slot, n_tail)`. (plan_reserve already handles the
   "pop n_tail samples for head_batch_id" case.)
4. Then the existing in-flight-wait loop (current 1015–1020) stays.

New: `prefetcher_ready_count_for_batch(p, batch_id)`. Same as
`_batch_full_ready` but returns the count instead of a 0/1 predicate.
Make `_full_ready` call this internally.

`damacy_stats_get` (current 1056–1086): replace zarr_meta_cache /
zarr_shard_cache stats reads with three `prefetch_cache_stats` reads.
Per plan §8 of damacy_rewrite_plan, keep the *names* (zarr_meta_hits /
shard_idx_hits) populated from `amc` and `sic` for ABI compat in PR-1;
PR-2 renames + adds chunk_layout fields.

```c
if (m->amc) {
  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(m->amc, &cs);
  out->zarr_meta_hits = cs.counters.hits;     /* map prefetch_cache stats names */
  out->zarr_meta_misses = cs.counters.misses;
}
if (m->sic) {
  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(m->sic, &cs);
  out->shard_idx_hits = cs.counters.hits;
  out->shard_idx_misses = cs.counters.misses;
}
```
(Drop the legacy `m->meta_cache` and `m->shard_cache` stats reads even
though those caches still run — they're now redundant with the prefetch
caches, and the user-visible numbers should reflect what the prefetcher
sees.)

`damacy_release`, `damacy_release_event`, `damacy_batch_info`,
`damacy_stats_reset`: no logic change. Pure move.

## CMakeLists.txt (src/)

Around line 238:
```cmake
add_cuda_lib(damacy
    SOURCES damacy.h
            damacy_internal.h
            damacy_lifecycle.c
            damacy_push.c
            damacy_plan.c
            damacy_pop.c
            damacy_scheduler.c
            damacy_status.c
    LINKS platform planner zarr_meta_cache zarr_shard_cache zarr store
          decoder assemble dtype log lookahead gpu_budget
          batch_pool wave_pool wave_budget damacy_stats damacy_config
          damacy_nvtx strbuf scheduler
          prefetch_cache prefetcher array_meta shard_index chunk_layout
          CUDA::cuda_driver
)
```
Drop `damacy.c` from SOURCES; add the 5 new C files + the internal h.
Add the 5 prefetch libs to LINKS. Keep `zarr_meta_cache` /
`zarr_shard_cache` in LINKS for PR-1 (planner still needs them).

`add_src_lib(planner ... LINKS ... )` — no change in PR-1.

## Test impact

| Test | PR-1 impact |
|------|-------------|
| `test_damacy.c` | Should pass unchanged — push validations that move to pop are not exercised in this C test (all test URIs exist). |
| `test_damacy_caps.c` | Cfg-only checks (rank, sample shape) still at push; unchanged. |
| `test_damacy_blosc.c` | Unchanged (end-to-end golden image). |
| `test_damacy_ctx.c` | Unchanged. |
| `python/tests/test_damacy.py` | **Changes required** — `test_unknown_uri_raises_notfound`, `test_unsupported_src_dtype_raises_dtype_mismatch`, `test_push_error_drops_offending_iterator` all assume push validates. Rewrite to expect the error at pop. *Defer to PR-3* per `damacy_rewrite_plan §10`. |
| `python/tests/test_native_internals.py` | Verify single DamacyError check still works. |
| `test_zarr_meta_cache.c`, `test_zarr_shard_cache.c`, `test_zarr_cache_threading.c` | Stay in PR-1 — the modules still exist. Delete in PR-2. |

**PR-1 is C-only behavioral.** Python tests stay broken at the push
contract until PR-3; we mark them as expected-failure or skip them in
CI for PR-1, with a note in the PR description. (Skipping is preferable;
xfail leaks state.)

## Commit sequence inside PR-1

1. `prefetcher: batch readiness query` — adds `prefetcher_batch_full_ready`,
   `prefetcher_ready_count_for_batch`, `prefetcher_in_flight`,
   `prefetcher_has_ready`. Pure additions; existing tests pass.
   *(Already landed on `worktree-prefetch` via cherry-pick.)*
2. `damacy: introduce damacy_internal.h` — extract struct/typedef
   definitions only; damacy.c includes the new header in place of inline
   definitions; nothing else moves. Build remains identical.
3. `damacy: extract lifecycle.c` — move damacy_create, damacy_destroy,
   destroy_inner, ctx_guard, damacy_get_device, damacy_config_describe,
   damacy_set_gpu_bytes_committed_for_test. damacy.c shrinks; build
   passes.
4. `damacy: extract push.c, plan.c, pop.c, scheduler.c` — move the
   remaining functions. damacy.c is deleted (or stays as a stub
   `#include`s). CMakeLists updated. Build passes; tests pass (no
   behavior change yet).
5. `damacy: wire prefetcher into lifecycle` — add a dedicated io_queue +
   the 3 prefetch caches + fetchers + executor adapter + prefetcher to
   damacy_create / destroy. Prefetcher runs but its outputs are unused.
   Existing tests still pass.
6. `damacy: rewire push to skip URI validation` — drop the
   `zarr_meta_cache_get` block from push. C tests still pass; Python
   tests start failing on the moved error semantics — annotate with
   `pytest.skip(reason="PR-3 will update expected errors")`.
7. `damacy: rewire plan and scheduler to prefetcher` — plan_reserve pops
   from prefetcher; scheduler gate uses
   `prefetcher_batch_full_ready`; plan_commit advances the
   watermark.
8. `damacy: rewire pop/flush idle predicates` — include prefetcher
   state in `damacy_pop`'s AGAIN gate and `damacy_flush`'s wait loops.
9. `damacy: update stats to read prefetch caches` — `damacy_stats_get`
   reads amc/sic counts instead of legacy caches.

Each commit builds + tests independently. The PR is reviewable
commit-by-commit.

## Open questions / risks

1. **head_batch_id derivation.** Plan uses `self->next_batch_id` as
   "next batch to plan". But `next_batch_id` is incremented in
   `plan_commit` *after* `slot->batch_id = self->next_batch_id++`. So
   inside plan_reserve, `self->next_batch_id` is the about-to-be-planned
   value. Confirm by tracing: if plan_reserve is called when 2 batches
   already committed (next_batch_id=2), it pops slots for batch_id=2.
   Looks right.
2. **Stale gating.** Between scheduler's `prefetcher_batch_full_ready(N)`
   check and `plan_reserve(N)` pop, another scheduler tick can't
   intervene (under scheduler_lock). The prefetcher worker can advance
   slots, but only state transitions FREE→PENDING or PENDING→READY/ERROR
   (never the reverse). So full_ready is monotonic for a given batch_id;
   no TOCTOU.
3. **Failed-prefetch flush.** If plan_reserve hits a PREFETCHER_ERROR
   sample, it latches `self->failed_status`. flush detects this on its
   `failed_status` re-check after the wait. damacy_pop surfaces it as
   the new error path; document in PR description.
4. **Watermark advance + legacy caches.** Advancing the prefetcher
   watermark unpins entries in amc/sic/clc; the planner uses
   `meta_cache`/`shard_cache` (legacy) which have independent LRU
   semantics. No interaction. PR-2 unifies.
5. **`store_fs_io_queue` accessor.** Lives in `src/store/store_fs.h`;
   simple getter, no concurrency concerns (the io_queue is created at
   store_fs_create and lives until store_destroy).
6. **`io_exec.post` signature mismatch.** Need to verify
   `prefetch_executor.post` and `io_queue_post` agree on `(fn, ctx,
   ctx_free)` arity + types before writing the shim. If they differ, the
   shim is non-trivial.
7. **batch_capacity sizing.** Plan picks
   `cfg->lookahead_batches + 2`. Hardcoded `2` matches
   `DAMACY_N_BATCH_SLOTS`. Use the macro if available; otherwise add it
   to damacy_limits.h.
8. **Prefetcher capacity overflow.** Push computes batch_id from
   `pushed_samples / batch_size`. After ~lookahead_batches of pushes
   without pop, the prefetcher's `batch_capacity` is exhausted and
   `admit_locked` silently fails. **This is a real bug** in the current
   prefetcher — admit failure should set the slot to ERROR so the user
   sees the saturation. File as a prefetcher-layer follow-up; PR-1 can
   sidestep by sizing `batch_capacity` to cover the worst case.

## Out of scope (PR-2 / PR-3)

- Delete `zarr_meta_cache` / `zarr_shard_cache` modules.
- Rewire planner to consume `prefetch_handle h_meta` / `h_shards[]` /
  `h_layout` per-sample (drop `zarr_meta_cache_get` etc. in planner).
- Config field rename (`n_zarrs_meta_cache` → `n_array_meta_cache`,
  add `n_chunk_layout_cache`).
- Python test rewrites for push-vs-pop error placement.
- Doc updates (api.md, distributed.md, troubleshooting.md).
- Bench JSON key migration.
