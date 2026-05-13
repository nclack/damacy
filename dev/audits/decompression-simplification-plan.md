# Decompression Simplification Plan

> Follow-up to `decompression-bottleneck.md`. Goal: increase decompress
> throughput by **reducing the number of nvcomp pipelines** standing up
> simultaneously and pushing **as many chunks as we can through a single
> `nvcompDecompress` call**. Opposite trade-off from the audit's §1
> recommendation (which optimized for cross-wave GPU concurrency):
> empirically, multiple concurrent nvcomp launches introduce
> instabilities and the LZ4 pipeline rarely earns the memory it costs.

## Target architecture (one sentence)

**Two H2D-double-buffered waves feed one shared zstd decoder on one
decompression stream, kept saturated by queueing each wave's decode +
post-decode + assemble behind the previous wave's events.**

## Vocabulary

- **Wave** (redefined). The state attached to *one*
  `nvcompBatchedZstdDecompressAsync` invocation: pinned host slab,
  device compressed mirror, device decompressed arena, fanout SOA,
  parse scratch, assemble metadata, per-stage CUevents. Two waves exist
  so wave A's H2D can overlap wave B's decode + assemble.
- **`stream_decode`** (new). The single CUDA stream that carries
  nvcomp decode + post-decode kernels (memcpy / shuffle / bitshuffle) +
  assemble. Replaces today's `stream_compute`, `stream_zstd`,
  `stream_lz4`.
- **`stream_h2d`** (unchanged). Bulk slab + fanout/op H2Ds.
- **Wave queue depth.** Decode is **single-issue**: at most one wave's
  decode is in-flight on `stream_decode` at a time, and a second
  wave's decode is queued behind it via `cuStreamWaitEvent`.

## What changes vs. today

| | Today | After |
|---|---|---|
| nvcomp decoders | 4 (2 waves × {zstd, lz4}) | **1** (zstd, pool-owned) |
| CUDA streams | 4 (h2d, compute, zstd, lz4) | **2** (h2d, decode) |
| nvcomp temp scratch | 2× zstd + 2× lz4 worst-case | 1× zstd, sized off runtime caps |
| LZ4 code paths | live | **removed**; planner rejects `CODEC_BLOSC_LZ4` |
| `DAMACY_MAX_CHUNKS_PER_WAVE` | compile-time `512` | runtime, observe-and-grow |
| `max_bytes_per_element` | sizes LZ4 fanout | **removed** (was LZ4-only) |
| `host_buffer_bytes` / `device_buffer_bytes` | required | **deprecated**; derived from `max_gpu_memory_bytes` |
| `max_gpu_memory_bytes` | optional hard cap | **primary budget knob**, required |
| Events per wave | 10 | 7 (drop `zstd_done`, `lz4_done`, `post_start`, …) |

## Phases

Each phase is independently mergeable. Phase numbers map roughly to PRs.

---

### Phase 1 — Drop LZ4 entirely

**Goal**: remove a code path that's both the largest memory consumer
(`131 072` worst-case substreams × per-substream cap × 2 waves) and the
secondary nvcomp instance contributing to "multiple nvcomps at once."

**Code deletions**
- `src/decoder/decoder_lz4.{c,h}`
- LZ4 branches in `src/decoder/blosc1_host.c` (`emit_one`,
  `walk_count`, `parse_count_one`).
- `wave->h_lz4_fan`, `wave->lz4_fan`, `wave->lz4_decoder`, their alloc
  + destroy entries (`src/wave/wave.c`).
- `wave_pool.stream_lz4`; create/destroy.
- `wave_events.lz4_done`; create/destroy; record/wait sites.
- `lz4_subs_per_wave()` and `DAMACY_BLOSC_MAX_TYPESIZE` from
  `src/damacy_limits.h`.
- `cfg.max_bytes_per_element` from `src/damacy.h` + `resolve_max_bpe()`
  in `damacy_config.{c,h}` (validation moves to "reject if set").
- LZ4 branch in `blosc1_totals` (`n_lz4`) + its uses in
  `kick_codec_batches` / `drain_wave_metrics` (collapsed into the
  unified zstd count after phase 2; until then just always-zero).

**Planner change**
- `planner.c:328` — assert `meta->inner_codec.id != CODEC_BLOSC_LZ4`;
  reject the array at planner time with a clear log message and
  `DAMACY_INVAL` returned upstream.
- `src/zarr/zarr_metadata.{c,h}` — keep the `CODEC_BLOSC_LZ4` enum
  entry so we still recognize it for the rejection path.

**Tests / fixtures**
- Delete LZ4-only test data and fixtures.
- Update any cross-codec tests that incidentally used LZ4 to use zstd.

**Verification**
- Build with `-Wunused-function`; no dead LZ4 helpers linger.
- `damacy_predict_bytes` (or `wave_predict_bytes`) shows the LZ4
  category at 0.
- Integration test against an existing zstd dataset retains parity.

**Expected size**: ~600 lines net deletion.

---

### Phase 2 — Collapse `stream_compute` + `stream_zstd` → `stream_decode`

**Goal**: one stream for all post-H2D GPU work. Removes the cross-stream
join overhead and the audit's §1 "wave-level concurrency was zero
anyway" observation becomes architecturally explicit.

**Code changes**
- `wave_pool.stream_zstd` → renamed `stream_decode`. Drop
  `stream_compute`.
- `kick_codec_batches` (`src/wave/wave.c:627`) — drops the
  `cuStreamWaitEvent(stream_zstd, h2d_end)` (already on stream_decode
  via FIFO); launches nvcomp directly on `stream_decode`.
- `kick_post_decode` — drops `cuStreamWaitEvent(stream_compute,
  zstd_done)`; runs on `stream_decode` in FIFO behind the nvcomp launch.
- `kick_assemble` — already on `stream_compute` today; just renames
  the stream variable.
- `kick_compute` — collapses; the `cuStreamWaitEvent(s, h2d_end)` at
  the top is the only cross-stream sync remaining (from `stream_h2d`).
- Drop `wave_events.zstd_done`, `wave_events.post_start`.
  `decomp_start` moves to "just before the nvcomp launch on
  `stream_decode`." `decomp_end` stays on `stream_decode`.
- Update `drain_wave_metrics` to use the surviving events.

**Cross-wave back-to-back queueing**
- The current code already issues all of wave A's work synchronously
  from the user thread, then transitions to `WAVE_ASSEMBLE`. As long
  as `wave_pool_advance` ticks fast enough to start kicking wave B's
  H2D before wave A's decode retires, FIFO ordering on
  `stream_decode` will queue B's decode behind A's automatically. No
  explicit `cuStreamWaitEvent` between waves needed (same stream =
  FIFO).
- A separate `cuStreamWaitEvent(stream_decode, h2d_end_B)` *is*
  needed because B's H2D runs on `stream_h2d` and B's decode runs on
  `stream_decode`; that's the lone cross-stream barrier per wave.

**NVTX (deferred to phase 6 but design-in-place)**: ranges around
nvcomp launch / memcpy / shuffle / bitshuffle / assemble all on the
same stream make the timeline trivially readable.

---

### Phase 3 — Move `decoder_zstd` to `wave_pool` — **DONE**

**Goal**: one nvcomp temp allocation, one `d_uncompressed_actual_sizes`,
one `d_statuses`. Sized for one wave's worth of substreams (not the
sum), because decodes serialize on `stream_decode`.

**Code changes**
- Move `decoder_zstd*` from `damacy_wave` to `wave_pool`. Create in
  `wave_pool_init`, destroy in `wave_pool_destroy`.
- `decoder_zstd_create`'s `max_batch_size` becomes the runtime cap
  (phase 4 wires that up — until then, use a placeholder constant).
- `kick_codec_batches` reads `wp->zstd_decoder` instead of
  `wave->zstd_decoder`.
- `decoder_status_reduce_launch` reads from the pool-shared statuses
  buffer; the per-wave `d_blosc1_totals->n_codec_errors` still
  accumulates per-wave (status reduce sums into it on the
  wave-currently-being-decoded; serial decode means no race).

**Sizing**
- `wave_predict_bytes` → `pool_predict_bytes`: counts nvcomp temp once
  for the whole pool, not 2×. Wave's own `dev_compressed` +
  `dev_decompressed` + `dev_unshuffle_scratch` still 2×.

**Verification**
- `damacy_stats` (or a fresh accounting endpoint) reports nvcomp temp
  bytes ÷ 2 from today's number, all else equal.

---

### Phase 4 — Runtime substream cap (observe-and-grow) — **DONE**

**Goal**: stop sizing nvcomp temp + SOA arrays for the worst-case
`DAMACY_MAX_CHUNKS_PER_WAVE × DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK = 16 384`
substreams when typical waves dispatch hundreds.

**Mechanism**
- Add `struct decoder_zstd::cur_max_batch` (mutable). Initial value
  comes from a small floor (e.g. 1024) or from `pool_predict_bytes`'s
  initial estimate.
- After parse, if `tot.n_zstd > cur_max_batch`, the decoder grows:
  - `cuMemFree` existing temp + actual_sizes + statuses.
  - Recompute `nvcompBatchedZstdDecompressGetTempSizeAsync` with the
    new cap (next power of 2 ≥ `tot.n_zstd`, capped at a hard ceiling
    derived from `max_gpu_memory_bytes`).
  - `cuMemAlloc` at the new sizes.
- SOA arrays (`h_zstd_fan` / `zstd_fan`) grow in parallel — pool-level
  helper that resizes the 4 host-pinned + 4 device buffers.
- If a grow would push total GPU bytes over `max_gpu_memory_bytes`,
  fail the wave with `DAMACY_OOM` and surface the substream count in
  the log so the user can raise the budget.

**Trade-off**
- First few waves of a workload pay the grow cost. Acceptable since
  amortizes across the run.
- Steady-state cost: zero (cap reaches the workload's max and stays).
- Pathological cases (one outlier wave with 10× the typical
  substream count) cause one grow event; the prior cap stays.

**Removes**
- `DAMACY_MAX_CHUNKS_PER_WAVE` as a compile-time cap on the SOA
  arrays. The constant can stay as a *planner-side hint* for peel
  sizing if useful, but the wave's SOA + nvcomp temp no longer
  derive from it directly.

---

### Phase 5 — Make `max_gpu_memory_bytes` the primary budget knob — **DONE**

**Goal**: one user-facing GPU memory ceiling; everything else
internally derived.

**Public surface changes (`src/damacy.h`)**
- `host_buffer_bytes` → **deprecated**; ignored if set, with a one-line
  log warning. Internal sizing computed from
  `max_gpu_memory_bytes` and the expected compression ratio (rough:
  pick `host_buffer_bytes_per_wave = 0.25 × dev_decompressed_per_wave`
  as a default, knob if needed later).
- `device_buffer_bytes` → **deprecated**; similar.
- `max_gpu_memory_bytes` → **required** (0 still allowed as
  "no cap = use legacy 1 GB-style default"). Validated at
  `damacy_create` to fit at least the smallest viable wave config.

**Internal allocation policy**
- `pool_predict_bytes` becomes the single source of truth for the
  GPU breakdown.
- Wave geometry resolution (host_slab_per_wave, dev_decompressed_per_wave,
  nvcomp_temp, etc.) lives in a new `wave_pool_resolve_sizing()`
  helper. It picks values that fit `max_gpu_memory_bytes`, rejecting
  the config at `damacy_create` if even the minimum doesn't fit.

**Migration**
- Keep the deprecated fields readable for one release cycle.
- Add `damacy_config_describe(const struct damacy_config*, FILE*)`
  that prints the resolved geometry on create.

---

### Phase 6 — NVTX + named streams

**Goal**: post-refactor timeline must read like a Gantt chart for the
next round of tuning to be data-driven.

- `cuStreamSetAttribute(..., CU_STREAM_ATTRIBUTE_NAME_NV, ...)` for
  `stream_h2d` and `stream_decode`.
- NVTX ranges around: `peel_wave`, `kick_h2d` (split: bulk H2D, host
  parse, fanout H2D), nvcomp launch, post-decode (memcpy / unshuffle /
  bitunshuffle), assemble, `finalize_wave`. Tag each with the wave
  index.
- Optional: `damacy_pop`-level NVTX so the user can see when their
  consumer thread is the bottleneck.

---

## Out of scope for this plan

- Per-wave streams for GPU concurrency (audit §1) — explicitly
  rejected.
- Moving the host parse off the pop thread (audit §2) — independent
  win; can land before, after, or in parallel.
- Promoting `peel_wave`-time work earlier (audit §5) — keep on the
  follow-up list.
- The poll-only `damacy_pop` becoming async (audit §4) — separate
  concern.
- GDS / IO improvements — covered by
  `dev/audits/issue-3-gds-design.md`.

## Open questions to revisit during implementation

- **Flush triggers beyond "slab full" + "drain"**: do we want a
  time-based watermark in phase 1-3, or wait until phase 6's profile
  data motivates one?
- **Per-wave decode arena sizing**: phase 5 picks
  `dev_decompressed_per_wave` from `max_gpu_memory_bytes`. If chunks
  are heterogeneous (some near the 2 MB ceiling, most ~64 KB), the
  arena cap can fire as a flush trigger long before the host slab
  fills. May want a separate config field if observed.
- **`DAMACY_MAX_CHUNKS_PER_BATCH`** (16384, planner output) — left
  alone for now; if a real workload pushes against it, raise it
  independently of the wave plumbing.

## Order of operations (suggested merge sequence)

1. Phase 1 (LZ4 drop) — smallest, biggest immediate win.
2. Phase 2 (stream merge) — touches the kick chain; clean baseline
   for the rest.
3. Phase 3 (shared decoder) — flushes the structural duplication.
4. Phase 4 (observe-and-grow) — first phase where typical-workload
   memory drops match the headline.
5. Phase 5 (config consolidation) — public-surface change; do last so
   internal changes are stable.
6. Phase 6 (NVTX) — measurement reset for the next tuning loop.

After phase 3 the architecture matches the target diagram and the
remaining phases are sizing + ergonomics.
