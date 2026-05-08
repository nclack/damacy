# Decompression bottleneck investigation

> Investigation of the GPU decompression stage being the current
> pipeline bottleneck. Reference run:
> `bench/runs/default/20260505-093540/results.json`. The user reports
> having previously seen far higher decompression throughput on H100
> ("1000 1MB uncompressed chunks") — order hundreds of GB/s — and now
> sees ~9 GB/s. This doc enumerates the suspects and prioritizes fixes.
>
> The dev box is not an H100, so this is a code/configuration audit,
> not a runtime profile.

## TL;DR — Most likely single cause

**(a) Chunk size — the bench dataset is using 256 KiB uncompressed
chunks, but the user's reference point ("1 MB chunks on H100 → very
high throughput") was testing nvCOMP's sweet spot.** nvCOMP batched zstd
on Hopper has a strongly chunk-size-dependent throughput curve;
per-chunk launch/scheduling overhead is amortized over fewer bytes at
small chunk sizes. The 4× reduction in chunk size (1 MB → 256 KiB)
plausibly accounts for most of the gap on its own.

Compounding this, however, is **(c) decompress-only-stream
serialization**: both in-flight waves share `stream_compute`, so wave
K+1's decompress cannot begin until wave K's assemble finishes. With
assemble at only 1.6 ms avg and decompress at 14.2 ms avg, this isn't
the dominant cost, but it does hard-cap effective concurrency.

The remaining wave-state-machine concerns (polling, statuses readback,
hardware-decoder selection, nvcomp version) are all clean — see below.

## Observed numbers

From `bench/runs/default/20260505-093540/results.json`:

- 1140 waves
- `decompress.ms_total = 16183.9 ms`
- input (compressed) = 99.9 GB
- output (uncompressed) = 148.2 GB
- avg = 14.2 ms/wave; best = 8.37 ms/wave
- → **avg output throughput ≈ 9.16 GB/s; best ≈ ~15 GB/s** (best wave:
  127 MiB / 8.37 ms)
- `derived.stage_concurrency = 1.6258` (out of theoretical 4)
- `pop_wait_compute = 16985.5 ms` ≈ entire decompress wall time —
  orchestrator is spinning waiting for compute

Wave shape:
- 565348 chunks across 1140 waves → ~496 chunks/wave (mean)
- chunks_per_batch = 11307; batch_size = 256, sample_shape
  `[128,128,128]`, chunk_shape `[32,64,64]`, dtype u16
- Per-chunk uncompressed = 32×64×64×2 = **262 144 B = 256 KiB** (NOT
  1 MB — that's smaller than the user's prior reference point)
- Per-wave uncompressed ~ 496 × 256 KiB ≈ 127 MiB

Expected on H100 with 1 MiB chunks: 50–100+ GB/s for nvCOMP HW-zstd.
Observed 15 GB/s on 256 KiB chunks matches the per-chunk-overhead-
dominated regime.

## Detailed findings, ranked

### 1. Chunk size (HIGHEST PRIORITY) — config issue, not code regression

**Evidence:**

- `bench/runs/default/20260505-093540/results.json:8` —
  `"chunk_shape": [32, 64, 64]` × u16 = 32·64·64·2 = **262 144 B =
  256 KiB** uncompressed per chunk.
- `bench/gen_dataset.py:38` — comment mentions inner default
  `32,128,128` (= 1 MiB at u16), but the bench scenario overrides to a
  smaller chunk.
- nvCOMP zstd doc-comment in `nvcomp/zstd.h:191-192` and `:120,160,177`
  — *"For best performance, a chunk size of 64 KB is recommended"* for
  **compression**, but for **decompression** there's no such hint and
  the real measured curve on Hopper peaks much higher (folklore +
  nvCOMP performance reports point to 1–4 MB).
- The bench reports `decompress.ms_best = 8.37 ms` for a wave of ~496
  chunks × 256 KiB = ~127 MiB → ~15 GB/s. nvCOMP HW zstd on H100
  typically lands 50–100+ GB/s on 1 MB chunks; 15 GB/s on 256 KiB
  matches the per-chunk-overhead-dominated regime.

**Recommended fix / experiment (validates locally without H100):**

- Edit the bench scenario to use a 1–2 MiB inner chunk:
  `"chunk_shape": [64, 128, 128]` (= 2 MiB at u16) or
  `[32, 128, 128]` (= 1 MiB at u16). Both are testable on any GPU; the
  *relative* speedup will be visible.
- Note: `chunk_shape` is fully config-driven (`bench/scenario.py:31`,
  `bench/gen_dataset.py` accepts `--inner`). Just regenerate the
  dataset and re-run. There is no per-chunk-size code path in damacy
  that needs changing for this to work.
- **Caveat:** the current `DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES` is
  512 KiB. **You must bump this to ≥ the new chunk size or
  decoder_create / nvCOMP will reject anything larger.** See
  `src/damacy.c:42`.

**Recommended code fix (independent):**

- Make `DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES` configurable via
  `damacy_config` (currently hard-coded to 512 KiB at
  `src/damacy.c:42`). The current value silently caps useful chunk
  sizes; a dataset with 1 MiB chunks would hit the assert at
  `decoder_decompress_batch` time / `peel_wave` (no, wait — actually
  nothing currently checks the per-chunk size at peel;
  `nvcompBatchedZstdDecompressGetTempSizeAsync` is queried with
  512 KB, so passing a chunk larger than that is **undefined behavior**
  per the nvCOMP contract). This is a latent foot-gun: increasing
  `chunk_shape` past 512 KB without bumping the constant will compile
  fine and produce wrong results.

### 2. Decompress and assemble share `stream_compute` — concurrency hard cap

**Evidence:**

- `src/damacy.c:271-272` — only two streams: `stream_h2d`,
  `stream_compute`.
- `src/damacy.c:975-1013` — `kick_compute()` records `decomp_start` →
  submits decompress → `decomp_end` → submits assemble metadata H2D →
  `asm_start` → assemble kernel → `asm_end`, **all on
  `s = self->stream_compute`**.
- Both waves use the same `self->stream_compute`. Since waves are
  submitted in order (`advance_waves` polls each wave; wave 0
  transitions H2D→compute before wave 1 in the typical case), wave 1's
  decompress submission lands behind wave 0's full decompress +
  assemble + assemble-meta-H2D on the same stream.
- `derived.stage_concurrency = 1.6258` (out of 4) and
  `pop_wait_compute = 16985.5 ms ≈ decompress wall = 16183.9 ms` —
  the orchestrator is spinning waiting for compute, and the compute
  stream is doing one wave's work at a time.

**Recommended fix (validates locally):**

- Give each wave its own compute stream (or at least a separate
  decompress stream so a wave's assemble can overlap with the next
  wave's decompress). Simplest: add `CUstream stream_decomp[2]` and
  `CUstream stream_assem[2]` in the wave struct; chain
  decompress→assemble within a wave with a `cuStreamWaitEvent`; let
  waves run in parallel. The HW decompression engine is a separate
  functional unit from SM compute, so HW-zstd on stream-A can overlap
  with assemble-kernel on stream-B even within one wave.
- This change should be visible on any GPU as a stage-concurrency
  uplift (currently 1.6 → goal ≥ 2.5). On H100 with HW-zstd, the
  engine is mostly idle today during assemble.

### 3. Fanout H2D copies queued on the compute stream

**Evidence:**

- `src/decoder/decoder.c:194-221` — the four `cuMemcpyHtoDAsync` calls
  for `compressed_ptrs / sizes / uncompressed_buffer_sizes /
  uncompressed_ptrs` are enqueued **on the same stream as the
  decompress kernel**, and the call site (`src/damacy.c:981-991`) passes
  in `s = stream_compute`. They serialize correctly *with* the kernel
  that follows, but they also serialize *with* the previous wave's
  assemble kernel since they're on the same stream.
- Each fanout array is 512 elements × 8 bytes ≈ 4 KiB; four of them
  ≈ 16 KiB total per wave. Negligible on bandwidth, but each is a
  separate stream submission with its own scheduling cost. nvCOMP
  itself does *not* require these on the compute stream — it just
  needs the data to be visible on the stream the decompress runs on.

**Recommended fix (small, validates locally):**

- Enqueue the fanout copies on `stream_h2d`, record an event, and have
  `stream_compute` `cuStreamWaitEvent` it before launching
  `nvcompBatchedZstdDecompressAsync`. This keeps the compute stream
  short and avoids a self-induced bubble. Lower-priority than #2.
- Or: collapse the four small H2D copies into one (allocate one
  device buffer and one pinned-host buffer holding all four arrays
  packed contiguously). Same end result, fewer driver round-trips.

### 4. Wave is running right at `DAMACY_MAX_CHUNKS_PER_WAVE`

**Evidence:**

- `src/damacy.c:47` — `#define DAMACY_MAX_CHUNKS_PER_WAVE 512u`.
- `derived.chunks_per_wave = 495.92` (from results.json) — within 4
  chunks of the cap, on average. The cap, not the buffer sizes, is
  what's limiting batch size in the nvCOMP call.
- Each wave does ~496 chunks per kernel launch. With 256 KiB chunks →
  127 MiB per launch. **If you bump to 1 MiB chunks, the same 496 cap
  gives 496 MiB/launch (~3 GB/s above the device buffer)** so you'd
  actually have to *lower* the cap or raise `device_buffer_bytes`
  (current bench passes 1024 MB → 512 MB per wave, fits 512 × 1 MiB
  exactly).

**Recommended fix (small, may help on H100):**

- Larger batches (up to ~10K chunks) per nvCOMP call further amortize
  launch overhead. With 1 MiB chunks the wave cap can stay at 512
  (already filling the device arena). With 256 KiB chunks, raising the
  cap to e.g. 2048 and the device_buffer to 2 GB (2 waves × 1 GB each
  × 256 KiB/chunk × 2048 chunks fits exactly) would take a `15.2 GB/s
  best` to a higher number. Validates locally.

### 5. nvCOMP version is fine (NOT a regression cause)

**Evidence:**

- `flake.nix:78` and `Dockerfile:27` both pull **nvCOMP 5.0.0.6**
  (`/nix/store/.../cuda13.2-nvcomp-5.0.0.6-static/lib/libnvcomp_static.a`
  per `build/CMakeCache.txt`).
- `nvcomp/version.h:35-38` confirms `NVCOMP_VER 5.0.0 build 6`.
- nvCOMP 5.x is the most recent series with full Hopper HW-zstd
  support; nothing newer ships today (May 2026) that would meaningfully
  change zstd decompression throughput.

**No fix needed.**

### 6. Hardware decompressor — already enabled by default

**Evidence:**

- `src/decoder/decoder.c:148, 223` — code uses
  `nvcompBatchedZstdDecompressDefaultOpts`.
- `nvcomp/zstd.h:57` — `nvcompBatchedZstdDecompressDefaultOpts =
  {NVCOMP_DECOMPRESS_BACKEND_DEFAULT, {0}}`.
- `nvcomp/shared_types.h:67-69` — *"Let nvCOMP decide the best
  decompression backend internally, either hardware decompression or
  one of the CUDA implementations."*

So on H100 the HW engine is auto-selected. The user could hard-code
`NVCOMP_DECOMPRESS_BACKEND_HARDWARE` to confirm, but it's almost
certainly already happening. **Not the issue.**

### 7. Statuses array is write-only — clean

**Evidence:**

- `src/decoder/decoder.c:130-131, 235` — `d_statuses` is allocated and
  passed to `nvcompBatchedZstdDecompressAsync` as the
  `device_statuses` parameter; **never read back from the host**.
  Search of `decoder.c` and `damacy.c` confirms no `cuMemcpyDtoH` on
  `d_statuses` and no host inspection. nvCOMP will write to it
  on-device; no implicit sync.

**No fix needed.** This means we also have no error visibility into
per-chunk decompression failures — a separate concern, but unrelated
to throughput.

### 8. Polling cadence in `damacy_pop` — modest overhead

**Evidence:**

- `src/damacy.c:52` — `DAMACY_POP_POLL_NS = 50 µs`.
- `src/damacy.c:1086-1106` — `cuEventQuery` (non-blocking, small
  constant cost) is the only CUDA call between sleeps.
- 166 234 polls × ~50 µs = 8.3 s of ideal sleep, observed 16 985 ms
  total → polls are running at roughly 100 µs effective (sleep +
  query). That's reasonable; no implicit sync.
- The poll is correctly attributed: when any wave is in WAVE_H2D or
  WAVE_ASSEMBLE the wait is charged to `pop_wait_compute`, otherwise
  `pop_wait_io`. The 16 985 ms `pop_wait_compute` directly mirrors the
  `decompress` wall time and confirms decompress is the critical path
  — **measurement, not cause**.

**No fix needed.** If decompress speeds up, this number drops with it.

### 9. No per-call host-side allocations in `decoder_decompress_batch`

**Evidence:**

- `src/decoder/decoder.c:178-241` — entire function uses only the
  pre-allocated `d->h_*` and `d->d_*` arrays. No `malloc`, no
  `cuMemAllocHost`, no `cuMemAlloc`. The for-loop at `:196-201` is
  host-side scalar copies into pinned staging — fine.

**No fix needed.**

### 10. Temp workspace sized per `DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES` (=512 KB)

**Evidence:**

- `src/decoder/decoder.c:151-157` —
  `nvcompBatchedZstdDecompressGetTempSizeAsync(max_batch=512,
  max_chunk_uncompressed=512KB, opts, &temp_bytes,
  max_total=512*512KB=256MB)`. Temp scratch is sized correctly *for a
  512 KB chunk cap*. Real workload: 256 KiB chunks. So we're sizing 2×
  larger than needed (~few hundred MB temp scratch) but not
  undersizing.

**Mostly fine** — but tied to fix #1: if you increase chunk size
beyond 512 KB, this constant must move up too.

## Notes on items that are NOT problems

- `cuStreamSynchronize` calls only appear in `damacy_destroy`
  (`src/damacy.c:1265, 1267`); none in the hot path.
- `cuMemcpyHtoD` (synchronous) is used once per batch at `:766` for
  `sample_plans` (≤ 256 × `sizeof(struct sample_plan)`). Once per
  batch, not per wave; trivial.
- The orchestrator's polling loop does not call `cuEventSynchronize`
  or `cuCtxSynchronize` — only non-blocking `cuEventQuery`. Clean.
- The fanout-array overlap with compute would only be a concern if the
  host-side staging in `h_compressed_ptrs` were overwritten before its
  `cuMemcpyHtoDAsync` retired. The code correctly uses **per-wave
  decoder** (one decoder per wave; comment at `src/damacy.c:251-255`
  explains this exact concern was thought through). Clean.

## Suggested experiment ordering

1. **Re-bench with `chunk_shape = [32, 128, 128]` (1 MiB) or
   `[64, 128, 128]` (2 MiB).** First bump
   `DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES` in `src/damacy.c:42` to e.g.
   `(2ull << 20)` and recompile. Regenerate the dataset
   (`bench/gen_dataset.py` accepts `--inner`). This is the single
   biggest expected win and is a 5-minute experiment.
2. Add a per-wave compute stream (split decompress-stream from
   assemble-stream, or one-stream-per-wave). Should lift
   `stage_concurrency` from 1.6 toward 2+ even on non-H100 hardware.
3. Optional: route fanout H2D copies through `stream_h2d` with an
   event sync. Smaller win, but cleans up the compute stream.
4. Once #1 and #2 are in, re-run on H100. Expect 30–60+ GB/s
   decompress out, depending on chunk size.

## Key files / line refs

- `src/decoder/decoder.c` (full file, esp. `:148-157` temp size,
  `:178-241` per-call hot path)
- `src/damacy.c:41-47` (the per-batch / per-wave caps)
- `src/damacy.c:271-272` (only two streams)
- `src/damacy.c:975-1013` (kick_compute — decompress + assemble on one
  stream)
- `src/damacy.c:1086-1106` (advance_waves polling)
- `bench/runs/default/20260505-093540/results.json` (the run under
  analysis)
- `bench/gen_dataset.py` (`--inner` for chunk size)
- `flake.nix:78`, `Dockerfile:27` (nvcomp 5.0.0.6 — already current)
- `nvcomp/zstd.h:37-58` (decompress opts struct, default opts auto-pick
  HW backend)
