# Decompression Bottleneck Audit

> Pre-optimization audit. Decompression is taking too much GPU time and
> slowing training; before committing to a Karpathy-style measure-and-tune
> loop, this report identifies the cleanup work that needs to land first
> so a profile actually points at the right thing.

## TL;DR (recommendation)

The pipeline is double-buffered on paper but **single-stream on the GPU**:
`stream_zstd`, `stream_lz4`, and `stream_compute` are all owned by
`struct damacy` (`src/damacy.c:327-330`), not by `struct damacy_wave`.
Per-wave `cuStreamWaitEvent`s queue wave B's nvcomp launch behind wave
A's on the same FIFO stream, so wave-level GPU concurrency is structurally
zero. The two-wave structure overlaps host I/O with GPU work but does not
overlap one wave's GPU work with another's. **If GPU is the bottleneck,
the effective wave parallelism is 1.** This is the single most consequential
finding in the audit and is consistent with "GPU usage too high relative
to throughput."

Three secondary issues compound it:

1. The host blosc1 parse runs synchronously on the user thread inside
   `kick_h2d` (`src/damacy.c:1361-1372`), with `bulk_h2d_end` already
   recorded — so `stream_h2d` sits idle for `parse_ms + fanout_h2d_ms`
   on every wave, and the codec streams (gated on `h2d_end`) wait that
   long before nvcomp starts.
2. nvcomp temp-workspace `max_batch_size` is queried at the worst-case
   compile-time cap (`16 384` zstd substreams; `131 072` LZ4 substreams
   at `max_bpe = 8`), and is allocated 2× because of the wave double-buffer
   that doesn't actually buy GPU concurrency. Tens to hundreds of MB are
   committed for a regime where typical waves dispatch a few hundred
   substreams.
3. `damacy_pop` is the only thing that drives state transitions
   (`advance_waves` is invoked from inside it, `src/damacy.c:2159`). All
   host pipeline work — including the parse — runs on whichever thread
   calls pop. In a real trainer, the pipeline halts during forward/backward.

**Suggested order of operations**, with rationale below:

1. NVTX ranges + per-wave streams. _Unblocks measurement._
2. Move host parse off the pop thread.
3. Lazy + (until per-wave streams land) shared nvcomp workspaces.
4. Right-size workspace `max_batch_size` from observed substream counts.
5. Pull `build_blosc1_host_chunks` and `build_assemble_meta` into `peel_wave`.
6. Parameterization renames / config-surface cleanup.

After step 1 the timeline will tell you whether subsequent steps land in
the order above or get reshuffled.

---

## 1. The big architectural starvation: global codec & compute streams

`struct damacy` owns four CUDA streams (`src/damacy.c:327-330`):

```c
CUstream stream_h2d;
CUstream stream_compute;   // post-decode + assemble for ALL waves
CUstream stream_zstd;      // nvcomp Zstd batch for ALL waves
CUstream stream_lz4;       // nvcomp LZ4 batch for ALL waves
```

Wave events (`zstd_done`, `h2d_end`, `asm_end`) are per-wave, but
`kick_codec_batches` (`src/damacy.c:1530, 1548`) and `kick_post_decode`
(`src/damacy.c:1581-1584`) push every wave's work onto the same FIFO
streams. Wave B's nvcomp launch *cannot* begin until wave A's has
retired. `stream_compute` is the worse offender — it carries
post-decode + memcpy + (bit)unshuffle + D2H + assemble, all of which
serialize wave A's chain in front of wave B's anything.

The TODO already calls out per-wave codec streams (`TODO.md:5-9`), but
flagging only `stream_zstd` / `stream_lz4` understates the issue:
`stream_compute` is where most of the wave's wall time on the GPU lives,
and the H2D of assemble metadata (`kick_assemble`, `src/damacy.c:1628-1632`)
sits on it too.

**Fix.** Move the three per-wave streams into `struct damacy_wave`:

```c
struct damacy_wave {
  // ...
  CUstream stream_zstd;
  CUstream stream_lz4;
  CUstream stream_compute;
};
```

Allocate in `wave_init` (`src/damacy.c:821`) with `CU_STREAM_NON_BLOCKING`,
destroy in `wave_destroy` (`src/damacy.c:1003`). The existing
`cuStreamWaitEvent`/`cuEventRecord` chain works unchanged — events are
already per-wave; only the streams need to split. `stream_h2d` may want
to stay global so the bulk-DMA path is in-order, but it's a coin-flip; if
it splits too, the H2D-driven gap in §2 also disappears.

**Why this is step 1.** Until wave-level concurrency exists on the GPU,
no other GPU optimization will measure correctly: every change will be
masked by the serial fan-in. NVTX is the second half of step 1 — see §6.

---

## 2. Stream-idle gaps inside one wave

`kick_h2d` does, in order (`src/damacy.c:1343-1461`):

1. `cuEventRecord(h2d_start)` on `stream_h2d`
2. Bulk `cuMemcpyHtoDAsync` (the slab) on `stream_h2d`
3. `cuEventRecord(bulk_h2d_end)` — *before* parse runs
4. `blosc1_host_parse` synchronously on the calling thread
   (`src/damacy.c:1361-1372`), with the threadpool fanning out the
   chunk-level work
5. Conditional fanout SOA + op H2Ds on `stream_h2d`
6. `cuMemsetD8Async` of the 4-byte error counter on `stream_h2d`
7. `cuEventRecord(h2d_end)` — codec streams gate on this

During step 4, **`stream_h2d` is idle**: the bulk DMA has retired, no
follow-up work has been queued. Codec streams gating on `h2d_end` (steps
5–7) wait the full `parse_ms + sum(fanout_H2D)` before they can begin. The
header comment at `src/damacy.c:1352-1355` acknowledges the gap.

The parse runs on the thread that called `damacy_pop` (`advance_waves`
is invoked at `src/damacy.c:2159`), which is also responsible for kicking
the next wave, finalizing the previous one, and calling `kick_compute`.

**Fix sketch.** Either:

- **(a)** Submit `blosc1_host_parse` to `compute_pool` from `kick_h2d` and
  return; advance the wave's state to `WAVE_PARSING` (new) and progress to
  `WAVE_H2D` only when both `bulk_h2d_end` is signaled and the parse
  future is ready. The TODO follow-up at `TODO.md:26-31` is in this
  shape.
- **(b)** Move `advance_waves` to a damacy-owned worker so user-thread
  pop calls just dequeue ready batches.

(a) is the smaller diff and gets back several ms per wave on
small-chunk workloads. (b) fixes the broader "no progress without pop"
problem and is closer to what a long-term trainer integration wants.

---

## 3. nvcomp workspaces: not the count you thought, but the size you didn't

The mental model "we have separate workspaces for {raw zstd, blosc-zstd,
blosc-lz4}" is partially out of date.

**Already unified.** `decoder/blosc1_host.c:270-276` (raw `CODEC_ZSTD`)
and `:321-325` (blosc-zstd substreams) both populate the same `zstd`
SOA. `kick_codec_batches` issues one `nvcompBatchedZstdDecompressAsync`
for the union (`src/damacy.c:1531-1537`). One nvcomp call per wave per
codec — that piece of the wish list is already in.

**Actual workspace count: 2 waves × 2 codecs = 4 sets.** Each is `nvcomp
temp + actual_sizes + statuses` (`src/decoder/decoder_zstd.c:18-25`,
`src/decoder/decoder_lz4.c:16-23`). Allocated unconditionally in
`wave_init` (`src/damacy.c:983-989`).

**The size is the real cost.** `wave_init` queries nvcomp via
`decoder_zstd_query_temp_bytes` / `decoder_lz4_query_temp_bytes` with
these caps:

| Codec | `max_batch_size` | `per_substream_uncompressed` |
|---|---|---|
| Zstd | `DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE = 512 × 32 = 16 384` | `runtime_chunk_cap` (default 512 KB) |
| LZ4  | `lz4_subs_per_wave(max_bpe=8) = 16 384 × 8 = 131 072` | `chunk_cap / max_bpe` (default 64 KB) |

(`src/damacy.c:55-66`, `:976-988`). The 2× for the double-buffer
multiplies this. For workloads that dispatch a few hundred substreams
per wave, the worst-case query is wildly pessimistic.

**Fixes, in increasing impact / decreasing safety:**

1. **Lazy allocation.** Move the `decoder_zstd_create` /
   `decoder_lz4_create` calls out of `wave_init` and into the first
   dispatch into that codec. Most NGFF datasets use one inner codec;
   the unused decoder never exists. Surface in `damacy_stats` so the
   user can confirm.
2. **Share across waves while the streams are still global.** Until
   per-wave streams land (§1), `stream_zstd` is FIFO across waves —
   nvcomp temp can be a single `struct damacy`-owned pair, allocated
   once. Promote back to per-wave only when per-wave streams are in.
3. **Decouple SOA capacity from `max_batch_size` for nvcomp temp
   queries.** Right now both share `DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE`
   (`src/damacy.c:983, 879, 502`). The SOA is host-pinned + small device
   pointer arrays — cheap. The `max_batch_size` argument to
   `nvcompBatched*DecompressGetTempSizeAsync` is what scales nvcomp's
   internal temp. Track empirical max substream counts during the first
   N waves and re-query the temp size with a tighter cap. (Alternative:
   re-create the decoder on observed-cap exceedance with a 2× growth.)

---

## 4. The poll-only `damacy_pop` is the host-side scheduler

`damacy_pop` (`src/damacy.c:2144-2216`) is a polling loop:

```c
for (;;) {
  advance_waves(self);     // host-side state transitions
  kick_new_waves(self);    // host-side dispatch + plan
  if (oldest_ready_slot()) return OK;
  if (queue_drained())     return AGAIN;
  platform_sleep_ns(50000); // 50 µs
}
```

`advance_waves` is the only thing that promotes `WAVE_IO → WAVE_H2D
→ WAVE_ASSEMBLE → WAVE_FREE`. **Nothing else drives the pipeline.** A
real trainer's main loop is `pop → forward → backward → step → pop`;
during the middle three, `advance_waves` doesn't run, so:

- I/O completions don't get noticed.
- `kick_h2d` (and its synchronous parse) doesn't kick.
- New waves don't get peeled off batch slots.

This compounds with §2: the parse can't even *start* until you call pop.

**Fix.** Either run `advance_waves` on a damacy-owned worker thread
(fixes "no progress without pop") or make pop itself async — wait on a
semaphore that `finalize_wave` posts. The thread approach requires
careful CUDA context handling (the worker needs the retained primary
pushed); the pattern at `ctx_guard_enter` (`src/damacy.c:353-365`) is
the existing template.

---

## 5. Smaller wins that should ride along

These are individually small but cleanup debt that the optimization loop
will keep tripping on:

- **Promote `peel_wave`-time work earlier.**
  `build_blosc1_host_chunks` (`src/damacy.c:1287-1305`) and
  `build_assemble_meta` (`src/damacy.c:1472-1516`) only need data the
  planner already wrote plus `dev_decompressed_offset` (assigned in
  `peel_wave` itself). Build both arrays in `peel_wave`. Then `kick_h2d`
  queues the assemble-metadata H2D on `stream_h2d` next to the fanout,
  and `kick_assemble` no longer holds an H2D in stream_compute's
  critical path.
- **Drop `n_codec_errors` D2H out of the critical path.**
  `cuMemcpyDtoHAsync` of 4 bytes onto `stream_compute`
  (`src/damacy.c:1605-1609`) sits between the last decode kernel and
  `decomp_end` — it's 4 bytes but it's a sync barrier in the queue.
  Either route it to a dedicated read-only stream, or accumulate into a
  device-side sticky `failed_status` and D2H once at flush/destroy.
- **`gpu_shuffle_op.tail_nbytes` is dead** (already in `TODO.md:19-22`).
- **`blosc1_chunk_hdr.err` codes 1..10** want a named enum
  (`src/decoder/blosc1_host.c:18-43`); the lookup table duplicates
  state.
- **`decoder_status_reduce_launch`** is a separate kernel after every
  nvcomp batch (`src/decoder/status_reduce.cu`). Two extra launches per
  wave on the codec streams; mergeable into the post-decode pass once
  per-wave streams land.
- **`assemble_kernel` `RANK_TPL` cascade** instantiates 1..8
  (`src/assemble/assemble.cu:248-256`); for the user's data rank this
  produces ~7 unused kernels that still go through PTX. If rank is
  knowable at create time, instantiate only what's needed.
- **`gpu_unshuffle` / `gpu_bitunshuffle`** stage every block through
  `dev_unshuffle_scratch`, allocated as a full second copy of the
  arena (`src/damacy.c:948`). If shuffle ops are sparse on real data,
  size scratch off the runtime workload, not the worst case.
- **`damacy_dtype_bpe` returns 0** in an unreachable default
  (`src/damacy.c:378-388`); kill the dead branch or
  `__builtin_unreachable()`.

---

## 6. Profiler hooks needed before any tuning

Without these, a profile capture is unreadable.

- **NVTX ranges.** Push/pop around `peel_wave`, `kick_h2d` (split into:
  bulk H2D, host parse, fanout H2D), `kick_codec_batches` (per codec),
  `kick_post_decode` (per kernel: memcpy, unshuffle, bitunshuffle,
  D2H), `kick_assemble`, `finalize_wave`. Tag the wave index in the
  range name — once per-wave streams land, the timeline reads like a
  Gantt chart.
- **Per-stream named markers.** `cuStreamSetAttribute(...,
  CU_STREAM_ATTRIBUTE_NAME_NV, ...)` for `stream_h2d / stream_compute /
  stream_zstd / stream_lz4` (or per-wave variants); nsys then labels
  rows in the timeline.
- **Cross-wave-overlap counter.** Once per-wave streams are in, add a
  metric "fraction of decompress wall-time during which both waves had
  decomp_start < now < decomp_end." That's the actual figure of merit
  for the whole §1 fix; without it the win is invisible in aggregate
  ms.

---

## 7. Parameterization issues you flagged

These bite immediately once optimization starts:

- **`host_buffer_bytes` / `device_buffer_bytes` are silently halved**
  per wave (`src/damacy.c:2029-2030`). Rename to
  `host_slab_bytes_per_wave` / `dev_arena_bytes_per_wave` (or expose
  the wave count).
- **`device_buffer_bytes` is decompressed-arena-only**, but two more
  device buffers ride on the same workload sizing: `dev_compressed`
  mirror of `host_buffer_bytes` (`src/damacy.c:834`) and
  `dev_unshuffle_scratch` as a full second arena copy
  (`src/damacy.c:948`). Three independent budgets riding on two knobs.
- **`max_bytes_per_element` is documented as global but is LZ4-only.**
  In practice it sizes the LZ4 fanout SOA + nvcomp lz4 temp scratch
  (`lz4_subs_per_wave`, `src/damacy.c:62-66`). Rename to
  `max_blosc_lz4_typesize` or fold into a codec-detection pass at
  planner time.
- **`max_chunk_uncompressed_bytes`** is per-chunk uncompressed (default
  512 KB, ceiling 2 MB), but its real role for nvcomp is "per-substream
  cap." With the unified blosc/raw zstd, "substream" is a blosc block
  for blosc-zstd and the whole chunk for raw zstd. Document, or split
  into two knobs.
- **`lookahead_batches >= 2`** is the only floor — no guidance on
  optimal value.
- **`DAMACY_MAX_CHUNKS_PER_WAVE = 512`** and
  **`DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK = 32`** are compile-time and
  bake themselves into nvcomp temp queries via
  `*_SUBS_PER_WAVE` (`src/damacy.c:55-56`,
  `src/damacy_limits.h:45,62`). Document in `damacy_config` that
  raising them pulls workspace size up by N×.
- **`DAMACY_MAX_IO_THREADS = 32`** (`src/damacy_limits.h:50`) is fine
  but the "consumer NVMe saturates well below 32" comment will go
  stale once GDS/multi-NVMe lands.

---

## 8. What the code already gets right (don't undo)

Useful to enumerate so the cleanup pass doesn't accidentally regress:

- **Unified zstd dispatch.** Raw `CODEC_ZSTD` chunks and blosc-zstd
  substreams share one nvcomp call (§3).
- **Pinned host slabs + device mirrors with byte-identical offsets**
  (`src/damacy.c:1287-1305`) — keeps `kick_h2d` to one bulk DMA plus
  small SOA copies.
- **Page-aligned reads from day one** in the planner
  (`src/planner/planner.c:186-211`) — GDS-ready.
- **`CU_STREAM_NON_BLOCKING`** for all streams
  (`src/damacy.c:1958-1961`) — no implicit serialization with the
  user's training stream.
- **Event-only sync, no `cuStreamSynchronize` in the hot path** —
  `advance_waves` polls via `cuEventQuery` (`src/damacy.c:1768, 1779`).
- **Status reduction is async on the codec stream** — error reporting
  doesn't block the host until finalize.

---

## 9. Out of scope for this audit

- IO-side perf (covered by `dev/audits/issue-3-gds-design.md`).
- Codec-internal nvcomp tuning (block size, decomp-opts).
- Bitshuffle phase-2 access pattern (already in `TODO.md:10-13`).
- Sparse zarr / out-of-bounds AABB handling.
- The Python `damacy.py` binding.

---

## Appendix A — Wave state machine reference

```
WAVE_FREE
  └─ peel_wave (src/damacy.c:1210)
       ├─ assigns chunks, builds store_reads, submits I/O
       └─ → WAVE_IO
WAVE_IO
  └─ advance_waves polls store_event (src/damacy.c:1760)
       └─ kick_h2d (src/damacy.c:1343)
            ├─ bulk H2D on stream_h2d
            ├─ host blosc1 parse (synchronous, user thread)
            ├─ fanout/op H2Ds on stream_h2d
            └─ → WAVE_H2D
WAVE_H2D
  └─ advance_waves polls h2d_end (src/damacy.c:1768)
       └─ kick_compute (src/damacy.c:1656)
            ├─ kick_codec_batches: zstd on stream_zstd, lz4 on stream_lz4
            ├─ kick_post_decode: stream_compute joins both, runs
            │     memcpy + (bit)unshuffle + D2H of error counter
            ├─ kick_assemble: H2D meta + assemble kernel on stream_compute
            └─ → WAVE_ASSEMBLE
WAVE_ASSEMBLE
  └─ advance_waves polls asm_end (src/damacy.c:1779)
       └─ finalize_wave (src/damacy.c:1727)
            ├─ drain timing events into stats
            ├─ slot.chunks_remaining -= n_chunks
            └─ → WAVE_FREE
```

## Appendix B — Sizing constants reference

| Constant | Value | Defined | Used in |
|---|---|---|---|
| `DAMACY_MAX_RANK` | 31 | `damacy_limits.h:7` | planner, assemble |
| `DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES` | 2 MB | `damacy_limits.h:34` | runtime cap ceiling |
| `DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES` | 512 KB | `damacy_limits.h:39` | runtime cap default |
| `DAMACY_MAX_CHUNKS_PER_WAVE` | 512 | `damacy_limits.h:45` | wave SOA caps |
| `DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK` | 32 | `damacy_limits.h:62` | parse + fanout sizing |
| `DAMACY_BLOSC_MAX_TYPESIZE` | 8 | `damacy_limits.h:63` | LZ4 substream multiplier |
| `DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES` | 16 MB | `damacy_limits.h:69` | parse ceiling (defensive) |
| `DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE` | 16 384 | `damacy.c:55-56` | nvcomp zstd `max_batch_size` |
| `lz4_subs_per_wave(8)` | 131 072 | `damacy.c:62-66` | nvcomp lz4 `max_batch_size` |
| `DAMACY_POP_POLL_NS` | 50 000 | `damacy.c:71` | pop spin interval |
