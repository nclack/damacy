# Issue #3 — GDS / cuFile Integration: Design Analysis

> Design analysis for [issue #3](https://github.com/.../issues/3): "try
> io via cufile. IO is not the bottleneck right now, but I'm curious to
> see how it impacts things."

## TL;DR (recommendation)

**Defer until after step 7 (coalescing) lands, then do Option C**: keep
host-staged IO as the default, add a cuFile alternative selected by a
`damacy_config` flag. Land it as a parallel `store` backend
(`store_cufile_create`) keyed off the existing vtable, so the wave state
machine in `src/damacy.c:1071` is untouched. The "IO is not the
bottleneck right now" framing makes this curiosity work, and the right
time to touch IO is after coalescing has stabilized read sizes — exactly
the regime where GDS first stops losing money to register/setup overhead.

The current architecture is *unusually* well-suited to a clean GDS
retrofit: every wave's `dev_compressed` is a fixed-size buffer allocated
once at create-time (`src/damacy.c:593`), the IO target is the only
place file→memory traffic enters, and the IO layer is already abstracted
behind `struct store_vtable` (`src/store/store_fs.h:17-27`). That gives
a *very* small blast radius for the change.

---

## 1. Preconditions, and what's actually missing on this box

### Already in place

- **libcufile in flake.nix.** `cudaPkgs.libcufile` is in `buildInputs`
  and `CUFILE_ROOT` is exported (`flake.nix:80,93,105`). Headers
  (`cufile.h`) and `libcufile.so` are on the include/link path today —
  you can `#include` and link without touching the flake.
- **Pinned device buffers, single-allocation lifetime.**
  `wave->dev_compressed` is `cuMemAlloc`'d once in `wave_init`
  (`src/damacy.c:593-594`) and freed in `wave_destroy`
  (`src/damacy.c:640`). This is the lifecycle `cuFileBufRegister` wants
  — register at create, deregister at destroy, never re-register per IO.
- **Page-aligned reads.** `read_op.file_offset` and `read_op.nbytes`
  are already aligned to `platform_page_alignment()` in the planner
  (`src/planner/planner.c:181-185`, `src/damacy.c:1177,1214`). cuFile
  native mode demands the same — no planner change needed (point 6
  below).
- **One file per shard, opened+cached behind store.**
  `src/store/store_fs.c:9-77` has a key→`platform_file*` cache; cuFile
  needs `cuFileHandleRegister(fd)` per FD, and that maps cleanly onto
  the existing slot.

### Missing on this NixOS dev box

- **nvidia-fs kernel module.** The cuFile native fast path requires
  the `nvidia-fs` kmod to be loaded. NixOS doesn't ship it as part of
  the standard nvidia driver wrapper (`/run/opengl-driver/lib`), so
  `cuFileDriverOpen()` will succeed but in **compat mode only** —
  meaning every "GDS" read actually pages through cuFile's internal
  pinned host buffer + `cuMemcpyHtoDAsync`. That's *worse* than what
  damacy does today (extra copy through a buffer cuFile manages, not
  yours).
- **Compatible filesystem.** Local ext4 on a typical workstation is
  compat-mode-only. Native GDS needs ext4/xfs on NVMe (driver-supported),
  or weka/lustre/gpfs. The "is this scenario ever going to win?"
  question can only be answered on the deployment target.

### Missing in the docker image

- The `Dockerfile` is `nvidia/cuda:13.0.1-devel-ubuntu24.04`
  (`Dockerfile:25`). That base image *does* ship libcufile and headers
  (CUDA 13 toolkit includes them). No extra apt step required for
  *building* against cuFile.
- The image does **not** have `nvidia-fs` or any DKMS bits. That's
  fine — the kmod lives on the host kernel, not in the container.
  Whether the container can use it depends on the host (`--gpus all`
  won't enable nvidia-fs by itself; you also need the host kernel
  module loaded and the device files exposed). No blocker for build,
  possible blocker for runtime — handle with the runtime probe in §5.

**Net:** CI/Docker can compile and link a cuFile-using build today.
Whether it gets the *fast* path is a host question, and your dev box
almost certainly will not.

---

## 2. cuFile API shape — pick one

The path I'd commit to:

```
damacy_create:
  cuFileDriverOpen() once
  for each wave: cuFileBufRegister(wave->dev_compressed, host_slab_cap, 0)

per-shard, lazily inside store_cufile (mirroring fs_get_file in store_fs.c):
  open(O_RDONLY|O_DIRECT) → cuFileHandleRegister(fd) → cache the CUfileHandle_t

per-wave IO submit (from io_queue worker thread, sync):
  cuFileRead(cf_handle, wave->dev_compressed, len, file_off, dst_buf_offset)
```

Why **synchronous `cuFileRead` from `io_queue` worker threads** rather
than `cuFileBatchIOSubmit` or `cuFileReadAsync`:

1. **Drop-in fit with the existing store vtable.** Today's `store_fs`
   posts one job per `store_read` to `io_queue`
   (`src/store/store_fs.c:131-150`). The cuFile equivalent is identical
   except the job body calls `cuFileRead` instead of `pread`. The wave
   state machine, `store_event_query`-driven IO→H2D transition
   (`src/damacy.c:1078-1085`), and the io_queue's seq-based completion
   all continue to work without modification.
2. **`cuFileBatchIOSubmit` is a worse fit for the wave shape.**
   Batching helps when you have N independent reads to overlap their
   queue submission. Damacy *already* has that overlap via N io_queue
   worker threads. Batch IO would fold the parallelism down to one
   thread (the wave kicker on the user thread) calling
   `cuFileBatchIOSubmit` with N entries — losing the existing
   thread-level latency hiding for `open()`/syscall entry on cold FDs.
   You'd also have to plumb a separate completion mechanism
   (`cuFileBatchIOGetStatus` polling) into `damacy_pop`'s state machine.
   Net negative.
3. **`cuFileReadAsync` (CUDA 12.2+) needs a stream and stream-ordered
   completion.** That collapses the IO and H2D stages into one, but
   couples them to the same stream — meaning the wave's
   `WAVE_IO`→`WAVE_H2D` transition gets replaced by a single
   `cuEventQuery` on `stream_h2d`. This is conceptually cleaner but
   requires reshaping the state machine; not worth the risk for an
   exploration. Park this as a follow-up if the sync version shows
   promise.

**One subtlety:** cuFile worker calls *block* the calling thread. With
`n_io_threads` workers in `io_queue`, you get exactly the same
parallelism as today, except each worker is now blocking inside the
cuFile driver instead of in `pread`. cuFile's driver is documented as
thread-safe for this use case and serializes internally as needed for
the kmod path.

---

## 3. Architectural fit — Option C is the right call

Re-reading the three options against the code:

**Option A (workers each call cuFileRead, eliminate host slab + H2D).**
Forces a rewrite of `damacy_wave` (`src/damacy.c:189-256`) to drop
`host_slab`, make `dev_compressed` the IO target directly, and merge
`WAVE_IO` and `WAVE_H2D`. State machine collapses to
`WAVE_IO → WAVE_ASSEMBLE`. **But:** this option only works when GDS is
available. In compat mode it's *strictly worse* than today (extra
cuFile-internal copy). Fall back is `damacy_create` failure or a
runtime branch that re-introduces all the host-slab plumbing — at which
point you've written Option C and called it Option A.

**Option B (orchestrator submits batch on compute path, no io_queue).**
Removes a thread role. Tempting for simplicity, but:

- Single-threaded submission loses the per-FD `open()` overlap currently
  hidden behind worker threads.
- `cuFileBatchIOSubmit` failure modes (per-entry status, partial
  completion) need to be plumbed into the wave state. Adds complexity
  that wasn't there before.
- Doesn't match how the rest of the pipeline (decompress, assemble) is
  already structured around per-wave events.

**Option C (cuFile alternative path, config-flag selected).** The
`store` vtable (`src/store/store_fs.h:17-27`) already exists for exactly
this kind of swap. The wave struct, state machine, and orchestration
loop never see the difference — they call `store_read_submit` and wait
on `store_event`. The cuFile backend either:

   - puts bytes into `wave->host_slab` (compat mode — same as today,
     slightly worse than today, surfaces as a configuration error), or
   - puts bytes into `wave->dev_compressed` directly (native mode), and
     the existing `kick_h2d` becomes a no-op when the cuFile path is
     active.

Concretely, Option C means:

- `struct store_read` grows a `void* dev_dst` (or `dst_kind` enum) so
  the cuFile backend knows whether to write to a host or device buffer.
  Existing callers fill `dst` (host); new code paths fill `dev_dst`.
- `damacy_wave_kick_h2d` (`src/damacy.c:878-892`) gets a guard: if the
  cuFile path was used, skip the `cuMemcpyHtoDAsync` and just record
  `ev.h2d_end` immediately on `stream_h2d` (so the rest of the pipeline
  still gets its expected event). One-line change, retains state-machine
  shape.
- `damacy_create` selects `store_fs_create` or `store_cufile_create`
  based on `cfg.use_gds`.
- The `wave->host_slab` allocation stays (it's harmless, ~MB-scale, and
  we want it as a fallback regardless).

**Verdict:** Option C wins on every axis except peak throughput in the
perfect-deployment case — and even there, you can revisit Option A as a
"pure GDS" fast path *after* Option C ships and proves the win on real
hardware.

---

## 4. Buffer registration lifecycle

`cuFileBufRegister` pins device memory with the cuFile driver. The cost
is a one-shot call into the kmod that records the pages as DMA-pinned;
subsequent `cuFileRead` calls into that buffer skip the pinning step.
Registration cost is documented as ~tens of microseconds per call.

Damacy maps onto this perfectly:

| buffer | created at | freed at | register? |
|---|---|---|---|
| `wave[w].dev_compressed` | `wave_init` (`src/damacy.c:593`) | `wave_destroy` (`src/damacy.c:640`) | **yes** — once, in `wave_init` |
| `wave[w].dev_decompressed` | `wave_init` (`src/damacy.c:595`) | `wave_destroy` (`src/damacy.c:642`) | **no** — never an IO target |
| `wave[w].d_assemble_chunks` | `wave_init` (`src/damacy.c:614`) | `wave_destroy` (`src/damacy.c:644`) | no |
| `slot->dev_ptr` (output tensor) | `batch_pool_allocate` (`src/damacy.c:496`) | `damacy_destroy` | no |

So the change is: in `wave_init`, after `cuMemAlloc` for `dev_compressed`,
call `cuFileBufRegister(dev_compressed, host_slab_bytes, 0)`. In
`wave_destroy`, before `cuMemFree`, call `cuFileBufDeregister(dev_compressed)`.
Total: 4 lines added.

The fact that `dev_compressed` is sized as `host_slab_bytes` (mirroring
the host slab; `src/damacy.c:593`) means the register-one-region pattern
works — every cuFile read targets some `dst_buf_offset` *within* this
region, and a single registered region covers all of them.

---

## 5. Compatibility-mode IO is a real trap

This is the most important paragraph in the analysis: **cuFile's compat
mode is worse than damacy's current path.** When `nvidia-fs` isn't
loaded or the FS doesn't support GDS, `cuFileRead` internally allocates
a pinned host bounce buffer, does a `pread` into it, then
`cuMemcpyHtoDAsync` to the registered device buffer. Your existing
pipeline already does that — but with *your* slab, *your* H2D event,
and *your* threads. Adding cuFile in compat mode just inserts an extra
layer between `pread` and the same H2D copy.

So damacy needs **a startup probe**:

```c
// In store_cufile_create, after cuFileDriverOpen:
CUfileDrvProps_t props;
cuFileDriverGetProperties(&props);
int native_supported =
    (props.nvfs.major_version >= 1) && /* nvfs kmod loaded */
    /* per-mount-point check via cuFileHandleRegister test on a sentinel fd */;
if (!native_supported && !cfg->force_compat) {
    log_warn("GDS requested but native path unavailable; falling back "
             "to host-staged IO. To force compat mode anyway, set "
             "force_compat=1 (not recommended).");
    cuFileDriverClose();
    // return NULL; caller will retry with store_fs_create
}
```

Concretely:

- `cuFileDriverGetProperties` is the public interrogation — gives
  `nvfs.major_version` (0 means kmod not loaded) and a list of supported
  FS IDs.
- For per-mount validation: open a sentinel file in the store root with
  `O_DIRECT`, `cuFileHandleRegister`, then call
  `cuFileBufRegister`+`cuFileRead`+check whether cuFile reports
  compat-mode for that handle. There's a `CU_FILE_DRIVER_PROPS` struct
  field `(NvfsCompatMode)` and per-handle stats can be queried via
  `cuFileGetParameterSizeT`. Worth a few minutes to nail down the right
  call when implementing — the principle is "fail visibly if we'd be
  slower."

Default policy: `cfg->use_gds = 0`. When `=1`, run the probe; on probe
failure, return `DAMACY_INVAL` with a clear log. No silent compat-mode
regression.

---

## 6. Coalescing & alignment — no changes needed

The planner's page alignment (`src/planner/planner.c:181-185`) already
aligns down `entry->offset` and aligns up `entry->offset + entry->nbytes`
to `page_alignment_bytes` (= `sysconf(_SC_PAGESIZE)` from
`platform.posix.c:16`). cuFile native mode requires:

- File offset multiple of 4 KiB (the `nvidia-fs` page size on x86_64).
  Already satisfied — typically same as `_SC_PAGESIZE` = 4 KiB.
- Read length multiple of 4 KiB. Already satisfied.
- Device buffer offset multiple of 4 KiB. The wave's `dst_buf_offset`
  is `host_cursor` accumulated as `+= r->nbytes` (`src/damacy.c:824`),
  starting at 0, with each `nbytes` already a 4 KiB multiple. Satisfied
  transitively.

The coalescing planned for step 7 keeps these invariants — coalesced
reads are still page-aligned page-multiples, just bigger. **No `read_op`
shape changes needed for GDS.**

One small thing worth noting: cuFile prefers larger reads (≥ 1 MiB
amortizes the kmod entry). Coalescing actively helps. This is another
argument for landing GDS *after* step 7 — pre-coalescing, many of
damacy's reads will be small (single ~1 MiB chunk + page padding), and
GDS-native overhead per call dominates. Post-coalescing, reads merge
across chunks within a wave and approach shard-scan sizes where GDS
shines.

---

## 7. Phasing — defer past step 7, ahead of step 8

Looking at `docs/plan.md:33-40`:

> 6. damacy_flush + damacy_stats. ...
> 7. Coalescing in the planner / scheduler. Merge adjacent read_ops within a wave (planner-side; the IO and decompress paths are unchanged).
> 8. IO thread tuning. io_queue size, backpressure on the queue.

GDS belongs in the slot **between 7 and 8**, not now. Reasoning:

1. **Pre-step-7 reads are small.** GDS is a fixed-overhead-per-call
   operation; pre-coalescing reads pessimize that. Measuring GDS now
   would *underestimate* the win.
2. **IO tuning (step 8) decisions depend on which IO path is in play.**
   The right `n_io_threads` for `pread`-based IO is "saturate the SSD
   queue depth" (~16-32). The right number for cuFile is much smaller
   (cuFile internally manages async submission against the kmod;
   ~4 threads is typical). If you do step 8 first, you tune for the
   wrong workload.
3. **The fail-the-stream error model is already in place**
   (`docs/api-design-internals-draft.md` Error model section). Adding a
   new failure mode (GDS probe failure → DAMACY_INVAL at create,
   runtime cuFile errors → DAMACY_IO) drops cleanly into it.
4. **Step 6 (`damacy_stats`) is a prerequisite for the A/B benchmark.**
   The `io` metric (`src/damacy.c:584` shows `bench/main.c` already
   emitting `stats.io`) is what proves GDS-on vs GDS-off. So step 6
   must be in before benchmarking, but does not gate the implementation.

So the order is: **finish 6, do 7, do GDS-as-step-7.5, then 8 with the
right IO model in hand.**

The phrase "IO is not the bottleneck right now" is the user's own
diagnosis. With the wave/decoder pipeline overlapping
IO+H2D+decompress+assemble, IO has to get genuinely fast before it stops
being a wash. That makes this exploratory; don't trade complexity for a
curiosity. Cap it: time-box to a week of work, ship Option C behind a
flag, A/B-bench, document, move on.

---

## 8. Test plan

### What to compare

The `damacy_stats` already emits per-stage cumulative `ms` and `bytes`
counters (`bench/main.c:583-587`). For an A/B:

- `stats.io` — IO submit→completion. Today: `pread` time. With GDS:
  `cuFileRead` time.
- `stats.h2d` — `cuMemcpyHtoDAsync` time. With GDS-native: ~zero
  (kick_h2d becomes a no-op, fires h2d_end immediately).
- `stats.pop_wait_io` — user-thread blocked time waiting for IO event.
  The end-to-end latency hit/win shows up here.
- `wall_ms` — steady-state total. The single number that matters.

### Tests to write

1. **Unit test: `tests/test_store_cufile.c`.** Mirror `test_store_fs`
   if it exists, but skipped unless `DAMACY_HAS_GDS=1` is set in env
   (CI gate). Tests open/handle-register/read/close lifecycle and a
   synthetic-shard read against a `mkdtemp` ext4 file. Will run in
   *compat mode* on most hosts; that's fine for correctness, marks the
   perf as "unmeasured."
2. **Integration test: `tests/test_damacy_gds.c`.** Same fixture as
   `test_damacy.c` but with `cfg.use_gds = 1`. Must produce
   byte-identical output to the non-GDS path against the
   `gen_dataset.py` zarrs. This is the correctness gate.
3. **A/B bench scenario: `bench/scenarios/gds_vs_host.json`.** Two
   `damacy_create` configs, otherwise identical, run back-to-back on
   the same data. `bench/run.py` already does scenario × config matrices.

### CI / hardware reality

- The dev box (NixOS, mid-tier GPU) will run compat-mode GDS only —
  useful for correctness, useless for perf.
- The H100 runner (per memory: dev box is not H100) is where the real
  measurement happens. *If* the H100 runner is in CI, the bench
  scenario runs there; if it's manual-only, document the manual run in
  `docs/devlog.md` and treat the result as a one-off measurement, not
  a CI-tracked metric.
- The Dockerfile build will succeed regardless; `--gpus all` at runtime
  gates the actual cuFile-driver-open call. The probe in §5 turns "no
  nvidia-fs on this host" into a graceful fallback rather than a hard
  fail.

---

## 9. Open questions

1. **Per-mount probe accuracy.** The cuFile docs are equivocal about
   the right way to detect compat-vs-native at runtime *for a specific
   filesystem*. `CU_FILE_DRIVER_PROPS` gives a global view; per-handle
   status is buried in less-documented APIs. Worth a 30-min read of
   the cuFile programming guide *before* committing to a probe shape —
   if it's flaky, fall back to "user must opt in via `force_native=1`
   and we error if a read returns compat-mode bytes."
2. **`O_DIRECT` plus cuFile.** cuFile internally manages the open-mode
   flag; you generally `open(O_RDONLY)` (no `O_DIRECT`) and let cuFile
   decide. But damacy currently passes `o_direct=0` to
   `platform_file_open_read` (the FS store doesn't request it;
   `src/store/store_fs.c:25` just calls
   `platform_file_open_read(strbuf_cstr(&path), 0)`). Verify: is the
   existing store actually getting O_DIRECT? Reading
   `src/platform/platform_io.posix.c:23-34` — no, `o_direct` is
   hardcoded to 0 at the call site. So damacy is *already* using
   buffered IO end-to-end. That means today's host-staging is going
   through the page cache (which is *fine* for repeat reads, but means
   today's IO numbers don't represent cold-cache GDS-vs-pread). **Worth
   fixing the existing code first**: pass `o_direct=1` through
   `store_fs_config` so the baseline is honest.
3. **nvidia-fs on the H100 runner.** Need to confirm with infra: is
   `nvidia-fs` loaded on the runner host? Is the FS holding the bench
   data on a GDS-compatible mount? If both no, the H100 measurement
   will *also* be compat-mode, and there's nothing to measure. Check
   before doing the work.
4. **Multi-instance contention.** `cuFileDriverOpen` is
   reference-counted. Multiple `damacy*` instances in one process work.
   But: each instance registers its own buffers, and `cuFileBufRegister`
   has a per-process limit (driver-defined). If anyone configures large
   `host_buffer_bytes` × multiple `damacy` instances × multiple GPUs,
   this becomes a knob. Document the registration cost in
   `docs/api-design-internals-draft.md`.
5. **Decompression source.** When GDS-native lands, `dev_compressed` is
   filled by cuFile rather than `cuMemcpyHtoDAsync`. The decoder's
   input pointers (`build_decoder_fanout`, `src/damacy.c:895-911`) are
   derived from `dev_compressed + dst_buf_offset + offset_in_read`.
   That math is unchanged. The wave event ordering (`stream_compute`
   waits on `ev.h2d_end`, `src/damacy.c:977`) needs `ev.h2d_end` to
   actually fire after the cuFile read completes — if cuFile reads are
   sync from worker threads, recording `ev.h2d_end` on `stream_h2d`
   *after* the worker returns (i.e., in `kick_h2d`'s no-op variant)
   preserves ordering. Verify this matches what
   `cuStreamWaitEvent(stream_compute, h2d_end)` expects: yes, since
   `cuEventRecord` on a stream just inserts a barrier — recording on
   the empty `stream_h2d` after the data is already on the device is
   correct and gives a 0-time event.
6. **Do we want `cuFileReadAsync` later?** Per §2 it's punted. Once
   GDS lands and shows a win, the natural next step is async +
   stream-event-driven IO, which would enable Option A and dissolve
   `WAVE_H2D` entirely. Frame this as a v2 if v1 measurements look good.

---

## Critical files for implementation

- `src/store/store_fs.c`
- `src/damacy.c`
- `src/store/store.h`
- `src/CMakeLists.txt`
- `src/platform/platform_io.posix.c`
