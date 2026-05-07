# TODO

## blosc1 GPU pipeline follow-ups (deferred from review)

- [ ] **Per-wave nvcomp streams investigation.** `stream_zstd` and
      `stream_lz4` are global. If wave A and B contend on them, wave B
      stalls waiting on A's nvcomp batch. Add nsys markers around the
      `decoder_*_batch_device` calls; if serialization shows up on a real
      timeline, split into per-wave streams (mirrors `stream_h2d`).
- [ ] **Bitshuffle phase 2 access pattern.** `gpu_bitunshuffle_kernel`
      does 8 strided global reads per output byte. Likely high register
      pressure and L2 traffic. Inspect with `--ptxas-options=-v`; if
      worth fixing, stage one bit-plane at a time through smem.
- [ ] **`emit_fanout` O(t × nblocks) prior-count loop.** Each thread
      re-walks earlier blocks' substream chains to find its write base
      (`blosc1.cu:395-417` range). Could compute per-block codec/raw
      counts cooperatively during the rank-sort phase and prefix-scan
      in registers. Defer until profiling says it matters.
- [ ] **`gpu_shuffle_op.tail_nbytes`.** Currently always 0; the shuffle
      kernels never read it. Decide: either populate from the parser
      (true partial-last-block support) or delete the field. Today the
      tail is implicitly zero-padded by the decompressed-arena setup.
- [ ] **`blosc1_chunk_hdr.err` codes.** Numeric 1–8; not decoded by any
      consumer. Either a named enum (`BLOSC1_ERR_TOO_SHORT`, …) or
      collapse to a boolean.
- [ ] **`kick_compute` parse-sync placement.** The `cuEventSynchronize`
      on `parse_done` is on the hot path. Currently safe because
      `stream_compute` is shared across waves (FIFO), so cross-wave
      serialization is implicit there anyway. If we ever go per-wave on
      stream_compute, this becomes a real stall — split into a
      non-blocking parse phase and a query-on-ready phase.

## To triage

- [ ] support for bf16 fp16
- [ ] testing the hashmap/lru
- [ ] add ngff and multiscale
- [ ] handle oob aabb's
- [ ] type translation (e.g. u16 to bf16)
- [ ] eval lru for compressed chunk
      - may be interesting to eval hit rate
      - system's virtual page cache may obviate this for host mem
- [x] how to add back blosc support - will need decompression on the cpu side
- [ ] codegen for h100s

## Fuzzing

- [ ] **Dockerfile + build.sh for the fuzz harnesses.** Base on an
      NVIDIA image (matches the rest of damacy's toolchain — even
      though the fuzz build itself doesn't need CUDA, sharing a base
      keeps things consistent). Use micromamba to install the C
      toolchain (clang, llvm-profdata, llvm-cov) from conda-forge
      rather than apt. Output should drop fuzzer binaries into `/out/`
      so the same image plugs into ClusterFuzzLite or OSS-Fuzz later
      without rework.

- [ ] **GitHub Actions wiring.** Once the Dockerfile exists: per-PR
      smoke job (~60s/harness against checked-in seeds), gated on
      paths affecting the parser (`src/util/{json,slice}.*`,
      `tests/fuzz/**`, the relevant CMake files, `flake.nix`). Fail
      the PR on any `crash-*` file. See `docs/fuzzing.md` for the
      full menu of options.

- [ ] **Scheduled long run with corpus persistence.** Weekly Actions
      cron, ~30 min/harness, restore + save runtime corpus as a
      workflow artifact (or a `corpus` branch). This is where new
      bugs actually surface — seeds-only runs plateau in seconds.

- [ ] **Tighten the differential oracle.** Currently skipped whenever
      the query contains any `SEG_ITER`, including inside
      `SEG_WHERE.path`. If the semantics of "first emission of a
      multi-valued query" are nailed down, the oracle can be
      extended to those cases too.

- [ ] **Corpus minimization step.** Once a runtime corpus has been
      built up, run `-merge=1` to drop redundant inputs. Worth doing
      before persisting as an artifact so the size stays sane.

- [ ] **Structure-aware mutators (only if needed).** If the query
      fuzzer plateaus on coverage, swap the byte-tape decoder for a
      protobuf/LPM grammar that mutates at the `json_seg` level
      directly. Probably overkill at the parser's current size.
