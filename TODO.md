# TODO

## To triage

- [ ] support for bf16 fp16
- [ ] testing the hashmap/lru

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
