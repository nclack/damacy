# Fuzzing the host-side parsers

libFuzzer harnesses for the host-side parsers:

- `fuzz_json_input` — lexer driven by arbitrary input bytes
- `fuzz_json_query` — query evaluator driven by arbitrary `json_query` shapes
- `fuzz_shard_index` — shard-index footer parser
- `fuzz_zarr_metadata` — zarr v3 metadata semantic-layer validation

All live under `tests/fuzz/` and only build when `DAMACY_FUZZ=ON`,
which also gates out the CUDA-dependent slice of the tree (decoder,
pipeline, bench). Requires clang — gcc has no libFuzzer.

## Harnesses

### `fuzz_json_input` — input-bytes fuzzer

Drives `json_resolve` and `json_iter_next` on the fuzz input itself.
Because the parser is lazy (it only lexes the bytes a query actually
touches), a passive harness would leave most of the parser uncovered.
This one runs three queries per input — empty path, `QUERY_ITER` over
root, and a handful of common keys — and calls every `json_as_*`
converter on each emitted node. That drags the lexer through every
child value of the input, so a single fuzz iteration exercises the full
parser.

Bug classes this catches: OOB reads on truncated input, signed-overflow
/ shift UB inside `json_as_int` and friends, brace-balance bugs in the
skip-past-value logic for non-matching keys.

### `fuzz_json_query` — query fuzzer with differential oracle

Decodes the fuzz input as a tape of bytes describing a `json_query`
(arena-allocated, depth- and width-bounded), then evaluates that query
against a small fixed JSON corpus. The corpus is hardcoded inside
`fuzz_json_query.c` — the fuzz input drives the query, not the JSON.
This stresses part dispatch, the bounded iter-frame stack, and
`QUERY_WHERE` recursion, none of which are reachable through the lexer
fuzzer.

The differential oracle: when the decoded query contains no `QUERY_ITER`
anywhere (including inside `QUERY_WHERE.part`), `json_resolve` must agree
byte-for-byte with the first emission of `json_iter_next` on the same
query. Both code paths exist; neither references the other; a
disagreement is a bug. Multi-valued queries skip the oracle — verifying
`json_resolve` returns the first iterator emission would re-implement
the iterator inside the harness.

Bounds on decoded queries:

| Cap                  | Value | Why                                          |
|----------------------|-------|----------------------------------------------|
| parts per level   | 6     | one above `JSON_ITER_MAX_FRAMES` so the OOM check is reachable |
| sub-query depth      | 2     | inside `QUERY_WHERE.part`                      |
| key / rhs literal    | 8     | enough to cover real key shapes              |
| arena size           | 4 KiB | one fuzz iteration of decoded parts           |

### `fuzz_shard_index` — shard-index footer parser

Drives `zarr_shard_index_parse` with both length-derived and
intentionally-mismatched `n_entries`.

### `fuzz_zarr_metadata` — semantic-layer fuzzer

Splices selector-driven value substrings into a templated zarr v3
document, then drives `zarr_metadata_parse` and `zarr_metadata_inner_per_shard`.

## Build

```fish
cmake --preset fuzz
cmake --build build-fuzz
```

Both harnesses link into `build-fuzz/tests/fuzz/`.

## Run

The libFuzzer convention is `./fuzzer <writable_corpus> [seed_dirs...]`.
The first directory is where new findings are saved; subsequent dirs
are read-only seeds. Mixing the two in one directory pollutes the seeds
with whatever the fuzzer discovers, so we keep them separate:

- `tests/fuzz/seeds/{input,query,shard_index,zarr_metadata}/` —
  hand-crafted seeds, checked in.
- `build-fuzz/corpus/{input,query,shard_index,zarr_metadata}/` —
  runtime corpus, gitignored via the `build-*` entry in `.gitignore`.

```fish
mkdir -p build-fuzz/corpus/input \
         build-fuzz/corpus/query \
         build-fuzz/corpus/shard_index \
         build-fuzz/corpus/zarr_metadata

./build-fuzz/tests/fuzz/fuzz_json_input \
  build-fuzz/corpus/input \
  tests/fuzz/seeds/input

./build-fuzz/tests/fuzz/fuzz_json_query \
  build-fuzz/corpus/query \
  tests/fuzz/seeds/query

./build-fuzz/tests/fuzz/fuzz_shard_index \
  build-fuzz/corpus/shard_index \
  tests/fuzz/seeds/shard_index

./build-fuzz/tests/fuzz/fuzz_zarr_metadata \
  build-fuzz/corpus/zarr_metadata \
  tests/fuzz/seeds/zarr_metadata
```

Useful libFuzzer flags:

- `-runs=N`              — stop after N executions (default: forever)
- `-max_total_time=SECS` — wall-clock cap
- `-jobs=N -workers=N`   — parallel processes
- `-only_ascii=1`        — restrict to printable ASCII (handy for input
                           fuzzer, useless for the query fuzzer which
                           reads raw byte tape)
- `-print_final_stats=1` — coverage summary at the end

A crash leaves a `crash-<sha1>` file in the cwd; replay with
`./fuzzer crash-<sha1>` to confirm and get the stack trace.

## Coverage

`-fsanitize=fuzzer-no-link` already gives libFuzzer its
edge-coverage feedback, but to *visualize* what got covered (per-line,
per-branch source report) you need source-based coverage on top.
Enable with `DAMACY_FUZZ_COVERAGE=ON`, which adds
`-fprofile-instr-generate -fcoverage-mapping` to every fuzz target.

```fish
cmake --preset fuzz-cov
cmake --build build-fuzz-cov

# Replay the corpus deterministically (no new mutations) so the .profraw
# reflects exactly the seeds + saved findings, not random fuzz bytes.
LLVM_PROFILE_FILE=build-fuzz-cov/cov.profraw \
  ./build-fuzz-cov/tests/fuzz/fuzz_json_input \
  -runs=0 \
  build-fuzz/corpus/input tests/fuzz/seeds/input

llvm-profdata merge -sparse \
  build-fuzz-cov/cov.profraw \
  -o build-fuzz-cov/cov.profdata

llvm-cov show \
  build-fuzz-cov/tests/fuzz/fuzz_json_input \
  -instr-profile=build-fuzz-cov/cov.profdata \
  -format=html \
  -output-dir=build-fuzz-cov/cov-html \
  src/util/json.c

# Or just a per-file summary on stdout:
llvm-cov report \
  build-fuzz-cov/tests/fuzz/fuzz_json_input \
  -instr-profile=build-fuzz-cov/cov.profdata \
  src/util/json.c
```

The coverage build is slower than the plain fuzz build (extra
instrumentation per basic block), so run high-throughput fuzz sessions
under `--preset fuzz` and only switch to `--preset fuzz-cov` when
generating a report.

llvm-profdata and llvm-cov ship with the standard llvm package; pull
them in ad-hoc with `nix shell nixpkgs#llvm` if they aren't already on
PATH.

## Continuous fuzzing

There's a spectrum of how serious people get about this. For damacy at
its current size, the right default is "smoke run on every PR + a
weekly longer run" via GitHub Actions. The heavy options exist if a
real CVE-class bug ever shows up.

- **Local smoke (developer machine).** Run for a few seconds against
  the seed corpus before pushing. Catches regressions where a recent
  change breaks an existing seed. `-runs=20000 -max_total_time=15`.

- **CI smoke (per PR).** A GitHub Actions job that builds the fuzz
  preset, runs each fuzzer for ~60s against the checked-in seeds, and
  fails the PR if any crash file appears. Cheap (fits in a free-tier
  runner) and catches obvious regressions. The corpus does not need to
  persist — every run starts from `tests/fuzz/seeds/`.

- **Scheduled long run (weekly cron in Actions).** Build the fuzz
  preset, restore a previous corpus from a workflow artifact (or from a
  dedicated `corpus` branch via git), run for ~30 minutes per fuzzer,
  upload the new corpus + any crashes as artifacts. This is where you
  actually find new bugs, because the seeds-only run usually plateaus
  in the first few seconds.

- **ClusterFuzzLite.** Google's GitHub Action that wraps both of the
  above into a turnkey workflow with corpus persistence and coverage
  reports. One YAML file. Worth it the moment damacy is open-source and
  has real users.

- **OSS-Fuzz.** Free continuous fuzzing on Google's infrastructure for
  open-source projects. Hours of CPU per day, an automated bug tracker,
  90-day disclosure window for crashes. Overkill until the project is
  upstream of someone else's stack, but the integration is essentially
  the libFuzzer harness we already have plus a Dockerfile.

- **Self-hosted dedicated host.** Run libFuzzer continuously on a spare
  box, persist corpus to a directory, alert on crash files. Works fine
  but you own the operational burden.

If you go the GitHub Actions route, the per-PR smoke job is ~20 lines
of YAML; the scheduled long run is similar plus an
`actions/upload-artifact` for the corpus. Worth adding once the parser
sees enough churn that regressions become a real concern — for now,
running the two harnesses by hand before pushing JSON-parser changes
is enough.

### CI workflow

`.github/workflows/fuzz.yml` runs all four harnesses nightly (07:00
UTC) and on `workflow_dispatch`, 10 minutes per harness on
`ubuntu-latest`. Corpus is cached per harness across runs. On a
crash the matrix entry fails and uploads `crash-*` / `leak-*` /
`oom-*` / `slow-unit-*` plus the corpus snapshot as an artifact named
`<harness>-crashes-<run_id>`. Reproduce locally with
`./build-fuzz/tests/fuzz/<harness> <crash-file>`.

## Adding a seed

Seeds are tiny — a few bytes each — and they're what libFuzzer starts
from on a fresh corpus. When you fix a parser bug that the fuzzer
found (or that came in through some other channel), drop the input
that triggered it under `tests/fuzz/seeds/input/` so future runs
re-exercise it. Same for query-fuzzer regressions in
`tests/fuzz/seeds/query/`.

For the input fuzzer the seeds are JSON files (any valid JSON). For
the query fuzzer the seeds are raw bytes interpreted by the
`decode_parts` tape format in `fuzz_json_query.c` — see that file for
the byte layout. The metadata fuzzer takes either valid zarr v3
documents (which become near-valid after mutation) or raw bytes; the
shard-index fuzzer takes either a valid `(offsets,nbytes)+crc32c` blob
or raw bytes.
