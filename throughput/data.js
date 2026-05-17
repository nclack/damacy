window.BENCHMARK_DATA = {
  "lastUpdate": 1779030947160,
  "repoUrl": "https://github.com/nclack/damacy",
  "entries": {
    "damacy throughput": [
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "21be666b3ebd18d3813b21f503184801338f467c",
          "message": "bench: daily cron + manual dispatch only (#13)\n\nDrops the `push: branches: [main]` trigger. The full bench takes ~30\nmin, and per-merge data points are noisy enough that a daily cadence\ngives more useful trend data without serializing every merge behind a\nhalf-hour GPU job.\n\n- `schedule:` cron `'23 7 * * *'` (07:23 UTC, off the top of the hour\n  to dodge GitHub's hourly cron surge)\n- `workflow_dispatch:` retained for ad hoc bench against any ref\n- README + bench/README-tracking.md updated to match\n\nTo compare a specific PR/commit, dispatch the workflow against that\nref from the Actions tab.",
          "timestamp": "2026-05-07T23:04:59Z",
          "url": "https://github.com/nclack/damacy/commit/21be666b3ebd18d3813b21f503184801338f467c"
        },
        "date": 1778195709856,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 2846.67,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 2645.65,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "04362abf8b24bcb9fbe5718e301268531518e69a",
          "message": "python: pytest suite for _native (#17)\n\nAdds a pytest target for `damacy._native` so the bindings are exercised\nin CI alongside the C ctest suite. Required surfacing the runtime caps\n(`max_chunk_uncompressed_bytes`, `max_gpu_memory_bytes`,\n`max_bytes_per_element`) and the `gpu_bytes_committed` stat through the\nbinding.\n\n- `python/CMakeLists.txt`: register `python_pytest` ctest target gated\non `import pytest`; sets `WRITE_ZARR_SCRIPT` so the fixture can locate\nthe C-side zarr writer.\n- `CMakeLists.txt`: hoist `include(CTest)` above\n`add_subdirectory(python)` so the new target sees `BUILD_TESTING`.\n- `python/damacy/_api.c`: parse the three caps kwargs\n(`max_chunk_uncompressed_bytes` required, the other two optional); emit\n`gpu_bytes_committed` from `stats()`.\n- `python/damacy/_native.c`: export `MAX_CHUNK_UNCOMPRESSED_BYTES`\nconstant.\n- `src/damacy.{h,c}`: add `gpu_bytes_committed` to `damacy_stats` and\npopulate it in `damacy_stats_get`.\n- `Dockerfile`: install pytest in the venv; exclude `python_pytest` from\nthe build-time ctest run (needs the editable install + GPU).\n\nCloses #10.\n\n## Tests\n\n- `python/tests/conftest.py`: `tiny_zarr` / `tiny_zarr_u32` fixtures\nthat shell out to `tests/write_zarr.py`.\n- `python/tests/test_damacy.py` (15 tests): module constants/version,\nmissing/oversize `max_chunk_uncompressed_bytes`, `max_gpu_memory_bytes`\ntoo small, dtype string + int forms, unknown dtype, push/pop/release\nend-to-end, unknown-uri, dtype mismatch, oversize-chunk surfacing at\npop, `gpu_bytes_committed` present + grows after first pop, log-sink\nsmoke.",
          "timestamp": "2026-05-08T02:53:01Z",
          "url": "https://github.com/nclack/damacy/commit/04362abf8b24bcb9fbe5718e301268531518e69a"
        },
        "date": 1778209402956,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5421.73,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5070.71,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "87ead67d474a7fbde3fef672e0a82d39fe799c79",
          "message": "python: nice api, doctests, mkdocs site (#19)\n\n- Python API: frozen-dataclass `Config`, typed exception hierarchy,\ncontext-managed `Batch`, `py.typed` + `_native.pyi`\n- Doctests + 33-test pytest suite \n- mkdocs+material+mkdocstrings \n- README: docs badge, Quick start, torchrun 8-GPU example\n\nCloses #18.",
          "timestamp": "2026-05-08T17:54:32Z",
          "url": "https://github.com/nclack/damacy/commit/87ead67d474a7fbde3fef672e0a82d39fe799c79"
        },
        "date": 1778263029300,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5729.92,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5333.08,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "fda3292956ce03f0db173a93580c9a1d57d6f328",
          "message": "ctx_guard at every public API entry (#25)\n\nPush the retained primary CUcontext per public API call instead of for\nthe pipeline lifetime, restoring the caller's thread state on every\nreturn.\n\nCloses #21.",
          "timestamp": "2026-05-09T00:24:11Z",
          "url": "https://github.com/nclack/damacy/commit/fda3292956ce03f0db173a93580c9a1d57d6f328"
        },
        "date": 1778286963404,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5670.18,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5302.72,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "fda3292956ce03f0db173a93580c9a1d57d6f328",
          "message": "ctx_guard at every public API entry (#25)\n\nPush the retained primary CUcontext per public API call instead of for\nthe pipeline lifetime, restoring the caller's thread state on every\nreturn.\n\nCloses #21.",
          "timestamp": "2026-05-09T00:24:11Z",
          "url": "https://github.com/nclack/damacy/commit/fda3292956ce03f0db173a93580c9a1d57d6f328"
        },
        "date": 1778331956573,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5798.34,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5481.12,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "id": "c1e3925a7a8f45c37ccb94d706bc4d804f3bf0ae",
          "message": "remove old docs",
          "timestamp": "2026-05-09T18:07:46Z",
          "url": "https://github.com/nclack/damacy/commit/c1e3925a7a8f45c37ccb94d706bc4d804f3bf0ae"
        },
        "date": 1778425575196,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5852.35,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5463.94,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "id": "c1e3925a7a8f45c37ccb94d706bc4d804f3bf0ae",
          "message": "remove old docs",
          "timestamp": "2026-05-09T18:07:46Z",
          "url": "https://github.com/nclack/damacy/commit/c1e3925a7a8f45c37ccb94d706bc4d804f3bf0ae"
        },
        "date": 1778506864778,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 6232.39,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5765.74,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "91b86c0df9f0b4c111d5ea8f03113ce41bef8ab0",
          "message": "factor damacy.c + _api.c into focused modules (#29)\n\nRefactor to isolate modules and reduce large files.",
          "timestamp": "2026-05-12T14:24:27Z",
          "url": "https://github.com/nclack/damacy/commit/91b86c0df9f0b4c111d5ea8f03113ce41bef8ab0"
        },
        "date": 1778682709776,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 6219.52,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5755.59,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "6fc638d03552356f86f3c15a8c9abd79af0da221",
          "message": "reserve gpu budget for batch pool (#43)",
          "timestamp": "2026-05-13T17:15:59Z",
          "url": "https://github.com/nclack/damacy/commit/6fc638d03552356f86f3c15a8c9abd79af0da221"
        },
        "date": 1778692779649,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 6125.51,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5676.11,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "6fc638d03552356f86f3c15a8c9abd79af0da221",
          "message": "reserve gpu budget for batch pool (#43)",
          "timestamp": "2026-05-13T17:15:59Z",
          "url": "https://github.com/nclack/damacy/commit/6fc638d03552356f86f3c15a8c9abd79af0da221"
        },
        "date": 1778769861684,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 6209.1,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5745.06,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "6aa6fb855c9198dffd96640fa7e9936cb507c8a4",
          "message": "reorganize wave/ + gpu_budget (#47)\n\nImplements the recommended sequencing from the #45 review.\n\nCloses nothing standalone; follow-up to #45.",
          "timestamp": "2026-05-14T21:26:06Z",
          "url": "https://github.com/nclack/damacy/commit/6aa6fb855c9198dffd96640fa7e9936cb507c8a4"
        },
        "date": 1778794185503,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5650.44,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5605.88,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "7a3cd53aa9ecc852b85ebebff567002492ebc1b8",
          "message": "deferred release via cuda event (#53)\n\nCloses #31\nCloses #35",
          "timestamp": "2026-05-15T02:14:30Z",
          "url": "https://github.com/nclack/damacy/commit/7a3cd53aa9ecc852b85ebebff567002492ebc1b8"
        },
        "date": 1778813274173,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5665.19,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5583.81,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "31a723316a30fec4664af320c710ef68602dc2c8",
          "message": "log: clarify numa enabled, quiet grow events (#57)",
          "timestamp": "2026-05-15T03:08:13Z",
          "url": "https://github.com/nclack/damacy/commit/31a723316a30fec4664af320c710ef68602dc2c8"
        },
        "date": 1778854189730,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5803.42,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5690.27,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ab5c77f5b7aa17082a5aa0dbcda90184a2376f23",
          "message": "blosc: gpu parse + cufile reader (#58)\n\nGPU blosc1 parse + opt-in cuFile reader.\n\n- **chunk-layout probe** — first non-fill emit reads the 16-byte blosc1\nheader, caches `nblocks` on the meta entry; tightens `need_zsubs` to\nactual.\n- **GPU parse** — `blosc1_parse.cu` on `stream_h2d`. Kernel A: per-chunk\nclassify (memcpyed vs codec'd) + shuffle fields. Kernel B: per-block\nfan-out (raw → memcpy_op, codec → zstd_fan). Host parser is gone.\n- **GDS store** — `store_fs_gds.c` is an independent store; composes a\nhost `store_fs` for stat/submit/map and overrides `submit_dev` with\n`cuFileReadAsync`. `damacy.c` picks the constructor.",
          "timestamp": "2026-05-16T05:56:20Z",
          "url": "https://github.com/nclack/damacy/commit/ab5c77f5b7aa17082a5aa0dbcda90184a2376f23"
        },
        "date": 1778911222573,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 6091.72,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5964.07,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "e121786b67472c91426bb2cebe1ce01353b753ad",
          "message": "README: GDS build flag, drop use_gpu_blosc_parse (#60)\n\nREADME still referenced `use_gpu_blosc_parse = 1` (removed when the host\nparser was dropped) and didn't mention the new `-DDAMACY_ENABLE_GDS=ON`\nbuild-time requirement.",
          "timestamp": "2026-05-16T06:06:49Z",
          "url": "https://github.com/nclack/damacy/commit/e121786b67472c91426bb2cebe1ce01353b753ad"
        },
        "date": 1778940928118,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5773.05,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5686.39,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ef093f8f9811c22a03c550e73df84941b950ce84",
          "message": "Intern shard paths and sample URIs (#76)\n\n## Glossary\n\n- `path_intern` — new utility (`src/util/path_intern.{h,c}`). `acquire`\nreturns a stable `const char*` with refcount++; `release` decrements\n(evict at zero); `reset` drops everything ignoring refcounts; `free`\ntears down. Equal strings always map to the same pointer.\n- `read_op` — page-aligned IO record emitted by the planner; carries a\n`shard_path` and an offset/length.\n- `shard_path` — a per-shard filesystem path derived from URI + shard\ncoordinates inside the planner via `zarr_shard_path_build`.\n- `damacy_sample.uri` — caller-supplied zarr array identifier passed to\n`damacy_push`.\n- `damacy_batch_slot` — one of two batch slots in the batch pool; owns\n`read_ops` until the user releases the batch.\n\n## Summary\n\nReplaces the 224-byte `DAMACY_MAX_PATH` inline buffer on `read_op` with\nan interned `const char*`, and collapses the per-sample `strdup` of\n`damacy_sample.uri` in the lookahead onto the same module. Result:\n`coalesce.c` checks same-shard fusion with `==` instead of `strcmp`;\nduplicate URIs across samples share one allocation; the 224-byte cap\ngoes away entirely.\n\nStart at **`src/util/path_intern.h`** for the contract, then\n`src/damacy.c` (search `path_intern_reset` and `&self->uris`) to see how\neach consumer plugs in.\n\nThe non-obvious choice is two intern tables, not one. `shard_path` uses\na **per-batch-slot table** reset on every BATCH_FREE→PLANNING transition\n— bound: distinct paths in one batch. URIs use a **damacy-owned\nrefcounted table** with `acquire` at `lookahead_push` and `release` at\n`sample_slot_clear` — bound: `lookahead.cap × distinct URIs in flight`.\nSame API supports both patterns; the slot variant never calls `release`\nand the URI variant never calls `reset`. This bounds the working set in\nboth directions without LRU machinery.\n\nOne subtlety in the module itself: open-addressed deletion requires\nbackward-shift to avoid orphaning probe chains. See `slot_evict` in\n`path_intern.c` — the load-bearing invariant is \"any entry whose ideal\nhome is at-or-before the gap migrates into it.\" A naive null-the-bucket\nwould silently lose later collisions.\n\n## Test plan\n\n- `tests/test_path_intern.c` covers the two invariants not reachable\nthrough coalesce/planner tests: rehash stability across 200 inserts (≥3\ngrows from cap=16), and backward-shift correctness with interleaved\nreleases on a 40-entry table.\n- `tests/test_chunk_layout.c::test_planner_populates_layout_blosc_zstd`\nnow drives two consecutive `planner_plan` calls with an explicit\n`path_intern_reset` between them, mirroring what `damacy.c::plan_run`\ndoes on real batches.\n- `tests/test_lookahead.c::test_pointer_identity` confirms duplicate\nURIs collapse to one pointer in the ring; `test_destroy_frees` asserts\n`uris.n == 0` after `lookahead_destroy` on a partially-full ring (the\nrefcount-release-on-destroy path).\n\nCloses #68",
          "timestamp": "2026-05-17T05:36:55Z",
          "url": "https://github.com/nclack/damacy/commit/ef093f8f9811c22a03c550e73df84941b950ce84"
        },
        "date": 1778996547727,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5730.67,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5667.8,
            "unit": "MB/s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Nathan Clack",
            "username": "nclack",
            "email": "nclack@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "ef093f8f9811c22a03c550e73df84941b950ce84",
          "message": "Intern shard paths and sample URIs (#76)\n\n## Glossary\n\n- `path_intern` — new utility (`src/util/path_intern.{h,c}`). `acquire`\nreturns a stable `const char*` with refcount++; `release` decrements\n(evict at zero); `reset` drops everything ignoring refcounts; `free`\ntears down. Equal strings always map to the same pointer.\n- `read_op` — page-aligned IO record emitted by the planner; carries a\n`shard_path` and an offset/length.\n- `shard_path` — a per-shard filesystem path derived from URI + shard\ncoordinates inside the planner via `zarr_shard_path_build`.\n- `damacy_sample.uri` — caller-supplied zarr array identifier passed to\n`damacy_push`.\n- `damacy_batch_slot` — one of two batch slots in the batch pool; owns\n`read_ops` until the user releases the batch.\n\n## Summary\n\nReplaces the 224-byte `DAMACY_MAX_PATH` inline buffer on `read_op` with\nan interned `const char*`, and collapses the per-sample `strdup` of\n`damacy_sample.uri` in the lookahead onto the same module. Result:\n`coalesce.c` checks same-shard fusion with `==` instead of `strcmp`;\nduplicate URIs across samples share one allocation; the 224-byte cap\ngoes away entirely.\n\nStart at **`src/util/path_intern.h`** for the contract, then\n`src/damacy.c` (search `path_intern_reset` and `&self->uris`) to see how\neach consumer plugs in.\n\nThe non-obvious choice is two intern tables, not one. `shard_path` uses\na **per-batch-slot table** reset on every BATCH_FREE→PLANNING transition\n— bound: distinct paths in one batch. URIs use a **damacy-owned\nrefcounted table** with `acquire` at `lookahead_push` and `release` at\n`sample_slot_clear` — bound: `lookahead.cap × distinct URIs in flight`.\nSame API supports both patterns; the slot variant never calls `release`\nand the URI variant never calls `reset`. This bounds the working set in\nboth directions without LRU machinery.\n\nOne subtlety in the module itself: open-addressed deletion requires\nbackward-shift to avoid orphaning probe chains. See `slot_evict` in\n`path_intern.c` — the load-bearing invariant is \"any entry whose ideal\nhome is at-or-before the gap migrates into it.\" A naive null-the-bucket\nwould silently lose later collisions.\n\n## Test plan\n\n- `tests/test_path_intern.c` covers the two invariants not reachable\nthrough coalesce/planner tests: rehash stability across 200 inserts (≥3\ngrows from cap=16), and backward-shift correctness with interleaved\nreleases on a 40-entry table.\n- `tests/test_chunk_layout.c::test_planner_populates_layout_blosc_zstd`\nnow drives two consecutive `planner_plan` calls with an explicit\n`path_intern_reset` between them, mirroring what `damacy.c::plan_run`\ndoes on real batches.\n- `tests/test_lookahead.c::test_pointer_identity` confirms duplicate\nURIs collapse to one pointer in the ring; `test_destroy_frees` asserts\n`uris.n == 0` after `lookahead_destroy` on a partially-full ring (the\nrefcount-release-on-destroy path).\n\nCloses #68",
          "timestamp": "2026-05-17T05:36:55Z",
          "url": "https://github.com/nclack/damacy/commit/ef093f8f9811c22a03c550e73df84941b950ce84"
        },
        "date": 1779030945786,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5898.8,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5834.56,
            "unit": "MB/s"
          }
        ]
      }
    ]
  }
}