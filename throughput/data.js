window.BENCHMARK_DATA = {
  "lastUpdate": 1779549730376,
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
          "id": "cceba581d712f3829d53d41361b52d30baa8391a",
          "message": "Route OS deps through platform shim (#88)\n\n## Summary\n\nEstablishes a one-rule discipline: anything that might depend on the OS\ngoes through `src/platform/`, even though damacy targets Linux only.\nSweeps the existing violations — direct `pthread_*` in four cache\nmodules, raw `dlopen`/`dlsym` in the GDS loader, `monotonic_ns` time\nmath, and a speculative `#ifdef _WIN32` branch that was never backed by\na real Windows build.\n\nThe shape of the change:\n\n- **New `platform/numa.h` + `numa.posix.c`** — OS-neutral NUMA\nprimitives (`platform_cpu_mask` opaque blob,\navailability/max-node/per-node-mask, thread affinity get/set). API\ndesigned against both POSIX (`cpu_set_t` + `pthread_setaffinity_np`) and\na hypothetical Win32 backing (`GROUP_AFFINITY` +\n`SetThreadGroupAffinity`) so the abstraction is honest, even though\nWin32 isn't implemented.\n- **`src/numa/numa.c` slimmed** — keeps only the CUDA-side policy (GPU →\nhost-NUMA-node resolution via driver attribute + sysfs PCI fallback).\nDrops libnuma dlopen plumbing, `_GNU_SOURCE`, pthread/sched includes,\nand the non-Linux `#else` no-op branch (the platform layer handles\navailability now).\n- **`platform_dl{open,sym,close,error}`** added; `store_fs_gds.c` and\n`platform/numa.posix.c` use them instead of raw libdl.\n- **Four cache modules** (`store_fs`, `store_fs_gds`, `zarr_meta_cache`,\n`zarr_shard_cache`) swap embedded `pthread_mutex_t` for `struct\nplatform_mutex*`. `store_fs_gds` also moves `pthread_once_t` →\n`platform_once` + `platform_call_once`.\n- **`monotonic_ns` → `platform_clock` + `platform_toc`** — five call\nsites in `damacy.c` and `wave_pool.c`. The two-timestamp\n`io_t_start_ns`/`io_t_end_ns` pair on `host_slab_slot` and `damacy_wave`\ncollapses to a single `struct platform_clock` + already-computed `float\nio_ms`, because `platform_toc` returns elapsed AND advances `last_ns` to\n\"now\", so the same clock measures both IO duration (start → end) and\nbind-wait (end → bind) without a second timestamp.\n- **`log/log.c`** uses the new `platform_localtime`; the `_WIN32`\n`localtime_s` branch is gone.\n\n## Where to look first\n\n`src/platform/numa.h` for the abstraction. The Win32 column in the\ndesign table (in the PR-draft conversation) is the actual motivation —\ngo through the file and ask \"would `GROUP_AFFINITY` +\n`SetThreadGroupAffinity` fit here?\". The CPU-mask blob is sized for\nglibc's `cpu_set_t` (128 bytes / 1024 CPUs); same blob holds a Win32\n`GROUP_AFFINITY` array.\n\nThen `src/numa/numa.c` to see what the policy layer looks like once the\nOS plumbing is gone — ~190 lines down from ~410.\n\n## Non-obvious decisions\n\n- **CUDA-side resolution stays out of `platform/`**. The GPU → host-NUMA\nmapping uses `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` +\n`/sys/bus/pci/.../numa_node`. Both are Linux-flavored but conceptually\nthey're CUDA + PCI sysfs, not OS primitives. Keeping them in\n`src/numa/numa.c` keeps `platform/` free of CUDA.\n- **`platform_dlopen` does no path translation**. Callers pass\n`\"libcufile.so.0\"` on POSIX and would pass `\"cufile.dll\"` on Windows.\nTranslating would invite per-library quirks and a name-mangling table;\nopting out is cleaner.\n- **`log` now PUBLIC-links `platform`** because `log.c` calls\n`platform_localtime`. Every consumer of `log` transitively gets\n`platform`. Almost everything already linked both; the few\nstandalone-log tests now pull in `platform` as well.\n- **`io_queue.posix.c` direct pthread use is NOT a violation** —\n`.posix.c` files ARE the POSIX implementation (mirror of\n`platform.posix.c`). The rule applies to `.c` files that don't carry a\nplatform suffix.\n\n## Out of scope (deliberate)\n\n- Tests' direct `<sys/stat.h>` + `unistd.h` for filesystem fixtures\n(mkdir/unlink). Would need a `platform_fs_*` surface that's larger than\nthis PR warrants.\n- Tests' direct `pthread_mutex_t` / `sched_yield`. Easy in isolation but\nbundled here only inflates the diff.\n\n## Test plan\n\n- `ctest -j8` on both `build/` (release) and `build-tsan/` (thread\nsanitizer): 23/23 and 22/22 pass. TSan exercises the cache locks under\ncontention (`test_zarr_cache_threading`) — relevant since four mutex\nimplementations changed.\n- Verified NUMA disabled path still logs once per process: NixOS dev box\nhas no libnuma → `platform_numa_available()` returns 0 → `numa_init`\nlogs \"unavailable\" once.",
          "timestamp": "2026-05-17T21:37:42Z",
          "url": "https://github.com/nclack/damacy/commit/cceba581d712f3829d53d41361b52d30baa8391a"
        },
        "date": 1779115448574,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 6320.33,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 6209.06,
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
          "id": "2b898b1db0ade02d2eba511876ab328e19ccc277",
          "message": "Remove MAX_CHUNK_UNCOMPRESSED_BYTES (#92)\n\nThe compile-time `DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES` (2 MB) was a\nredundant validation guard: no kernel array depended on it, the parser\nalready has its own cap (`DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES`, 16\nMB), and the resolver derives actual sizing from `max_gpu_memory_bytes`.\nA user passing an oversize `max_chunk_uncompressed_bytes` still gets a\nclear `InvalidArgument` from `Pipeline(cfg)` via the existing\n`validate_config` checks, just on a different field — there is no value\nin exposing a private 2 MB ceiling as a public Python constant.\n\nRemoved:\n- C macro + the `<=` check in `validate_config` and the matching clamp\nin `resolve_max_chunk_uncompressed`.\n- The `MAX_CHUNK_UNCOMPRESSED_BYTES` export in `_native.c` and its\n`.pyi` declaration.\n- C test `test_chunk_cap_too_high` and the two Python tests that built a\nconfig with `_native.MAX_CHUNK_UNCOMPRESSED_BYTES + 1`. The Python tests\nwere rewritten around a different INVAL trigger (`n_zarrs_meta_cache=0`)\nso the exception-mapping coverage stays intact.\n\n`DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES` (the 0-default) and the\nparser's `DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES` are unrelated and\nleft alone.\n\nKey file: `src/damacy_config.c`.",
          "timestamp": "2026-05-19T02:44:38Z",
          "url": "https://github.com/nclack/damacy/commit/2b898b1db0ade02d2eba511876ab328e19ccc277"
        },
        "date": 1779201303975,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 6276.39,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 6128.74,
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
          "id": "67de5190f37f1b02098b2b5fafcea22352a1d264",
          "message": "lru: atomic refcount, unlock release path (#111)\n\n## What\n\nMake the per-entry refcount atomic and drop the cache mutex from the\nrelease path. Acquire still takes the mutex — incrementing from zero has\nto be serialized against the eviction scan, which reads refcounts to\ndecide what's safe to evict.\n\n## Why\n\nThe refcount is a per-entry counter. It was under the cache mutex only\nbecause the field lived inside the entry struct, not because the value\nneeded that lock.\n\n## Results\n\nNot a lot of measured performance change on my laptop. The change is\nstill worth landing for cleaner refcount semantics and for the new\ncontention test that exercises the release path under concurrency, but\ndon't expect this workload to get faster.\n\n## Key file\n\n`src/util/lru.c` — refcount, acquire/release, eviction reads.\n\nCloses #108.",
          "timestamp": "2026-05-20T17:14:02Z",
          "url": "https://github.com/nclack/damacy/commit/67de5190f37f1b02098b2b5fafcea22352a1d264"
        },
        "date": 1779373691244,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 6293.96,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 6178.07,
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
          "id": "5672f7e6bbe2a34a7f45529db72e8951c076fb90",
          "message": "Runtime tuning: chunks/wave + substreams/chunk (#112)\n\n## Approach\n\nClose #101 by removing the dead static fallback and replacing the\nad-hoc structural constants it derived from with explicit\n`damacy_tuning` knobs.\n\n`chunk_substreams_upper_bound` (formerly `chunk_zsubs_upper_bound`)\nin `src/wave/wave_pool.c` sizes the per-wave fanout SOA and the\nshared nvcomp zstd decoder scratch. Its `!sp->layout_probed` fallback\nreturned a hardcoded `DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK = 32` — the\nadversarial worst case. But `wave_chunks_eligible` (per-chunk gate,\nruns before `prepare_decode_caps` in `kick_h2d`) rejects any wave\ncontaining an unprobed BLOSC_ZSTD chunk with `DAMACY_INVAL`, so the\nfallback is structurally unreachable. The \"perf\" framing of the\noriginal issue was moot.\n\nThis PR:\n\n- **Turns the implicit gate-vs-sizer contract into an explicit\n  check.** `chunk_substreams_upper_bound` now returns\n  `enum damacy_status`; on unprobed BLOSC it returns `DAMACY_INVAL`\n  with a `log_error(\"gate-vs-sizer contract violated\")` at the\n  caller. A future gate regression now fails loudly instead of\n  silently undersizing the fanout SOA.\n- **Replaces the two compile-time constants**\n(`DAMACY_MAX_CHUNKS_PER_WAVE`,\n`DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK`) with\n`damacy_tuning.max_chunks_per_wave`\n  and `damacy_tuning.max_substreams_per_chunk`. The parser, planner,\n  coalesce, wave_pool, fanout, wave_budget, and meta_cache all thread\n  the effective values through their existing param chains. New\n  `DAMACY_DEFAULT_*` siblings preserve current behavior; `0` in either\n  field resolves to the default. `WAVE_ZSUBS_STRUCTURAL_MAX` becomes\n  a runtime field `wave_pool.max_substreams_per_wave` derived once at\n  init.\n- **Drops the dead substream rename target.** `zsubs` was a\n  contraction that read as zstd-specific; renames to `substreams`\n  everywhere (the noun that matches both BLOSC1 spec language and the\n  nvcomp batched-decode input it actually counts).\n- **Strips machinery wired only to the unreachable branch:** the\n  `_Atomic(uint16_t) observed_max_nblocks_per_chunk` slot, its\n  `atomic_u16_observe_max` CAS-loop helper (`src/util/atomic_max.h`),\n  the meta-cache observer setter, the bump sites in\n  `zarr_meta_cache_layout_set` / `_probe_layout`, and the wiring in\n  `damacy_create`. `zarr/zarr_meta_cache.h` returns to `extern \"C\"`\n  shape (matches main) — the C-only `static_assert` is no longer\n  needed.\n\n## API\n\nTwo new optional fields on `damacy_tuning` (Python `Config`):\n\n- `max_chunks_per_wave: int = 0` — `0` → 512 (current behavior).\n  Clamped to `0xFFFFu` (the 16-bit chunk_idx packing in\n  `d_block_chunk_map`).\n- `max_substreams_per_chunk: int = 0` — `0` → 32 (current behavior).\n  Parser rejects blosc1 layouts above this with `DAMACY_DECODE`.\n\n## Key file\n\n`src/wave/wave_pool.c:355` — `chunk_substreams_upper_bound` (the\ncontract check) and `prepare_decode_caps` (caller).\n\nCloses #101.",
          "timestamp": "2026-05-21T18:39:34Z",
          "url": "https://github.com/nclack/damacy/commit/5672f7e6bbe2a34a7f45529db72e8951c076fb90"
        },
        "date": 1779462312600,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5907.2,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5824.56,
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
          "id": "5672f7e6bbe2a34a7f45529db72e8951c076fb90",
          "message": "Runtime tuning: chunks/wave + substreams/chunk (#112)\n\n## Approach\n\nClose #101 by removing the dead static fallback and replacing the\nad-hoc structural constants it derived from with explicit\n`damacy_tuning` knobs.\n\n`chunk_substreams_upper_bound` (formerly `chunk_zsubs_upper_bound`)\nin `src/wave/wave_pool.c` sizes the per-wave fanout SOA and the\nshared nvcomp zstd decoder scratch. Its `!sp->layout_probed` fallback\nreturned a hardcoded `DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK = 32` — the\nadversarial worst case. But `wave_chunks_eligible` (per-chunk gate,\nruns before `prepare_decode_caps` in `kick_h2d`) rejects any wave\ncontaining an unprobed BLOSC_ZSTD chunk with `DAMACY_INVAL`, so the\nfallback is structurally unreachable. The \"perf\" framing of the\noriginal issue was moot.\n\nThis PR:\n\n- **Turns the implicit gate-vs-sizer contract into an explicit\n  check.** `chunk_substreams_upper_bound` now returns\n  `enum damacy_status`; on unprobed BLOSC it returns `DAMACY_INVAL`\n  with a `log_error(\"gate-vs-sizer contract violated\")` at the\n  caller. A future gate regression now fails loudly instead of\n  silently undersizing the fanout SOA.\n- **Replaces the two compile-time constants**\n(`DAMACY_MAX_CHUNKS_PER_WAVE`,\n`DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK`) with\n`damacy_tuning.max_chunks_per_wave`\n  and `damacy_tuning.max_substreams_per_chunk`. The parser, planner,\n  coalesce, wave_pool, fanout, wave_budget, and meta_cache all thread\n  the effective values through their existing param chains. New\n  `DAMACY_DEFAULT_*` siblings preserve current behavior; `0` in either\n  field resolves to the default. `WAVE_ZSUBS_STRUCTURAL_MAX` becomes\n  a runtime field `wave_pool.max_substreams_per_wave` derived once at\n  init.\n- **Drops the dead substream rename target.** `zsubs` was a\n  contraction that read as zstd-specific; renames to `substreams`\n  everywhere (the noun that matches both BLOSC1 spec language and the\n  nvcomp batched-decode input it actually counts).\n- **Strips machinery wired only to the unreachable branch:** the\n  `_Atomic(uint16_t) observed_max_nblocks_per_chunk` slot, its\n  `atomic_u16_observe_max` CAS-loop helper (`src/util/atomic_max.h`),\n  the meta-cache observer setter, the bump sites in\n  `zarr_meta_cache_layout_set` / `_probe_layout`, and the wiring in\n  `damacy_create`. `zarr/zarr_meta_cache.h` returns to `extern \"C\"`\n  shape (matches main) — the C-only `static_assert` is no longer\n  needed.\n\n## API\n\nTwo new optional fields on `damacy_tuning` (Python `Config`):\n\n- `max_chunks_per_wave: int = 0` — `0` → 512 (current behavior).\n  Clamped to `0xFFFFu` (the 16-bit chunk_idx packing in\n  `d_block_chunk_map`).\n- `max_substreams_per_chunk: int = 0` — `0` → 32 (current behavior).\n  Parser rejects blosc1 layouts above this with `DAMACY_DECODE`.\n\n## Key file\n\n`src/wave/wave_pool.c:355` — `chunk_substreams_upper_bound` (the\ncontract check) and `prepare_decode_caps` (caller).\n\nCloses #101.",
          "timestamp": "2026-05-21T18:39:34Z",
          "url": "https://github.com/nclack/damacy/commit/5672f7e6bbe2a34a7f45529db72e8951c076fb90"
        },
        "date": 1779549729123,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "damacy/default/throughput",
            "value": 5915.39,
            "unit": "MB/s"
          },
          {
            "name": "damacy/mixed/throughput",
            "value": 5841.46,
            "unit": "MB/s"
          }
        ]
      }
    ]
  }
}