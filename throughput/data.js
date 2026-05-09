window.BENCHMARK_DATA = {
  "lastUpdate": 1778331957285,
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
      }
    ]
  }
}