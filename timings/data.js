window.BENCHMARK_DATA = {
  "lastUpdate": 1781262679420,
  "repoUrl": "https://github.com/nclack/damacy",
  "entries": {
    "damacy timings": [
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
        "date": 1778195707197,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 358.552,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 629.104,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 18859.6,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.28774,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.23434,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 14.3987,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.63477,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.296179,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.zstd.ms_avg",
            "value": 14.0939,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.post.ms_avg",
            "value": 0.00354392,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 294.18,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 398.096,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 20292.6,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.32869,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.34945,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 15.6397,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.64746,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.310008,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.zstd.ms_avg",
            "value": 14.3751,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.post.ms_avg",
            "value": 0.949436,
            "unit": "ms"
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
        "date": 1778209398967,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 139.183,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 399.992,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9902.21,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.44786,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.03911,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 14.228,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.65045,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.552126,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.zstd.ms_avg",
            "value": 13.6673,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.post.ms_avg",
            "value": 0.00350357,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 149.704,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 241.6,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 10587.7,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.46895,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.15799,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 15.371,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.65688,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.577779,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.zstd.ms_avg",
            "value": 13.8636,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.post.ms_avg",
            "value": 0.92438,
            "unit": "ms"
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
        "date": 1778263025155,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 171.074,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 562.518,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9369.6,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.37847,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.03033,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 14.0811,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.68203,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0455955,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.zstd.ms_avg",
            "value": 14.0256,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.post.ms_avg",
            "value": 0.00447369,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 144.959,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 531.753,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 10066.8,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.49188,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.14569,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 15.1973,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.67581,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0807012,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.zstd.ms_avg",
            "value": 14.179,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.post.ms_avg",
            "value": 0.932125,
            "unit": "ms"
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
        "date": 1778286959634,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 145.728,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 359.067,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9468.32,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.22741,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.03335,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 14.088,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.84839,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0124051,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.zstd.ms_avg",
            "value": 19.9246,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.post.ms_avg",
            "value": 0.00858477,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 144.258,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 217.351,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 10124.5,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.38411,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.15045,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 15.1146,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.86292,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0701765,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.zstd.ms_avg",
            "value": 20.9406,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.post.ms_avg",
            "value": 0.936173,
            "unit": "ms"
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
        "date": 1778331953068,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 131.384,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 377.003,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9259.04,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.20669,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.04711,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 13.7084,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.80528,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0142385,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.zstd.ms_avg",
            "value": 19.1869,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.post.ms_avg",
            "value": 0.00848038,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 128.384,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 216.283,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9794.9,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.20024,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.15921,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 14.6659,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.80033,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0767814,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.zstd.ms_avg",
            "value": 20.1793,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.post.ms_avg",
            "value": 0.927328,
            "unit": "ms"
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
        "date": 1778425571771,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 138.029,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 351.004,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9173.6,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.15653,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.04591,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 13.6193,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.78936,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0121475,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.zstd.ms_avg",
            "value": 19.0388,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.post.ms_avg",
            "value": 0.00858041,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 134.294,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 221.624,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9825.72,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.38328,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.16664,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 14.6455,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.80356,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0823704,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.zstd.ms_avg",
            "value": 19.9747,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.post.ms_avg",
            "value": 0.926474,
            "unit": "ms"
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
        "date": 1778506860087,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 216.958,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 513.004,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8614.2,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.60954,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.05856,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 12.6417,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.71649,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0195526,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.zstd.ms_avg",
            "value": 16.699,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.post.ms_avg",
            "value": 0.009208,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 214.374,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 228.72,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9311.4,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.61201,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.16492,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 13.8159,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.70067,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.100395,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.zstd.ms_avg",
            "value": 17.9464,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.post.ms_avg",
            "value": 1.05024,
            "unit": "ms"
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
        "date": 1778682705597,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 220.074,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 524.928,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8632.04,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.65275,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.07619,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 12.6771,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.71935,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0186362,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.zstd.ms_avg",
            "value": 16.7092,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.post.ms_avg",
            "value": 0.00899298,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 228.033,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 247.416,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9327.82,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.65338,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.17851,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 13.838,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.70354,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0990329,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.zstd.ms_avg",
            "value": 17.9487,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.post.ms_avg",
            "value": 1.05175,
            "unit": "ms"
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
        "date": 1778692776242,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 141.478,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 410.838,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8764.51,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.53461,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.07815,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 13.0518,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.62258,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0174187,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 126.04,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 259.653,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9458.44,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.57964,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.2015,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 14.1746,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.6127,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0731956,
            "unit": "ms"
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
        "date": 1778769858647,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 124.862,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 365.358,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8646.51,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.64131,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.12119,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.ms_avg",
            "value": 12.8531,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.58518,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0189775,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 124.749,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 252.48,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9344.91,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.74005,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.23592,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.ms_avg",
            "value": 13.9945,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.57716,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0705636,
            "unit": "ms"
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
        "date": 1778794181213,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 78.7684,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 648.091,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9501.39,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.09809,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.04684,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.65444,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.016541,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 79.3757,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 237.885,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9576.93,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.19116,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.16243,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.81032,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0743274,
            "unit": "ms"
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
        "date": 1778813270189,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 97.4601,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1060.26,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9476.67,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.23543,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.04608,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.75204,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0158181,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 95.7468,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 222.727,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9614.78,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.31484,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.16102,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.97498,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0774709,
            "unit": "ms"
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
        "date": 1778854186503,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 79.5541,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 923.092,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9250.94,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.11355,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.04314,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.71036,
            "unit": "ms"
          },
          {
            "name": "damacy/default/decompress.parse.ms_avg",
            "value": 0.0177785,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 77.6219,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 229.806,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9434.89,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.29346,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.16081,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.93798,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/decompress.parse.ms_avg",
            "value": 0.0752841,
            "unit": "ms"
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
        "date": 1778911217918,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 82.4301,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1100.39,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8813.13,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.85435,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.08581,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.6788,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 75.9081,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 256.947,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9001.75,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.78764,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.18046,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.8751,
            "unit": "ms"
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
        "date": 1778940924244,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 107.098,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1513.02,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9299.61,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 4.3343,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 6.03545,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.71798,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 105.784,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 250.105,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9441.33,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 4.34065,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 6.14764,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.92679,
            "unit": "ms"
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
        "date": 1778996543678,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 84.066,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1037.84,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9368.38,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.00005,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.87477,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.73766,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 84.5073,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 218.337,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9472.31,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.03546,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.94877,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.95031,
            "unit": "ms"
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
        "date": 1779030940623,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 109.35,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1625.05,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9101.36,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 2.85182,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.85369,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.68005,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 106.468,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 241.039,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9201.57,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.03997,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.9159,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.89201,
            "unit": "ms"
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
        "date": 1779115444667,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 109.65,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1610.39,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8494.35,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 2.97561,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.84923,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.61857,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 111.168,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 239.872,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8646.57,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.08375,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.91658,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.8075,
            "unit": "ms"
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
        "date": 1779201299105,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 91.737,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1156.86,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8553.81,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.00191,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.85077,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.63593,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 92.0802,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 219.071,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8759.89,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.06062,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.92402,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.82911,
            "unit": "ms"
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
        "date": 1779373687246,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 70.3021,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1002.4,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8529.94,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.08227,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.86701,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.62576,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 70.6754,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 229.638,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8689.95,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.03879,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.94194,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.8139,
            "unit": "ms"
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
        "date": 1779462308857,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 85.1204,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1010.68,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9088.41,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.11161,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.85218,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.68412,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 82.4072,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 225.983,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9217.37,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.0011,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.92157,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.89981,
            "unit": "ms"
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
        "date": 1779549726145,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 91.7519,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 993.324,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9075.83,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.031,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.83712,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.67618,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 89.5666,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 223.531,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9190.69,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 2.96116,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.90908,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.8894,
            "unit": "ms"
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
        "date": 1779644130368,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 77.7766,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1009.54,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8501.49,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.10811,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.86281,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.62736,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 79.489,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 221.485,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8679.95,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.00842,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.9285,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.82366,
            "unit": "ms"
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
        "date": 1779719591355,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 74.0841,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 988.79,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9052.19,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.0044,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.85955,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.67664,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 72.8253,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 205.857,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9193.55,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 2.98475,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.92815,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.88926,
            "unit": "ms"
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
        "date": 1779891951302,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 82.1194,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1075.14,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8615.3,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.28719,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.86079,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.65014,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 81.1325,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 403.784,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8792.11,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.28895,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.93187,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.84736,
            "unit": "ms"
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
        "date": 1780064457738,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 83.6967,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1100.98,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8599.25,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.0232,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.84596,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.67862,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 70.6674,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 240.247,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8745.94,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.13903,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.9213,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.872,
            "unit": "ms"
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
        "date": 1780149365436,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 97.5638,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1235.68,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9108.27,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.01801,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.86244,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.68792,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 80.0292,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 226.976,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9255.28,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 2.86372,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.9266,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.90176,
            "unit": "ms"
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
        "date": 1780325114440,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 114.848,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1630.01,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8717.02,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.22582,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.87786,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.65716,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 112.595,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 238.684,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8802.69,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.12033,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.94242,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.83818,
            "unit": "ms"
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
        "date": 1780403568414,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 99.9615,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1269.47,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9139.92,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 2.91413,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.85742,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.69089,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 72.1534,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 202.863,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9261.07,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 2.94103,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.92676,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.90307,
            "unit": "ms"
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
        "date": 1780501915840,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 110.323,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1586.21,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8486.26,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.11463,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.8529,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.61708,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 110.437,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 243.111,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8660.95,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.02922,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.91355,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.80453,
            "unit": "ms"
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
        "date": 1780583875783,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 82.0187,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1089.76,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8520.56,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.06961,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.86275,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.62253,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 74.1255,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 224.629,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8679.84,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.11819,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.93774,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.81173,
            "unit": "ms"
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
        "date": 1780666263096,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 118.907,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1638.64,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8592.97,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.09452,
            "unit": "ms"
          },
          {
            "name": "damacy/default/h2d.ms_avg",
            "value": 3.85075,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.63318,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 117.941,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 246.025,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8763.58,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.28284,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/h2d.ms_avg",
            "value": 3.93719,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.82636,
            "unit": "ms"
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
          "id": "7dcc03f7b7cd4dea194a552f3ec76f6db3fc8c4d",
          "message": "damacy rewrite: metadata prefetch (#120)\n\n## Summary\n\n- Add the metadata prefetch pipeline: `prefetch_cache`, array metadata\nfetcher, shard-index fetcher, chunk-layout fetcher, and the prefetcher\nworker.\n- Split the old monolithic `damacy.c` orchestrator into lifecycle, push,\nplan, pop, and scheduler modules.\n- Make the prefetcher the producer for planning, so samples are planned\nonly after metadata/shard/layout prefetch has reached a terminal state.\n- Rewire the planner to consume prefetch handles directly and delete the\nlegacy synchronous `zarr_meta_cache` / `zarr_shard_cache` path.\n- Preserve sparse-zarr behavior: missing shard files become fill chunks,\nwhile IO, permission, malformed-shard, and decode errors still fail the\nsample.\n- Update public config/stats, Python bindings, docs, and benchmark\nschema for the new `array_meta`, `shard_index`, and `chunk_layout`\ncaches.\n\n## Reviewer Notes\n\n- Start with `src/prefetch/prefetcher.c`,\n`src/prefetch/prefetch_cache.c`, `src/damacy_plan.c`,\n`src/damacy_scheduler.c`, and `src/planner/planner.c`.\n- Store-derived validation is now asynchronous. Missing URIs,\nunsupported source dtypes, per-array rank mismatch, and decode failures\nsurface from `pop()`, not `push()`.\n- Prefetch slots preserve push order with `admit_seq`, even if metadata\nrequests complete out of order.\n- `store_stat` now distinguishes `NOT_FOUND` from other stat failures so\nsparse data does not mask IO/permission errors.\n- Chunk-layout probing uses a ready shard touched by the sample rather\nthan assuming the origin shard exists.\n\n## Tests\n\n- `cmake --build build --target damacy test_chunk_layout_cache\ntest_prefetcher test_planner`\n- `env UV_CACHE_DIR=/tmp/uv-cache timeout 90s ctest --test-dir build -R\n'test_(planner|chunk_layout_cache|prefetcher)$' --output-on-failure`\n\nAdditional coverage in this branch includes unit/integration tests for\n`prefetch_cache`, array metadata fetch, shard index fetch, chunk layout\nfetch, prefetcher ordering/readiness/error paths, sample-shard\niteration, and planner handle consumption.",
          "timestamp": "2026-06-05T16:52:49Z",
          "url": "https://github.com/nclack/damacy/commit/7dcc03f7b7cd4dea194a552f3ec76f6db3fc8c4d"
        },
        "date": 1780682269613,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 72.456,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1013.67,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9340.85,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 2.83435,
            "unit": "ms"
          },
          {
            "name": "damacy/default/input_transfer.ms_avg",
            "value": 3.85244,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.73594,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 75.946,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 237.83,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9388.55,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 2.91779,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/input_transfer.ms_avg",
            "value": 3.92597,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.92952,
            "unit": "ms"
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
          "id": "7dcc03f7b7cd4dea194a552f3ec76f6db3fc8c4d",
          "message": "damacy rewrite: metadata prefetch (#120)\n\n## Summary\n\n- Add the metadata prefetch pipeline: `prefetch_cache`, array metadata\nfetcher, shard-index fetcher, chunk-layout fetcher, and the prefetcher\nworker.\n- Split the old monolithic `damacy.c` orchestrator into lifecycle, push,\nplan, pop, and scheduler modules.\n- Make the prefetcher the producer for planning, so samples are planned\nonly after metadata/shard/layout prefetch has reached a terminal state.\n- Rewire the planner to consume prefetch handles directly and delete the\nlegacy synchronous `zarr_meta_cache` / `zarr_shard_cache` path.\n- Preserve sparse-zarr behavior: missing shard files become fill chunks,\nwhile IO, permission, malformed-shard, and decode errors still fail the\nsample.\n- Update public config/stats, Python bindings, docs, and benchmark\nschema for the new `array_meta`, `shard_index`, and `chunk_layout`\ncaches.\n\n## Reviewer Notes\n\n- Start with `src/prefetch/prefetcher.c`,\n`src/prefetch/prefetch_cache.c`, `src/damacy_plan.c`,\n`src/damacy_scheduler.c`, and `src/planner/planner.c`.\n- Store-derived validation is now asynchronous. Missing URIs,\nunsupported source dtypes, per-array rank mismatch, and decode failures\nsurface from `pop()`, not `push()`.\n- Prefetch slots preserve push order with `admit_seq`, even if metadata\nrequests complete out of order.\n- `store_stat` now distinguishes `NOT_FOUND` from other stat failures so\nsparse data does not mask IO/permission errors.\n- Chunk-layout probing uses a ready shard touched by the sample rather\nthan assuming the origin shard exists.\n\n## Tests\n\n- `cmake --build build --target damacy test_chunk_layout_cache\ntest_prefetcher test_planner`\n- `env UV_CACHE_DIR=/tmp/uv-cache timeout 90s ctest --test-dir build -R\n'test_(planner|chunk_layout_cache|prefetcher)$' --output-on-failure`\n\nAdditional coverage in this branch includes unit/integration tests for\n`prefetch_cache`, array metadata fetch, shard index fetch, chunk layout\nfetch, prefetcher ordering/readiness/error paths, sample-shard\niteration, and planner handle consumption.",
          "timestamp": "2026-06-05T16:52:49Z",
          "url": "https://github.com/nclack/damacy/commit/7dcc03f7b7cd4dea194a552f3ec76f6db3fc8c4d"
        },
        "date": 1780777006154,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 73.2669,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1038.39,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8617.63,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.06639,
            "unit": "ms"
          },
          {
            "name": "damacy/default/input_transfer.ms_avg",
            "value": 3.89174,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.68289,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 73.7987,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 240.573,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8747.07,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.14347,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/input_transfer.ms_avg",
            "value": 3.96042,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.87736,
            "unit": "ms"
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
          "id": "7dcc03f7b7cd4dea194a552f3ec76f6db3fc8c4d",
          "message": "damacy rewrite: metadata prefetch (#120)\n\n## Summary\n\n- Add the metadata prefetch pipeline: `prefetch_cache`, array metadata\nfetcher, shard-index fetcher, chunk-layout fetcher, and the prefetcher\nworker.\n- Split the old monolithic `damacy.c` orchestrator into lifecycle, push,\nplan, pop, and scheduler modules.\n- Make the prefetcher the producer for planning, so samples are planned\nonly after metadata/shard/layout prefetch has reached a terminal state.\n- Rewire the planner to consume prefetch handles directly and delete the\nlegacy synchronous `zarr_meta_cache` / `zarr_shard_cache` path.\n- Preserve sparse-zarr behavior: missing shard files become fill chunks,\nwhile IO, permission, malformed-shard, and decode errors still fail the\nsample.\n- Update public config/stats, Python bindings, docs, and benchmark\nschema for the new `array_meta`, `shard_index`, and `chunk_layout`\ncaches.\n\n## Reviewer Notes\n\n- Start with `src/prefetch/prefetcher.c`,\n`src/prefetch/prefetch_cache.c`, `src/damacy_plan.c`,\n`src/damacy_scheduler.c`, and `src/planner/planner.c`.\n- Store-derived validation is now asynchronous. Missing URIs,\nunsupported source dtypes, per-array rank mismatch, and decode failures\nsurface from `pop()`, not `push()`.\n- Prefetch slots preserve push order with `admit_seq`, even if metadata\nrequests complete out of order.\n- `store_stat` now distinguishes `NOT_FOUND` from other stat failures so\nsparse data does not mask IO/permission errors.\n- Chunk-layout probing uses a ready shard touched by the sample rather\nthan assuming the origin shard exists.\n\n## Tests\n\n- `cmake --build build --target damacy test_chunk_layout_cache\ntest_prefetcher test_planner`\n- `env UV_CACHE_DIR=/tmp/uv-cache timeout 90s ctest --test-dir build -R\n'test_(planner|chunk_layout_cache|prefetcher)$' --output-on-failure`\n\nAdditional coverage in this branch includes unit/integration tests for\n`prefetch_cache`, array metadata fetch, shard index fetch, chunk layout\nfetch, prefetcher ordering/readiness/error paths, sample-shard\niteration, and planner handle consumption.",
          "timestamp": "2026-06-05T16:52:49Z",
          "url": "https://github.com/nclack/damacy/commit/7dcc03f7b7cd4dea194a552f3ec76f6db3fc8c4d"
        },
        "date": 1780936298042,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 110.93,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1390.7,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 8678.87,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.29577,
            "unit": "ms"
          },
          {
            "name": "damacy/default/input_transfer.ms_avg",
            "value": 3.89789,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.64768,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 84.766,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 236.615,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 8844.99,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.38303,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/input_transfer.ms_avg",
            "value": 3.96638,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.84258,
            "unit": "ms"
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
          "id": "0db456b8d1a8f2c4fc3174f8681bd276bfa8636f",
          "message": "python: fix stale n_io_threads docstring (#141)\n\nThe n_io_threads parameter docstring still said `None` selects\n`max_concurrency()`. After #136 made the cluster-tuned values the\ndefault, n_io_threads defaults to the literal `DEFAULT_IO_THREADS` and\nno longer accepts `None`. Docstring-only; no functional change.\n\n---------\n\nCo-authored-by: Nathan Clack <nclack@biohub.org>",
          "timestamp": "2026-06-09T21:21:44Z",
          "url": "https://github.com/nclack/damacy/commit/0db456b8d1a8f2c4fc3174f8681bd276bfa8636f"
        },
        "date": 1781045684727,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 77.9587,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 1005.52,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9201.68,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 2.94419,
            "unit": "ms"
          },
          {
            "name": "damacy/default/input_transfer.ms_avg",
            "value": 3.84823,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.69994,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 77.5424,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 228.386,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9306.8,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 2.98002,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/input_transfer.ms_avg",
            "value": 3.91929,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.91133,
            "unit": "ms"
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
          "id": "dbc633eaafe28fa4c3b67219b4d1271867733814",
          "message": "config: size metadata caches to fit lookahead (closes #134) (#142)\n\n## Problem\n\nOn the metadata-prefetch path, a saturated metadata cache surfaced as a\n**fatal `DAMACY_BUDGET`** from `pop()` instead of back-pressuring —\nfailing the sample and taking the whole pipeline terminal. The design\ndoc said saturation should *stall* the stage until the watermark\nadvances. Closes #134.\n\n## What\n\nSize the caches so the in-flight + staging working set always fits, and\nmake any residual saturation a recoverable stall rather than fatal:\n\n- **Cache floors at create.** array-meta / chunk-layout caches must be\n`>= lookahead_samples + 2*samples_per_batch`; shard-index `>=\n(lookahead_samples + 2*samples_per_batch) * max_shards_per_sample`. The\n`+ 2*samples_per_batch` is the staging lag — samples\npopped-but-not-yet-committed still pin their cache entries until the\nwatermark passes them (bounded by the 2 batch slots). Sub-floor configs\nare rejected at `Pipeline()` with a clear `InvalidArgument`.\n- **Bounded per-sample shard fan-out.** New `max_shards_per_sample`\nknob; a sample whose AABB touches more shards is rejected at runtime\n(`DAMACY_INVAL`), making the shard floor a sound bound.\n- **Saturation is recoverable, not fatal.** The cache returns\n`DAMACY_AGAIN`; the prefetcher stalls the sample and replays it on a\nlater tick once the watermark advances (the #127 back-pressure\nconvention, and the design doc's original \"stall until the watermark\nadvances\" intent). No `abort()`, no crash.\n- `max_shards_per_sample` defaults via `damacy_tuning_defaults()` (the\nsingle source of truth, #136) — explicit, no `0`-sentinel. The design\ndoc was updated to match.\n\n## Verification\n\n- L40 (sm_89): C **ctest 33/33**, **pytest 92 passed / 2 skipped**\n(torch-absent).\n- New `test_saturation_returns_again` drives the all-pinned regime and\nasserts `DAMACY_AGAIN` + retry-succeeds-after-`advance_watermark` (not a\ncrash).\n- CI on the self-hosted runner.\n\n---------\n\nCo-authored-by: Nathan Clack <nclack@biohub.org>",
          "timestamp": "2026-06-10T03:23:45Z",
          "url": "https://github.com/nclack/damacy/commit/dbc633eaafe28fa4c3b67219b4d1271867733814"
        },
        "date": 1781101975300,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 189.149,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 980.458,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9266.32,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.05293,
            "unit": "ms"
          },
          {
            "name": "damacy/default/input_transfer.ms_avg",
            "value": 3.76594,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.6856,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 165.682,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 218.132,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9323.09,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.08129,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/input_transfer.ms_avg",
            "value": 3.83551,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.89509,
            "unit": "ms"
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
          "id": "ff5ad9c15b06a5d06fbae5de1b224aa61c2e9326",
          "message": "config: n_io_threads capped by sanity limit, not host CPU count (#145)\n\n## Problem\n\n#136 set `damacy_tuning_defaults().n_io_threads = 64`, but\n`validate_config` required `n_io_threads <= host CPU count` — so any\ndefaults-built config fails `damacy_create` on a sub-64-CPU host. The\nself-hosted runner's `test` job has been red since #136\n(`test_validate_accepts_tuning_defaults`). Diagnosed while shepherding\n#144.\n\n## Fix\n\nIO workers are blocking and IO-bound; oversubscribing CPUs is\nlegitimate. Replace the host-CPU bound with an explicit sanity cap\n`DAMACY_MAX_IO_THREADS = 1024`, mirrored to Python via\n`_native.MAX_IO_THREADS`. Test updated to assert the new contract\n(over-CPU valid; over-cap invalid).\n\n## Verification\n\nL40: ctest 33/33; pytest 92 passed / 2 skipped / 1 failed — the failure\nis `test_infinite_generator_feeds_multi_sample_batches`, the\npre-existing main breakage fixed by #144 (this branch forks main without\nit). Should turn the runner's `test` job green (together with #144).\n\nCo-authored-by: Nathan Clack <nclack@biohub.org>",
          "timestamp": "2026-06-11T00:01:13Z",
          "url": "https://github.com/nclack/damacy/commit/ff5ad9c15b06a5d06fbae5de1b224aa61c2e9326"
        },
        "date": 1781178713372,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 169.259,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 987.97,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9268.02,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 2.98898,
            "unit": "ms"
          },
          {
            "name": "damacy/default/input_transfer.ms_avg",
            "value": 3.7639,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.68463,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 157.853,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 212.394,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9351.03,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.06928,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/input_transfer.ms_avg",
            "value": 3.83522,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.90133,
            "unit": "ms"
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
          "id": "cca3b0928dff78059f451420406a4ede908d369a",
          "message": "bench: ceiling probe + wire-rate report (#150)\n\nRead amplification from unaligned sampling is by design, so \"how fast is\nthe bench\" needs a denominator: what could the mount deliver for this op\nstream if the pipeline were perfect? This adds that accounting end to\nend.\n\nThree pieces:\n\n1. Report: two summary lines after `throughput`, computed from fields\n   already in results.json (old archived runs re-render with them too):\n   - `read amplification` = io bytes / sample bytes — a constant per\n     sample-shape x chunk-geometry; drift flags a planner regression.\n   - `wire rate` = io bytes / wall — what the mount actually sustained.\n\n2. `bench/preadreplay`: the trace-replay harness from the 06-10/06-12\n   investigations, promoted from scratch tooling. Replays a captured op\n   stream with bare pread workers pulling from a shared cursor (mirrors\n   the io_queue), so its throughput is the mount ceiling for exactly\n   that workload. `pinned direct <trace> 64` is the canonical cold probe\n   (O_DIRECT keeps it honest on a node that already read the data).\n\n3. Capture: `DAMACY_TRACE_READS=<file>` appends one line per posted read\n   op (\"path offset len\", submission order) in store_fs submit. Opt-in,\n   off-path when unset.\n\nThe reading: `wire rate / ceiling` = pipeline efficiency;\n`ceiling / amplification` = expected peak throughput.\n\nValidated on L40 (33/33 ctest), full loop on the 42-FOV dynacell preset:\n\n    wall (steady-state)                                3.20 s\n    throughput              4.19 GB/s  (sample volume / wall)\n    read amplification       2.80x  (io bytes / sample bytes)\n    wire rate                   11.73 GB/s  (io bytes / wall)\n\n    $ build/bench/preadreplay pinned direct <captured-trace> 64\npinned direct nthreads=64 nops=30513 nfiles=1144 41.15 GB in 2.49s =\n16.52 GB/s\n\ni.e. that run: 71% of the mount that hour, expected peak 16.52 / 2.80 =\n5.9 GB/s. The fresh capture matched the canonical trace op-for-op\n(30,513 ops, 1,144 files) — the planner is deterministic under a seed.\n\nIndependent of #149 (touches adjacent but disjoint hunks in store_fs.c;\nmerges clean in either order).\n\nCo-authored-by: Nathan Clack <nclack@biohub.org>",
          "timestamp": "2026-06-12T02:15:42Z",
          "url": "https://github.com/nclack/damacy/commit/cca3b0928dff78059f451420406a4ede908d369a"
        },
        "date": 1781262674620,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "damacy/default/init",
            "value": 360.656,
            "unit": "ms"
          },
          {
            "name": "damacy/default/time_to_first_batch",
            "value": 964.623,
            "unit": "ms"
          },
          {
            "name": "damacy/default/wall",
            "value": 9373.26,
            "unit": "ms"
          },
          {
            "name": "damacy/default/io.ms_avg",
            "value": 3.76291,
            "unit": "ms"
          },
          {
            "name": "damacy/default/input_transfer.ms_avg",
            "value": 3.75739,
            "unit": "ms"
          },
          {
            "name": "damacy/default/assemble.ms_avg",
            "value": 1.71576,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/init",
            "value": 322.275,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/time_to_first_batch",
            "value": 344.751,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/wall",
            "value": 9504.92,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/io.ms_avg",
            "value": 3.84462,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/input_transfer.ms_avg",
            "value": 3.82867,
            "unit": "ms"
          },
          {
            "name": "damacy/mixed/assemble.ms_avg",
            "value": 1.9282,
            "unit": "ms"
          }
        ]
      }
    ]
  }
}