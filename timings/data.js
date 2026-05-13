window.BENCHMARK_DATA = {
  "lastUpdate": 1778682707150,
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
      }
    ]
  }
}