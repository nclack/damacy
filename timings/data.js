window.BENCHMARK_DATA = {
  "lastUpdate": 1779115446034,
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
      }
    ]
  }
}