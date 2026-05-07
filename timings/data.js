window.BENCHMARK_DATA = {
  "lastUpdate": 1778195708126,
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
      }
    ]
  }
}