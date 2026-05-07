window.BENCHMARK_DATA = {
  "lastUpdate": 1778195710555,
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
      }
    ]
  }
}