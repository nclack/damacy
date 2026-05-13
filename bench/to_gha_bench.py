#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Convert bench/runs/<scenario>/<ts>/results.json into the JSON shape
expected by github-action-benchmark.

Emits two files (one per direction, since the action takes one tool
format per call):

  --out-smaller PATH   timings + per-stage averages (ms; smaller=better)
  --out-bigger  PATH   throughput (MB/s; bigger=better)

Each file is a JSON array of {name, unit, value} objects. Metric names
are prefixed with the scenario so different scenarios chart separately.

  uv run bench/to_gha_bench.py bench/runs/default/20260507-135830/results.json \\
    --out-smaller bench-smaller.json --out-bigger bench-bigger.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "results", type=Path, help="path to bench/runs/<scenario>/<ts>/results.json"
    )
    ap.add_argument("--out-smaller", type=Path, required=True)
    ap.add_argument("--out-bigger", type=Path, required=True)
    ap.add_argument(
        "--scenario",
        default=None,
        help="scenario name prefix (default: parent dir name)",
    )
    ap.add_argument(
        "--runner",
        default=None,
        help="optional runner-name prefix; metrics become "
        "<runner>/<scenario>/<metric>. Lets you keep "
        "histories from different machines separable on "
        "the gh-pages dashboard.",
    )
    args = ap.parse_args()

    scenario = args.scenario or args.results.parent.parent.name
    # Reject empty/whitespace runner explicitly so we never publish
    # leading-slash metric names like "/default/throughput", which would
    # pollute the gh-pages history with un-segregable rows.
    runner = (args.runner or "").strip() or None
    prefix = f"{runner}/{scenario}" if runner else scenario
    r = json.loads(args.results.read_text())
    timings = r.get("timings_ms", {})
    derived = r.get("derived", {})
    stages = {s["name"]: s for s in r.get("stages", [])}

    def stage_avg(name: str) -> float | None:
        s = stages.get(name)
        if s is None or s.get("count", 0) == 0:
            return None
        return s.get("ms_avg")

    smaller: list[dict] = []
    for key in ("init", "time_to_first_batch", "wall"):
        if key in timings:
            smaller.append(
                {
                    "name": f"{prefix}/{key}",
                    "unit": "ms",
                    "value": timings[key],
                }
            )
    # Aggregate decompress + per-codec sub-stages. The mixed scenario
    # exercises zstd and blosc-zstd; tracking only the aggregate would
    # hide codec-specific regressions.
    tracked_stages = (
        "io",
        "h2d",
        "decompress",
        "assemble",
        "decompress.parse",
        "decompress.zstd",
        "decompress.post",
    )
    for stage in tracked_stages:
        v = stage_avg(stage)
        if v is not None:
            smaller.append(
                {
                    "name": f"{prefix}/{stage}.ms_avg",
                    "unit": "ms",
                    "value": v,
                }
            )

    bigger: list[dict] = []
    if "throughput_mb_s" in derived:
        bigger.append(
            {
                "name": f"{prefix}/throughput",
                "unit": "MB/s",
                "value": derived["throughput_mb_s"],
            }
        )

    args.out_smaller.write_text(json.dumps(smaller, indent=2))
    args.out_bigger.write_text(json.dumps(bigger, indent=2))
    print(
        f"wrote {len(smaller)} smaller-is-better metrics to {args.out_smaller}",
        file=sys.stderr,
    )
    print(
        f"wrote {len(bigger)} bigger-is-better metrics to {args.out_bigger}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
