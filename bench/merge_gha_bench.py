#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Concatenate per-scenario github-action-benchmark JSON files into one.

The action takes one tool format per call (smaller-is-better vs
bigger-is-better), so we publish two merged files. This script lets the
workflow drive the scenario list from a single loop without duplicating
the names in a Python merge block.

  uv run bench/merge_gha_bench.py \\
    --out merged.json bench-default-smaller.json bench-mixed-smaller.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    merged: list[dict] = []
    for p in args.inputs:
        merged.extend(json.loads(p.read_text()))
    args.out.write_text(json.dumps(merged, indent=2))
    print(
        f"merged {len(args.inputs)} file(s), {len(merged)} metrics, into {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
