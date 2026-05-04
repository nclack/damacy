#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""One-shot damacy bench runner.

Generates the dataset (via bench/gen_dataset.py) on first run, then invokes
the compiled damacy_bench binary against it. Defaults match gen_dataset.py
so a no-arg invocation works after `cmake --build build`.

  uv run bench/run.py                  # gen if missing, then run
  uv run bench/run.py --regen          # delete + regenerate dataset first
  uv run bench/run.py --batches 32 --peek   # extra flags pass through

Use --root to point at a different store. Bench-side knobs (--shape, --uri,
--rank) live on this script and are forwarded; everything else is passed
through to damacy_bench unchanged. To customize the on-disk geometry, run
bench/gen_dataset.py manually and point --root at the result.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = REPO_ROOT / "bench" / "data" / "bench.zarr"
GEN_SCRIPT = REPO_ROOT / "bench" / "gen_dataset.py"
BENCH_BIN = REPO_ROOT / "build" / "bench" / "damacy_bench"
DEFAULT_URI = "scale0/image"
DEFAULT_SHAPE = "32,128,128"  # one inner chunk per sample
DEFAULT_RANK = 3


def gen_dataset(root: Path) -> int:
    root.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["uv", "run", str(GEN_SCRIPT), "--out", str(root)]
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd).returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        allow_abbrev=False,
        description=__doc__.splitlines()[0],
    )
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"zarr store path (default: {DEFAULT_ROOT})")
    ap.add_argument("--uri", default=DEFAULT_URI,
                    help=f"array name relative to --root (default: {DEFAULT_URI})")
    ap.add_argument("--shape", default=DEFAULT_SHAPE,
                    help=f"sample shape csv (default: {DEFAULT_SHAPE})")
    ap.add_argument("--rank", type=int, default=DEFAULT_RANK,
                    help=f"sample rank (default: {DEFAULT_RANK})")
    ap.add_argument("--regen", action="store_true",
                    help="delete and regenerate the dataset before running")
    args, passthrough = ap.parse_known_args()

    if args.regen and args.root.exists():
        print(f"+ rm -rf {args.root}", file=sys.stderr)
        shutil.rmtree(args.root)
    if not args.root.exists():
        rc = gen_dataset(args.root)
        if rc != 0:
            return rc

    if not BENCH_BIN.exists():
        print(f"error: {BENCH_BIN} not found; run `cmake --build build` first",
              file=sys.stderr)
        return 1

    cmd = [
        str(BENCH_BIN),
        "--root", str(args.root),
        "--uri", args.uri,
        "--rank", str(args.rank),
        "--shape", args.shape,
        *passthrough,
    ]
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
