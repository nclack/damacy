#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Emit a damacy bench scenario that samples real arrays found under a plate.

Walks an OME-NGFF / zarr-v3 plate (`<plate>/<row>/<col>/<fov>/<level>`) for
level-N array directories and writes a scenario whose `dataset.uris` lists them
relative to --store-root. The bench reads each array's own shape from its
zarr.json, so heterogeneous FOV shapes need no uniform `zarr_shape`.

  uv run bench/gen_manifest.py --store-root /mnt/.../data \
      --plate dynacell/10_27_26/EXP.zarr --max-arrays 128 --out bench/scenarios/dynacell-plate.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-root", required=True)
    ap.add_argument("--plate", required=True, help="plate dir relative to --store-root")
    ap.add_argument("--level", default="0", help="multiscale level dir name (default 0)")
    ap.add_argument("--max-arrays", type=int, default=128)
    ap.add_argument("--name", default="dynacell-plate")
    ap.add_argument("--sample-shape", default="1,1,16,256,256")
    ap.add_argument("--samples-per-batch", type=int, default=64)
    ap.add_argument("--n-batches", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--lookahead", type=int, default=256)
    ap.add_argument("--gpu-mb", type=int, default=8192)
    ap.add_argument("--max-chunk-mb", type=int, default=4)
    ap.add_argument("--max-read-op-kb", type=int, default=0, help="0 = library default")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    store_root = Path(a.store_root)
    plate = store_root / a.plate
    found = subprocess.run(
        ["find", str(plate), "-mindepth", "4", "-maxdepth", "4", "-type", "d", "-name", a.level],
        capture_output=True, text=True, timeout=180,
    )
    dirs = sorted(line for line in found.stdout.splitlines() if line)
    if not dirs:
        sys.exit(f"no level-{a.level!r} arrays under {plate}")
    dirs = dirs[: a.max_arrays]
    uris = [str(Path(d).relative_to(store_root)) for d in dirs]

    pipeline = {
        "dtype": "f32",
        "lookahead_samples": a.lookahead,
        "n_io_threads": 16,
        "metadata_io_concurrency": 64,
        "max_gpu_memory_mb": a.gpu_mb,
        "max_chunk_uncompressed_mb": a.max_chunk_mb,
        "n_array_meta_cache": 4096,
        "n_shard_index_cache": 16384,
        "n_chunk_layout_cache": 4096,
    }
    if a.max_read_op_kb:
        pipeline["max_read_op_kb"] = a.max_read_op_kb

    scenario = {
        "name": a.name,
        "dataset": {"store_root": str(store_root), "n_zarrs": len(uris), "uris": uris},
        "sampling": {
            "sample_shape": [int(x) for x in a.sample_shape.split(",")],
            "n_batches": a.n_batches,
            "n_warmup_batches": a.warmup,
            "samples_per_batch": a.samples_per_batch,
            "seed": 1234,
        },
        "pipeline": pipeline,
    }
    Path(a.out).write_text(json.dumps(scenario, indent=2) + "\n")
    print(f"wrote {a.out}: {len(uris)} level-{a.level} arrays under {a.plate}")


if __name__ == "__main__":
    main()
