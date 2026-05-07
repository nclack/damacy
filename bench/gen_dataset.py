#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "zarr>=3",
#   "ngff-zarr>=0.13",
# ]
# ///
"""Generate a synthetic sharded zarr v3 store for the damacy bench.

Writes a single-scale NGFF v0.5 image. Defaults are tuned for a quick smoke
test (~100 MB on disk at default entropy).

Run with: uv run bench/gen_dataset.py --out path/to/store.zarr
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import ngff_zarr as nz
from zarr.codecs import BloscCname, BloscCodec, BloscShuffle, ZstdCodec


def parse_shape(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def make_compressors(codec: str, clevel: int, dtype: np.dtype):
    if codec == "none":
        return []
    if codec == "zstd":
        return [ZstdCodec(level=clevel, checksum=False)]
    if codec == "blosc-zstd":
        return [BloscCodec(
            cname=BloscCname.zstd, clevel=clevel,
            shuffle=BloscShuffle.shuffle, typesize=int(dtype.itemsize),
        )]
    if codec == "blosc-lz4":
        return [BloscCodec(
            cname=BloscCname.lz4, clevel=clevel,
            shuffle=BloscShuffle.shuffle, typesize=int(dtype.itemsize),
        )]
    raise SystemExit(f"unknown --codec {codec!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output store path")
    ap.add_argument("--shape", default="64,1024,1024", help="array shape (csv)")
    ap.add_argument(
        "--inner",
        default="32,128,128",
        help="inner chunk shape (csv)",
    )
    ap.add_argument(
        "--shard",
        default="64,512,512",
        help="shard / outer chunk shape (csv); must be a multiple of inner",
    )
    ap.add_argument("--dtype", default="uint16")
    ap.add_argument("--codec", default="zstd",
                    choices=["none", "zstd", "blosc-zstd", "blosc-lz4"])
    ap.add_argument("--clevel", type=int, default=3,
                    help="compression level passed to zstd / blosc")
    ap.add_argument(
        "--entropy",
        type=float,
        default=0.5,
        help="0.0 = constant (highly compressible), 1.0 = uniform random "
        "(barely compressible)",
    )
    ap.add_argument(
        "--seed", type=int, default=42, help="rng seed for reproducibility"
    )
    args = ap.parse_args()

    shape = parse_shape(args.shape)
    inner = parse_shape(args.inner)
    shard = parse_shape(args.shard)
    if len(shape) != len(inner) or len(shape) != len(shard):
        print("rank mismatch between shape/inner/shard", file=sys.stderr)
        return 1
    for d, (s, sh) in enumerate(zip(inner, shard)):
        if sh % s != 0:
            print(f"shard[{d}]={sh} is not a multiple of inner[{d}]={s}",
                  file=sys.stderr)
            return 1
    chunks_per_shard = tuple(sh // s for s, sh in zip(inner, shard))

    dtype = np.dtype(args.dtype)
    out = Path(args.out)
    if out.exists():
        print(f"refusing to overwrite existing path: {out}", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    n = int(np.prod(shape))
    if args.entropy <= 0.0:
        data = np.zeros(shape, dtype=dtype)
    else:
        if dtype == np.uint8:
            full = rng.integers(0, 256, size=n, dtype=np.uint8)
        elif dtype == np.uint16:
            full = rng.integers(0, 65536, size=n, dtype=np.uint16)
        elif dtype == np.float32:
            full = rng.standard_normal(n, dtype=np.float32)
        else:
            full = rng.integers(0, 256, size=n).astype(dtype)
        if args.entropy < 1.0:
            mask = rng.random(n) < args.entropy
            base = np.zeros(n, dtype=dtype)
            base[mask] = full[mask]
            full = base
        data = full.reshape(shape)

    image = nz.to_ngff_image(data)
    multiscales = nz.to_multiscales(image, scale_factors=[], chunks=inner)
    nz.to_ngff_zarr(
        str(out),
        multiscales,
        version="0.5",
        chunks_per_shard=chunks_per_shard,
        compressors=make_compressors(args.codec, args.clevel, dtype),
    )
    print(f"done. store: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
