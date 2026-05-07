#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "zarr>=3",
# ]
# ///
"""Write a synthetic sharded zarr v3 array for damacy tests.

Content is a deterministic row-major linearization (data[i] = (i + offset)
masked to the dtype's range) so C tests can reconstruct the expected
value at any (y, x, ...) without needing a shared RNG.

Used by tests/test_damacy.c via tests/fixture.c::fixture_write_zarr.
"""
import argparse
import sys

import numpy as np
import zarr
from zarr.codecs import BloscCname, BloscCodec, BloscShuffle, ZstdCodec


def parse_shape(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def make_compressors(codec: str, dtype: np.dtype):
    if codec == "zstd":
        return [ZstdCodec(level=3, checksum=False)]
    if codec == "blosc-zstd":
        return [BloscCodec(
            cname=BloscCname.zstd,
            clevel=3,
            shuffle=BloscShuffle.shuffle,
            typesize=int(dtype.itemsize),
        )]
    if codec == "blosc-lz4":
        return [BloscCodec(
            cname=BloscCname.lz4,
            clevel=3,
            shuffle=BloscShuffle.shuffle,
            typesize=int(dtype.itemsize),
        )]
    raise SystemExit(f"unknown --codec {codec!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output array path")
    ap.add_argument("--shape", required=True, type=parse_shape)
    ap.add_argument("--inner", required=True, type=parse_shape,
                    help="inner chunk shape (within a shard)")
    ap.add_argument("--shard", required=True, type=parse_shape,
                    help="shard (outer chunk) shape; must be a multiple of inner")
    ap.add_argument("--dtype", default="uint16")
    ap.add_argument("--offset", type=int, default=0,
                    help="added to each element before dtype masking")
    ap.add_argument("--codec", default="zstd",
                    choices=["zstd", "blosc-zstd", "blosc-lz4"],
                    help="inner codec inside the sharding wrapper")
    args = ap.parse_args()

    if not (len(args.shape) == len(args.inner) == len(args.shard)):
        print("rank mismatch between shape/inner/shard", file=sys.stderr)
        return 1
    for d, (i, s) in enumerate(zip(args.inner, args.shard)):
        if s % i != 0:
            print(f"shard[{d}]={s} is not a multiple of inner[{d}]={i}",
                  file=sys.stderr)
            return 1

    np_dtype = np.dtype(args.dtype)
    n = int(np.prod(args.shape))

    data = np.arange(n, dtype=np.int64) + args.offset
    if np.issubdtype(np_dtype, np.unsignedinteger):
        modulus = int(np.iinfo(np_dtype).max) + 1
        data = data % modulus
    data = data.astype(np_dtype).reshape(args.shape)

    arr = zarr.create_array(
        store=args.out,
        shape=args.shape,
        dtype=np_dtype,
        chunks=args.inner,
        shards=args.shard,
        compressors=make_compressors(args.codec, np_dtype),
    )
    arr[...] = data
    return 0


if __name__ == "__main__":
    sys.exit(main())
