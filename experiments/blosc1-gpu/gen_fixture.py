#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "numcodecs<0.13",
# ]
# ///
"""Generate blosc1 fixtures for the GPU decompress derisk.

Emits five (.blosc1, .raw) pairs alongside this script:

  lz4_noshuffle_ts4      cname=lz4   shuffle=off  typesize=4  64 KB / 1 block
  lz4_shuffle_ts4        cname=lz4   shuffle=on   typesize=4  64 KB / 1 block
  zstd_noshuffle_ts4     cname=zstd  shuffle=off  typesize=4  64 KB / 1 block
  lz4_noshuffle_ts1      cname=lz4   shuffle=off  typesize=1  64 KB / 1 block
  lz4_noshuffle_ts4_mb   cname=lz4   shuffle=off  typesize=4  4 MB / 64 blocks

Each .blosc1 is the codec.encode(...) output; each .raw is the original
bytes. The spike consumes both and verifies round-trip.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from numcodecs import Blosc


HERE = Path(__file__).resolve().parent


def make_pattern_int32(nelems: int) -> np.ndarray:
    """Repeating low-entropy pattern that compresses well under both LZ4 and zstd."""
    pat = np.arange(64, dtype=np.int32)
    reps = (nelems + len(pat) - 1) // len(pat)
    return np.tile(pat, reps)[:nelems]


def emit(name: str, *, data: np.ndarray, cname: str, shuffle: int, typesize: int,
         blocksize: int = 0) -> None:
    """Encode `data` with the given Blosc options and write fixtures."""
    codec = Blosc(cname=cname, clevel=5, shuffle=shuffle, blocksize=blocksize)
    # Pass ndarray so dtype.itemsize → typesize. For typesize=1 we override
    # by passing the bytes view (uint8 ndarray).
    if typesize == data.dtype.itemsize:
        encoded = bytes(codec.encode(data))
    else:
        encoded = bytes(codec.encode(data.view(np.uint8)))
    raw = data.tobytes()

    # Parse header to print + sanity-check.
    version, versionlz, flags, ts = struct.unpack_from("BBBB", encoded, 0)
    nbytes, bs, cb = struct.unpack_from("<III", encoded, 4)
    nblocks = (nbytes + bs - 1) // bs
    bstarts = list(struct.unpack_from(f"<{nblocks}i", encoded, 16))
    print(f"{name:30s}  {len(encoded):>9} bytes  ratio={len(encoded)/len(raw):.3f}")
    print(f"  flags=0x{flags:02x} typesize={ts} nbytes={nbytes} blocksize={bs}"
          f" nblocks={nblocks}")
    if ts != typesize:
        print(f"  WARNING: requested typesize={typesize} but header says {ts}",
              file=sys.stderr)

    (HERE / f"{name}.blosc1").write_bytes(encoded)
    (HERE / f"{name}.raw").write_bytes(raw)


def main() -> int:
    # 64 KB at typesize=4 = 16384 int32s.
    data_64k_i32 = make_pattern_int32(16384)
    # 64 KB at typesize=1 = 65536 uint8s with the same low-entropy pattern.
    data_64k_u8 = data_64k_i32.view(np.uint8).copy()
    # 4 MB at typesize=4 = 1048576 int32s.
    data_4m_i32 = make_pattern_int32(1024 * 1024)

    emit("lz4_noshuffle_ts4", data=data_64k_i32,
         cname="lz4", shuffle=Blosc.NOSHUFFLE, typesize=4)
    emit("lz4_shuffle_ts4", data=data_64k_i32,
         cname="lz4", shuffle=Blosc.SHUFFLE, typesize=4)
    emit("zstd_noshuffle_ts4", data=data_64k_i32,
         cname="zstd", shuffle=Blosc.NOSHUFFLE, typesize=4)
    emit("lz4_noshuffle_ts1", data=data_64k_u8,
         cname="lz4", shuffle=Blosc.NOSHUFFLE, typesize=1)
    emit("lz4_noshuffle_ts4_mb", data=data_4m_i32,
         cname="lz4", shuffle=Blosc.NOSHUFFLE, typesize=4,
         blocksize=65536)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
