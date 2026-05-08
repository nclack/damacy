#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "numcodecs<0.13",
# ]
# ///
"""Seed corpus for fuzz_blosc1_host.

Drops a handful of small blosc1 chunks (varied codec / shuffle / typesize /
nblocks) plus boundary cases (empty, truncated header, header-with-no-payload,
header with absurd nbytes) into the script's directory.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from numcodecs import Blosc

HERE = Path(__file__).resolve().parent


def write(name: str, payload: bytes) -> None:
    (HERE / name).write_bytes(payload)


def encode(data: np.ndarray, *, cname: str, shuffle: int, blocksize: int = 0) -> bytes:
    codec = Blosc(cname=cname, clevel=5, shuffle=shuffle, blocksize=blocksize)
    return bytes(codec.encode(data))


def main() -> int:
    pat = np.tile(np.arange(64, dtype=np.int32), 16)  # 4 KB

    write(
        "valid_lz4_noshuffle_ts4.bin",
        encode(pat, cname="lz4", shuffle=Blosc.NOSHUFFLE),
    )
    write(
        "valid_lz4_shuffle_ts4.bin",
        encode(pat, cname="lz4", shuffle=Blosc.SHUFFLE),
    )
    write(
        "valid_lz4_bitshuffle_ts4.bin",
        encode(pat, cname="lz4", shuffle=Blosc.BITSHUFFLE),
    )
    write(
        "valid_zstd_noshuffle_ts4.bin",
        encode(pat, cname="zstd", shuffle=Blosc.NOSHUFFLE),
    )
    # typesize=1 so LZ4 takes the single-substream path.
    write(
        "valid_lz4_noshuffle_ts1.bin",
        encode(pat.view(np.uint8).copy(), cname="lz4", shuffle=Blosc.NOSHUFFLE),
    )
    # Multiple blocks (4 KB / 1 KB block = 4 blocks).
    write(
        "valid_lz4_multiblock.bin",
        encode(pat, cname="lz4", shuffle=Blosc.NOSHUFFLE, blocksize=1024),
    )

    # Boundary: empty input — every err branch is gated on size >= 16.
    write("empty.bin", b"")

    # Boundary: header truncated at byte 15 (one short of the 16-byte
    # minimum, so err=1 fires).
    write("trunc_header.bin", b"\x00" * 15)

    # Boundary: 16-byte header with blocksize=0 (err=2).
    write(
        "blocksize_zero.bin",
        struct.pack("<BBBB III", 0, 0, (1 << 5), 4, 4096, 0, 100),
    )

    # Boundary: header with absurd nbytes that nominally succeeds the
    # divide but blows past the nblocks cap (err=5: nblocks > 32).
    write(
        "absurd_nbytes.bin",
        struct.pack("<BBBB III", 0, 0, (1 << 5), 4, 0xFFFFFFFF, 1, 200),
    )

    # Boundary: valid 16-byte header but payload truncated (cbytes mid-stream).
    # err=4 fires because cbytes != compressed_nbytes when the harness
    # passes size < cbytes.
    write(
        "valid_header_no_payload.bin",
        struct.pack("<BBBB III", 0, 0, (1 << 5), 4, 4096, 1024, 1024)
        + b"\x00" * 16,  # bstarts only, no block data
    )

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
