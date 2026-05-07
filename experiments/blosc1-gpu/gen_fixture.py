#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "numcodecs<0.13",
# ]
# ///
"""Generate a single blosc1+lz4 fixture for the GPU decompress spike.

Writes two files alongside this script:
  fixture_lz4_noshuffle.blosc1   — bytes from numcodecs.Blosc(cname='lz4', shuffle=NOSHUFFLE)
  fixture_lz4_noshuffle.raw      — the same bytes uncompressed (ground truth)

Also prints the parsed blosc1 header so the spike can be sanity-checked.

The aim is a *small* single-block fixture. A small, highly-compressible
buffer keeps the test simple — one block, one LZ4 frame, no shuffle to
reverse.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
from numcodecs import Blosc


HERE = Path(__file__).resolve().parent


def main() -> int:
    # Build a small, strongly-compressible buffer: repeat a 64-int pattern
    # 256× → 16384 int32s = 64 KB. LZ4 will produce a real compressed block
    # (no MEMCPYED fallback). Using a low-entropy repeating pattern keeps
    # the LZ4 frame small and the spike fast.
    pattern = np.arange(64, dtype=np.int32)
    data = np.tile(pattern, 256)  # 16384 elements, 64 KB
    raw = data.tobytes()

    codec = Blosc(
        cname="lz4",
        clevel=5,
        shuffle=Blosc.NOSHUFFLE,
        blocksize=0,  # auto
    )
    # Pass the ndarray directly so Blosc picks up dtype.itemsize as typesize.
    encoded = bytes(codec.encode(data))

    # Sanity-parse the 16-byte blosc1 header so the spike author knows what
    # to expect.
    if len(encoded) < 16:
        print(f"unexpected encoded size {len(encoded)}", file=sys.stderr)
        return 1
    version, versionlz, flags, typesize = struct.unpack_from("BBBB", encoded, 0)
    nbytes, blocksize, cbytes = struct.unpack_from("<III", encoded, 4)
    nblocks = (nbytes + blocksize - 1) // blocksize
    bstarts = list(struct.unpack_from(f"<{nblocks}i", encoded, 16))

    print(f"encoded: {len(encoded)} bytes ({len(encoded) / len(raw):.3f}× of raw)")
    print(f"  version={version} versionlz={versionlz} flags=0x{flags:02x} typesize={typesize}")
    print(f"  nbytes={nbytes} blocksize={blocksize} cbytes={cbytes}")
    print(f"  nblocks={nblocks} bstarts={bstarts}")
    # Show per-block compressed sizes.
    for i, off in enumerate(bstarts):
        nxt = bstarts[i + 1] if i + 1 < nblocks else cbytes
        print(f"  block[{i}] offset={off} compressed_size={nxt - off}")

    out_blosc = HERE / "fixture_lz4_noshuffle.blosc1"
    out_raw = HERE / "fixture_lz4_noshuffle.raw"
    out_blosc.write_bytes(encoded)
    out_raw.write_bytes(raw)
    print(f"wrote {out_blosc}")
    print(f"wrote {out_raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
