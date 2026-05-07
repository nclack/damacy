# blosc1-gpu derisk

Standalone spike: does `nvcompBatchedLZ4DecompressAsync` consume the LZ4
bytes that blosc1 produces? If yes, GPU-side blosc1 decompression is
viable and we can skip the CPU-decode stage proposed in
`docs/issue-2-blosc-design.md`.

## Run

```
./gen_fixture.py
NVCOMP_INC=$(grep '^NVCOMP_INCLUDE_DIR' ../../build/CMakeCache.txt | cut -d= -f2)
NVCOMP_LIB=$(grep '^NVCOMP_LIBRARY' ../../build/CMakeCache.txt | cut -d= -f2)
nvcc -O2 -std=c++17 -arch=sm_75 spike.cu -I"$NVCOMP_INC" "$NVCOMP_LIB" -lcuda -o spike
./spike
```

## Result (May 2026)

PASS — 16384 bytes round-trip byte-perfect. nvcomp's batched LZ4 consumes
raw `LZ4_compress_default` output as produced by blosc1 in its split
sub-streams.

## Format detail surfaced

A blosc1 block with codec ∈ {lz4, lz4hc, zstd}, typesize > 1, and
non-trivial blocksize is *split* into `typesize` independent
LZ4/zstd sub-streams concatenated as
`[int32 cbytes_i][frame_i] × typesize`. Each sub-stream decompresses to
`blocksize/typesize` bytes. The GPU batching unit is the sub-stream, not
the block — so per-chunk batched-decompress count is `nblocks × typesize`.

## Suggested follow-ups (cheap variations)

1. `shuffle=SHUFFLE` + typesize=4 — verify the unshuffle path with a
   reference CPU transpose
2. blosc-zstd equivalent — confirm `nvcompBatchedZstdDecompressAsync`
   accepts the same sub-stream shape
3. typesize=1 — confirm split is off (block = single frame, no prefix)
4. Multi-block fixture (4 MB / 64 KB blocksize) — exercise offset-table
   walk

These are minor extensions of `spike.cu`.
