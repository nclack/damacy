# blosc1-gpu derisk

Standalone harness validating that nvcomp's batched LZ4 and Zstd
decompressors consume the bytes that blosc1 produces, so the GPU-side
blosc1 pipeline proposed in `docs/issue-2-blosc-design.md` is feasible.

## Run

From inside `nix develop`:

```
./gen_fixture.py
NVCOMP_INC=$(grep '^NVCOMP_INCLUDE_DIR' ../../build/CMakeCache.txt | cut -d= -f2)
NVCOMP_LIB=$(grep '^NVCOMP_LIBRARY' ../../build/CMakeCache.txt | cut -d= -f2)
nvcc -O2 -std=c++17 -arch=sm_75 spike.cu -I"$NVCOMP_INC" "$NVCOMP_LIB" -lcuda -o spike
for f in lz4_noshuffle_ts4 lz4_shuffle_ts4 zstd_noshuffle_ts4 \
         lz4_noshuffle_ts1 lz4_noshuffle_ts4_mb; do
  ./spike "$f"
done
```

## Result matrix (May 2026, all PASS)

| fixture | codec | shuffle | typesize | blocks | sub-streams/block | total subs | bytes round-tripped |
|---|---|---|---|---|---|---|---|
| `lz4_noshuffle_ts4`    | lz4  | off | 4 |  1 | 4 |   4 |  65536 |
| `lz4_shuffle_ts4`      | lz4  | on  | 4 |  1 | 4 |   4 |  65536 |
| `zstd_noshuffle_ts4`   | zstd | off | 4 |  1 | 1 |   1 |  65536 |
| `lz4_noshuffle_ts1`    | lz4  | off | 1 |  1 | 1 |   1 |  65536 |
| `lz4_noshuffle_ts4_mb` | lz4  | off | 4 | 16 | 4 |  64 | 4194304 |

## Format details surfaced

These three came out of the spike and are *not* documented in the c-blosc1
README. Each is load-bearing for any GPU parser.

1. **Sub-stream prefixing is universal.** Every block in every codec
   variant starts with an `int32` cbytes prefix, even when the block is
   "single sub-stream" (zstd, or lz4 with typesize=1).

2. **Sub-stream count per block is codec-dependent, not just typesize-dependent:**
   - lz4 / lz4hc with typesize > 1 → split into `typesize` sub-streams.
   - lz4 / lz4hc with typesize = 1 → 1 sub-stream.
   - zstd → 1 sub-stream regardless of typesize.

   The header carries no "split" flag — the count is implicit and must be
   discovered by walking the block payload until accumulated sub-stream
   sizes equal the block's compressed extent.

3. **`bstarts` are NOT in block-index order.** Blosc1 compresses blocks
   on multiple writer threads and writes each block to the output buffer
   in completion order, recording the offset under that block's index.
   `bstarts[1]` may point earlier in the buffer than `bstarts[0]`. Each
   block's compressed-payload end-offset must be derived by sorting the
   `bstarts` values and taking the next-higher one.

## Implications for the design

- The sub-stream — not the block — is the unit handed to nvcomp's batched
  API. For `1 MB chunks × 64 KB blocks × typesize=4` LZ4 data that's
  `16 × 4 = 64` LZ4 calls per chunk batched together. Generous parallel
  width; metadata table grows by `typesize×` for the LZ4 case.
- For zstd inner codec, batching is per-block (1 zstd call per block).
- Byte-shuffle reverse is a per-block CPU/GPU transpose applied after
  decompress. Tested on CPU here; will become a small CUDA kernel in
  the actual implementation.

## Suggested follow-ups (not done here)

- Memcpyed flag (forced-uncompressed blocks): synthesize via random data
  or `clevel=0` and verify the spike's parser detects + handles them.
- Bitshuffle (`shuffle=2`): fixture + reference reversal; punt the GPU
  bitshuffle kernel until needed.
- Partial last block: fixture where `nbytes % blocksize != 0`; verify
  the per-substream dst sizing on the trailing block.
