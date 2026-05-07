// Compile-time limits shared across damacy's modules.
#pragma once

#include <stdint.h>

// Maximum tensor rank we'll plan over.
#define DAMACY_MAX_RANK 31

// Per-chunk byte caps. compressed_nbytes / decompressed_nbytes in
// chunk_plan are uint32_t; we deliberately cap chunks at 4 GB so the
// 32-bit fields are safe. Target chunks are ~1 MB.
#define DAMACY_MAX_CHUNK_BYTES UINT32_MAX

// Per-shard file size cap. file_offset is uint64_t but we only ever
// expect to use ~46 bits. Asserted at shard-index parse time. Target
// shards are ~1 GB.
#define DAMACY_MAX_SHARD_BYTES (1ull << 46)

// Max bytes (including trailing NUL) for a shard path inlined into
// struct read_op. The plan queue stores read_ops by value across batches
// (step 5+), so paths can't be heap pointers owned by the planner —
// they'd dangle on the next plan. 224 leaves headroom for store_root +
// long uri + chunk-grid coordinates without inflating chunk_plan-sized
// records too much.
#define DAMACY_MAX_PATH 224

// Per-chunk blosc1 caps. The blosc1 wrapper splits each chunk into 1+
// blocks of fixed `blocksize` (encoder picks; typically 64 KB). With our
// chunk-size cap of ~1 MB and a 64 KB lower-bound blocksize, 16 blocks
// covers any chunk we'll see in practice. Bumped guardrails kick in if a
// real-world dataset blows past either; the GPU parser rejects with
// DAMACY_DECODE.
#define DAMACY_MAX_BLOCKS_PER_CHUNK 16u

// Maximum element size used by blosc shuffle/bitshuffle. Values beyond
// 8 don't appear in the codec wild (blosc1 stores typesize in a uint8
// header byte but in practice it's 1, 2, 4, or 8 — element sizes of
// common dtypes). Sized so the lz4 split factor stays bounded.
#define DAMACY_MAX_TYPESIZE 8u
