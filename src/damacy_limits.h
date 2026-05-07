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

// Per-substream uncompressed byte cap passed to nvcomp's batched decode.
// Compile-time ceiling on the per-batch-element size. Larger values
// inflate nvcomp's temp scratch (≈ MAX_CHUNKS_PER_WAVE × this × 2 waves
// × 2 codecs); 512 KB keeps the worst case under 1 GB on the default
// 1 GB device buffer config. Actual scratch sizing also takes the
// runtime per-wave decompress budget — see wave_init.
#define DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES (512ull << 10) // 512 KB

// Wave cap. Decoupled from the per-batch cap (DAMACY_MAX_CHUNKS_PER_BATCH
// in damacy.c): nvcomp temp scratch is sized as MAX_CHUNKS_PER_WAVE ×
// MAX_CHUNK_UNCOMPRESSED (× 2 waves), so we keep this small and let
// large batches split across multiple waves.
#define DAMACY_MAX_CHUNKS_PER_WAVE 512u

// Per-zarr-chunk blosc1 nblocks cap. Derivation:
//   max chunk uncompressed = DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES (512 KB)
//   blosc encoder min blocksize = 64 KB (compute_blocksize floor for
//                                        splittable codecs, c-blosc)
//   ⇒ worst-case nblocks = 512 KB / 64 KB = 8 (cap doubled for safety).
// Inputs with more blocks are rejected at parse with DAMACY_DECODE.
#define DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK 16u
#define DAMACY_BLOSC_MAX_TYPESIZE 8u
