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

#define DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK 16u
#define DAMACY_BLOSC_MAX_TYPESIZE 8u
