// Compile-time limits shared across damacy's modules.
#pragma once

#include <stdint.h>

// Maximum tensor rank we'll plan over. NGFF is canonically 5D; this leaves
// headroom and keeps fixed-size on-stack arrays small.
#define DAMACY_MAX_RANK 32

// Per-chunk byte caps. compressed_nbytes / decompressed_nbytes in
// chunk_plan are uint32_t; we deliberately cap chunks at 4 GB so the
// 32-bit fields are safe. Target chunks are ~1 MB.
#define DAMACY_MAX_CHUNK_BYTES UINT32_MAX

// Per-shard file size cap. file_offset is uint64_t but we only ever
// expect to use ~46 bits. Asserted at shard-index parse time. Target
// shards are ~1 GB.
#define DAMACY_MAX_SHARD_BYTES (1ull << 46)
