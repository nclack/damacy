// Internal: parse the per-shard "chunk index" footer/header.
//
// Format: a contiguous block of (offset_u64_le, nbytes_u64_le) per inner
// chunk in the shard's row-major chunk grid, followed by a 4-byte CRC32C
// (little-endian) over the index bytes (excluding the CRC itself).
//
// "index_location": "end"   → block sits at the end of the shard file
// "index_location": "start" → block sits at the start
//
// Empty (not-stored) inner chunks have offset == 0xFFFFFFFFFFFFFFFF and
// nbytes == 0xFFFFFFFFFFFFFFFF.
#pragma once

#include <stddef.h>
#include <stdint.h>

struct strbuf;

#define ZARR_SHARD_EMPTY_OFFSET 0xFFFFFFFFFFFFFFFFull
#define ZARR_SHARD_EMPTY_NBYTES 0xFFFFFFFFFFFFFFFFull

struct zarr_shard_entry
{
  uint64_t offset;
  uint64_t nbytes;
};

// Parse a contiguous index buffer of size `n_entries * 16 + 4`. On
// success, fills `entries[0..n_entries)` with decoded values. Returns 0 on
// success, non-zero if the trailing CRC does not match.
int
zarr_shard_index_parse(const void* buf,
                       size_t buf_len,
                       size_t n_entries,
                       struct zarr_shard_entry* entries);

// Index size in bytes for a shard with `n_entries` inner chunks.
size_t
zarr_shard_index_size(size_t n_entries);

// Build "<prefix>/c/<a>/<b>/...<coord_{rank-1}>" into sb (which is reset
// first). prefix may be empty/NULL; coord and rank describe the shard's
// row-major position in the shard grid. Returns 0 on success.
int
zarr_shard_path_build(struct strbuf* sb,
                      const char* prefix,
                      const uint64_t* shard_coord,
                      uint8_t rank);
