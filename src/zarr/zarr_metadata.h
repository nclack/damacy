// Internal: parse a zarr v3 array's zarr.json into a flat metadata struct.
#pragma once

#include "damacy_limits.h"
#include "dtype/dtype.h"

#include <stdint.h>

enum compression_codec
{
  CODEC_NONE = 0,
  CODEC_ZSTD = 1,
  CODEC_BLOSC_LZ4 = 2,
  CODEC_BLOSC_ZSTD = 3,
  // Sentinel for absent chunks emitted as fill_value by the planner.
  // Never appears in zarr.json — the planner sets it on fill chunk_plans
  // so the GPU parse skips them (no substreams, no memcpy ops).
  CODEC_FILL = 4,
};

struct codec_config
{
  enum compression_codec id;
};

// Max element size in bytes for any currently-supported dtype (u64/i64/f64).
// Sized fill_value buffers below and the chunk_plan fill slot.
#define DAMACY_MAX_DTYPE_BYTES 8

struct zarr_metadata
{
  enum dtype dtype;
  uint8_t rank;
  uint64_t shape[DAMACY_MAX_RANK];
  uint64_t inner_chunk_shape[DAMACY_MAX_RANK];
  uint64_t shard_shape[DAMACY_MAX_RANK]; // == outer chunk-grid shape
  int sharded;
  struct codec_config inner_codec;
  int index_location_end; // 1 = footer, 0 = header
  // Per-element fill bytes (size == dtype_bpe(dtype)). Defaults to all-zero
  // if the JSON has no fill_value; absent fill_value is logged as a warning.
  uint8_t fill_value[DAMACY_MAX_DTYPE_BYTES];
};

// Parse src into out. Returns 0 on success.
int
zarr_metadata_parse(const char* src, size_t src_len, struct zarr_metadata* out);

// Inner-chunks-per-shard along each dim, and the product across all
// dims. Either output array may be NULL if not needed; out_per_dim, when
// provided, must point to storage for at least meta->rank u64s.
//
// Returns 0 on success; non-zero if shard_shape isn't a clean multiple
// of inner_chunk_shape on some dim, or if the per-dim ratios overflow
// when accumulated into out_total.
int
zarr_metadata_inner_per_shard(const struct zarr_metadata* meta,
                              uint64_t* out_per_dim,
                              uint64_t* out_total);
