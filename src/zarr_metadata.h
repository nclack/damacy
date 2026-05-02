// Internal: parse a zarr v3 array's zarr.json into a flat metadata struct.
#pragma once

#include "dtype.h"
#include "limits.h"
#include "types.codec.h"

#include <stdint.h>

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
};

// Parse src into out. Returns 0 on success.
int
zarr_metadata_parse(const char* src, size_t src_len, struct zarr_metadata* out);
