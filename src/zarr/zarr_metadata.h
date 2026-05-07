// Internal: parse a zarr v3 array's zarr.json into a flat metadata struct.
#pragma once

#include "damacy_limits.h"
#include "dtype/dtype.h"

#include <stdint.h>

// Inner-codec subset we actually parse out of zarr.json. The blosc
// variants are inner-codec-resolved at parse time from the
// `configuration.cname` of a "blosc" codec entry — by the time the tag
// reaches the decoder we already know which nvcomp batched call to
// dispatch. Per-chunk blosc parameters (typesize, shuffle, blocksize)
// are NOT carried here; they live in the chunk's blosc1 header bytes
// and are read at wave-decode time.
enum compression_codec
{
  CODEC_NONE = 0,
  CODEC_ZSTD = 1,
  CODEC_BLOSC_LZ4 = 2,  // blosc1 wrapper, inner = lz4 / lz4hc (compformat 1)
  CODEC_BLOSC_ZSTD = 3, // blosc1 wrapper, inner = zstd          (compformat 4)
};

struct codec_config
{
  enum compression_codec id;
};

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
