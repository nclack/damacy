// Read-side dimension descriptor. Mirrors chucky's struct shape so the two
// projects can share types if we ever lift these into a common package.
//
// Damacy's reader populates dimensions from parsed zarr.json:
//   size              <- "shape"[i]
//   chunk_size        <- inner chunk shape (under sharding codec,
//   "chunk_shape") chunks_per_shard  <- "shard_shape"[i] / chunk_shape[i]   (1
//   if unsharded) storage_position  <- 0..rank-1 identity (transpose codec is
//   v1+) name              <- optional axis name (heap-duplicated from JSON)
//
// downsample is unused on the read side and stays zero.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct dimension
  {
    uint64_t size;             // 0 means unbounded; not expected on read side
    uint64_t chunk_size;       // inner chunk extent
    uint64_t chunks_per_shard; // 1 when unsharded
    const char* name;          // optional, heap-owned when non-NULL
    int downsample;            // unused on read side
    uint8_t storage_position;  // identity in v0
  };

  // Deep-copy rank dimensions from src to dst, duplicating name strings.
  // After success, dst owns its name strings; free with dims_free_names.
  int dims_copy(struct dimension* dst,
                const struct dimension* src,
                uint8_t rank);

  // Free name strings previously allocated by dims_copy (or any caller that
  // heap-allocated them). Safe on zero-initialized slots.
  void dims_free_names(struct dimension* dims, uint8_t rank);

  // Number of chunks along dimension d (ceil(size / chunk_size)).
  uint64_t dim_n_chunks(const struct dimension* d);

  // Shard shape for dimension d (chunk_size * chunks_per_shard).
  uint64_t dim_shard_extent(const struct dimension* d);

#ifdef __cplusplus
}
#endif
