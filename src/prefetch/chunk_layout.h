#pragma once

#include "damacy_limits.h"
#include "prefetch/prefetch_cache.h"
#include "zarr/zarr_chunk_layout.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;

  struct chunk_layout_key
  {
    const char* uri;
    uint8_t rank;
    uint32_t n_shards;
    const uint64_t* shard_coords;           // flat [n_shards][rank]
    const struct prefetch_handle* h_shards; // [n_shards]
  };

  // Caller ensures upstream entries stay pinned across the fetch.
  struct chunk_layout_fetcher
  {
    struct prefetch_fetcher base;
    struct store* store;
    struct prefetch_cache* array_meta_cache;
    struct prefetch_cache* shard_index_cache;
    uint32_t max_substreams_per_chunk;
  };

  void chunk_layout_fetcher_init(struct chunk_layout_fetcher* f,
                                 struct store* store,
                                 struct prefetch_cache* array_meta_cache,
                                 struct prefetch_cache* shard_index_cache,
                                 uint32_t max_substreams_per_chunk);

  extern const struct prefetch_ops chunk_layout_ops;

  uint64_t chunk_layout_key_hash(const struct chunk_layout_key* k);

#ifdef __cplusplus
}
#endif
