#pragma once

#include "prefetch/prefetch_cache.h"
#include "zarr/zarr_chunk_layout.h"

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;

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

#ifdef __cplusplus
}
#endif
