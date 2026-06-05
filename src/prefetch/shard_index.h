#pragma once

#include "damacy_limits.h"
#include "prefetch/prefetch_cache.h"
#include "zarr/zarr_shard_index.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;
  struct metadata_store_async;

  struct shard_index_key
  {
    const char* uri;
    uint64_t shard_coord[DAMACY_MAX_RANK];
    uint8_t rank;
  };

  struct shard_index_value
  {
    struct zarr_shard_entry* entries;
    uint64_t n_entries;
  };

  // Caller ensures array_meta_cache entry pins outlast the fetch.
  struct shard_index_fetcher
  {
    struct prefetch_fetcher base;
    struct store* store;
    struct prefetch_cache* array_meta_cache;
  };

  struct shard_index_async_fetcher
  {
    struct prefetch_async_fetcher base;
    struct metadata_store_async* store;
    struct prefetch_cache* array_meta_cache;
  };

  void shard_index_fetcher_init(struct shard_index_fetcher* f,
                                struct store* store,
                                struct prefetch_cache* array_meta_cache);

  void shard_index_async_fetcher_init(struct shard_index_async_fetcher* f,
                                      struct metadata_store_async* store,
                                      struct prefetch_cache* array_meta_cache);

  extern const struct prefetch_ops shard_index_ops;

  uint64_t shard_index_key_hash(const struct shard_index_key* k);

#ifdef __cplusplus
}
#endif
