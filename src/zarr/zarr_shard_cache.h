// LRU cache of parsed zarr v3 shard indices, keyed by (uri, shard_coord).
//
// The cache borrows the store; the caller must keep the store alive for
// at least as long as the cache. Metadata for each array (shard layout,
// index location) is passed in by the caller so this layer doesn't
// need to reach into the metadata cache.
#pragma once

#include "damacy.h"   // damacy_status
#include "util/lru.h" // struct lru_counters
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_index.h" // struct zarr_shard_entry

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;
  struct zarr_shard_cache;

  struct zarr_shard_cache* zarr_shard_cache_create(struct store* store,
                                                   uint32_t capacity);

  void zarr_shard_cache_destroy(struct zarr_shard_cache* c);

  // Look up the parsed shard index for the shard at `shard_coord`
  // within the array at `uri` (described by `meta`). On miss, fetches
  // the index from the shard file (location per meta->index_location_end)
  // via the store and caches it. shard_coord has meta->rank entries.
  //
  // Thread-safe. Returns a borrow: (entries, n_entries) are owned by
  // the cache and remain valid until the entry is evicted. v1 has no
  // pin/release API, so size `capacity` above the working set to keep
  // returned pointers usable across long operations.
  enum damacy_status zarr_shard_cache_get(
    struct zarr_shard_cache* c,
    const char* uri,
    const struct zarr_metadata* meta,
    const uint64_t* shard_coord,
    const struct zarr_shard_entry** out_entries,
    uint64_t* out_n_entries);

  struct zarr_shard_cache_stats
  {
    struct lru_counters counters;
    uint32_t size;
    uint32_t capacity;
  };

  void zarr_shard_cache_stats_get(const struct zarr_shard_cache* c,
                                  struct zarr_shard_cache_stats* out);

#ifdef __cplusplus
}
#endif
