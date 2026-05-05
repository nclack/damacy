// LRU cache of parsed zarr v3 array metadata, keyed by URI (path within
// a store). Each cached entry holds the parsed `zarr_metadata` struct
// and the URI string used to fetch it.
//
// The cache borrows the store; the caller must keep the store alive for
// at least as long as the cache.
#pragma once

#include "damacy.h"   // damacy_status
#include "util/lru.h" // struct lru_counters
#include "zarr/zarr_metadata.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;
  struct zarr_meta_cache;

  // Create a metadata cache backed by `store`, holding up to `capacity`
  // entries. Returns NULL on alloc failure or invalid args.
  struct zarr_meta_cache* zarr_meta_cache_create(struct store* store,
                                                 uint32_t capacity);

  void zarr_meta_cache_destroy(struct zarr_meta_cache* c);

  // Look up metadata for the array at `uri` (path within the store). On
  // hit returns the cached pointer; on miss, fetches and parses
  // <uri>/zarr.json from the store, caches the result, and returns it.
  //
  // The returned pointer is owned by the cache; valid until the entry
  // is evicted (which can happen on a subsequent get). v1: no
  // pinning — callers should not retain the pointer across other cache
  // operations.
  //
  // Returns DAMACY_OK on success; DAMACY_INVAL on bad args;
  // DAMACY_NOTFOUND if zarr.json could not be opened; DAMACY_DECODE if
  // the metadata is malformed; DAMACY_OOM on alloc failure or full
  // cache with all entries pinned.
  enum damacy_status zarr_meta_cache_get(struct zarr_meta_cache* c,
                                         const char* uri,
                                         const struct zarr_metadata** out);

  struct zarr_meta_cache_stats
  {
    struct lru_counters counters;
    uint32_t size;
    uint32_t capacity;
  };

  void zarr_meta_cache_stats_get(const struct zarr_meta_cache* c,
                                 struct zarr_meta_cache_stats* out);

#ifdef __cplusplus
}
#endif
