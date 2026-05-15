// LRU cache of parsed zarr v3 array metadata, keyed by URI (path within
// a store). Each cached entry holds the parsed `zarr_metadata` struct
// and the URI string used to fetch it.
//
// The cache borrows the store; the caller must keep the store alive for
// at least as long as the cache.
#pragma once

#include "damacy.h"   // damacy_status
#include "util/lru.h" // struct lru_counters
#include "zarr/zarr_chunk_layout.h"
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

  // Returns the cached chunk layout for `uri`, or NULL if no layout has
  // been probed yet (or the URI isn't cached). The cache must already
  // hold a meta entry for the URI — call zarr_meta_cache_get first.
  const struct chunk_layout* zarr_meta_cache_layout_get(
    struct zarr_meta_cache* c,
    const char* uri);

  // Records a probed layout against the meta entry for `uri`. No-op if
  // a layout has already been set (the first probe wins; uniformity is
  // asserted per-chunk by the host parser). Returns 0 on success.
  int zarr_meta_cache_layout_set(struct zarr_meta_cache* c,
                                 const char* uri,
                                 const struct chunk_layout* layout);

  // Cached probe: returns the cached layout if present, otherwise reads
  // 16 bytes at first_chunk_off in shard_path via the cache's store,
  // caches the result, and returns the cached pointer. Returns NULL on
  // probe failure or if no meta entry exists for `uri`.
  const struct chunk_layout* zarr_meta_cache_probe_layout(
    struct zarr_meta_cache* c,
    const char* uri,
    const char* shard_path,
    uint64_t first_chunk_off,
    uint32_t first_chunk_cbytes,
    uint8_t codec_id);

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
