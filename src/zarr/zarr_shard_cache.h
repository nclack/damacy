// LRU cache of parsed zarr v3 shard indices, keyed by (uri, shard_coord).
//
// CONTRACT: the URI portion of the key is compared by pointer identity.
// Callers MUST pass a path_intern-acquired pointer for `uri`; a raw
// strdup will miss every lookup and pollute the LRU. `shard_coord` stays
// content-keyed.
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

  // Opaque pin handle. Returned by zarr_shard_cache_get; the caller
  // must pass it back to zarr_shard_cache_release once they are done
  // reading the returned (entries, n_entries). A zero-initialized pin
  // (`.opaque = NULL`) is a no-op for release — safe to release an
  // un-acquired pin.
  struct zarr_shard_pin
  {
    void* opaque;
  };

  struct zarr_shard_cache* zarr_shard_cache_create(struct store* store,
                                                   uint32_t capacity);

  void zarr_shard_cache_destroy(struct zarr_shard_cache* c);

  // Look up the parsed shard index for the shard at `shard_coord`
  // within the array at `uri` (described by `meta`). On miss, fetches
  // the index from the shard file (location per meta->index_location_end)
  // via the store and caches it. shard_coord has meta->rank entries.
  //
  // On DAMACY_OK, the cache pins the returned entry — *out_entries /
  // *out_n_entries are guaranteed valid until the caller passes
  // *out_pin to zarr_shard_cache_release. On any non-OK status,
  // *out_pin is left zero (no release needed but a release call is
  // still safe).
  //
  // Thread-safe; the pin survives concurrent eviction (eviction skips
  // pinned slots). Size `capacity` above the working set so put
  // failures don't fire under contention.
  enum damacy_status zarr_shard_cache_get(
    struct zarr_shard_cache* c,
    const char* uri,
    const struct zarr_metadata* meta,
    const uint64_t* shard_coord,
    struct zarr_shard_pin* out_pin,
    const struct zarr_shard_entry** out_entries,
    uint64_t* out_n_entries);

  // Release a pin acquired via zarr_shard_cache_get. Safe to call with
  // a zero-initialized pin (no-op). Must NOT be called more than once
  // per acquired pin.
  void zarr_shard_cache_release(struct zarr_shard_cache* c,
                                struct zarr_shard_pin pin);

  struct zarr_shard_cache_stats
  {
    struct lru_counters counters;
    uint32_t size;
    uint32_t capacity;
  };

  void zarr_shard_cache_stats_get(struct zarr_shard_cache* c,
                                  struct zarr_shard_cache_stats* out);

#ifdef __cplusplus
}
#endif
