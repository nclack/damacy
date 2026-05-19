// LRU cache of parsed zarr v3 array metadata, keyed by URI (path within
// a store). Each cached entry holds the parsed `zarr_metadata` struct
// and the URI string used to fetch it.
//
// CONTRACT: URI keys are compared by pointer identity, not strcmp. The
// caller MUST pass a path_intern-acquired pointer; a raw strdup will
// miss every lookup and pollute the LRU. The contract is not statically
// enforced — `const char*` can't distinguish interned from non-interned.
// When `uris` is supplied to _create, the cache asserts ownership in
// debug builds; under NDEBUG the asserts vanish.
//
// The cache borrows the store and `uris`; both must outlive the cache.
#pragma once

#ifdef __cplusplus
_Static_assert(0,
               "zarr_meta_cache.h is C-only (atomic observer slot uses "
               "_Atomic). For C++/CUDA TUs, forward-declare "
               "`struct zarr_meta_cache` instead.");
#endif

#include "damacy.h"   // damacy_status
#include "util/lru.h" // struct lru_counters
#include "zarr/zarr_chunk_layout.h"
#include "zarr/zarr_metadata.h"

#include <stdatomic.h>
#include <stdint.h>

struct path_intern;
struct store;
struct zarr_meta_cache;

// `uris` is the path_intern that owns every URI key; pass NULL to
// disable the debug-only ownership assertion (legacy tests).
struct zarr_meta_cache*
zarr_meta_cache_create(struct store* store,
                       struct path_intern* uris,
                       uint32_t capacity);

void
zarr_meta_cache_destroy(struct zarr_meta_cache* c);

// Wires a slot the cache bumps via atomic_u16_observe_max on every
// first probe / layout_set.
void
zarr_meta_cache_set_blosc_nblocks_observer(struct zarr_meta_cache* c,
                                           _Atomic(uint16_t)* observer);

// Look up metadata for the array at `uri` (path within the store). On
// hit copies the cached metadata into *out; on miss, fetches and parses
// <uri>/zarr.json from the store, caches the result, and copies it.
//
// Thread-safe. The copy happens under the internal mutex so *out is
// independent of any subsequent eviction — the caller owns the bytes
// and the cache's entry may be evicted at any time after return.
//
// Returns DAMACY_OK on success; DAMACY_INVAL on bad args;
// DAMACY_NOTFOUND if zarr.json could not be opened; DAMACY_DECODE if
// the metadata is malformed; DAMACY_OOM on alloc failure.
enum damacy_status
zarr_meta_cache_get(struct zarr_meta_cache* c,
                    const char* uri,
                    struct zarr_metadata* out);

// Copies the cached chunk layout for `uri` into *out. Returns 0 on
// success, non-zero if the URI isn't cached or no layout has been
// probed yet. The cache must already hold a meta entry for the URI
// — call zarr_meta_cache_get first. Thread-safe; the copy happens
// under the cache mutex, so *out is independent of subsequent
// evictions.
int
zarr_meta_cache_layout_get(struct zarr_meta_cache* c,
                           const char* uri,
                           struct chunk_layout* out);

// Records a probed layout against the meta entry for `uri`. No-op if
// a layout has already been set (the first probe wins). Returns 0 on
// success. Thread-safe.
int
zarr_meta_cache_layout_set(struct zarr_meta_cache* c,
                           const char* uri,
                           const struct chunk_layout* layout);

// Cached probe: copies the cached layout into *out if present,
// otherwise reads 16 bytes at first_chunk_off in shard_path via the
// cache's store, caches the result, and copies it into *out. The
// copy happens under the cache mutex, so the caller's *out is
// independent of subsequent cache mutations. Returns 0 on success,
// non-zero on probe failure or if no meta entry exists for `uri`.
// Thread-safe; the underlying I/O runs unlocked, so concurrent
// first-probes for the same URI may each issue a read.
int
zarr_meta_cache_probe_layout(struct zarr_meta_cache* c,
                             const char* uri,
                             const char* shard_path,
                             uint64_t first_chunk_off,
                             uint32_t first_chunk_cbytes,
                             uint8_t codec_id,
                             struct chunk_layout* out);

struct zarr_meta_cache_stats
{
  struct lru_counters counters;
  uint32_t size;
  uint32_t capacity;
};

void
zarr_meta_cache_stats_get(const struct zarr_meta_cache* c,
                          struct zarr_meta_cache_stats* out);
