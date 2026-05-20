// Generic LRU cache with refcount-pinning.
//
// Implementation: fixed-size slot array (allocated at create) keyed by a
// robin-hood hash table over slot indices. The hash table size is the
// next power of two >= 2 * capacity, so load factor stays <= 50%. On
// insert the probe distance is bounded by `max_probe`; if a chain is
// full and probe-too-deep, the LRU tail (skipping pinned entries) is
// evicted to make room. Slot pointers (`struct lru_entry*`) are stable
// for the lifetime of an entry — robin hood reorders the index, not
// the slots themselves.
//
// Caller responsibilities:
//   - Compute the hash for each value before put/get. The hash must be
//     stable across put/get of the same logical key.
//   - Provide `eq(value, probe_key)` to disambiguate hash collisions.
//   - Provide `destroy(value)` to free values on eviction or
//     lru_destroy.
//
// The LRU does not interpret keys at all — it only ever sees `void*`
// pointers (values) and `void*` probe keys (passed by callers and
// forwarded to ops.eq).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct lru;
  struct lru_entry;

  struct lru_ops
  {
    // Returns nonzero if `value` (currently stored in a slot) matches
    // `probe_key` (the lookup key the caller passed to lru_get/lru_put).
    int (*eq)(const void* value, const void* probe_key, void* user);
    // Called when the LRU evicts a value, on collision-replace, and on
    // lru_destroy. Caller's value ownership transfers to the LRU on put;
    // destroy releases it.
    void (*destroy)(void* value, void* user);
    void* user;
  };

  // Create an LRU. capacity = max simultaneous entries. max_probe = the
  // hard cap on robin-hood probe distance (typical: log2(capacity)+4).
  // Returns NULL on alloc failure.
  struct lru* lru_create(uint32_t capacity,
                         uint32_t max_probe,
                         const struct lru_ops* ops);

  // Tear down. Calls ops.destroy on every value still in the cache.
  void lru_destroy(struct lru* l);

  // Look up by hash + probe key. Returns NULL on miss. On hit, the
  // entry is promoted to MRU.
  struct lru_entry* lru_get(struct lru* l,
                            uint64_t hash,
                            const void* probe_key);

  // Like lru_get but does not promote the entry to MRU and does not
  // update hit/miss counters. Intended for race-recheck paths where
  // counting the probe as a real lookup would skew stats.
  struct lru_entry* lru_peek(struct lru* l,
                             uint64_t hash,
                             const void* probe_key);

  // Insert. Takes ownership of `value`: it will be passed to
  // ops.destroy on eviction, replacement (matching key), or
  // lru_destroy. `probe_key` is used to detect a matching existing
  // entry via ops.eq (same shape as lru_get). If a matching entry is
  // found, the old value is destroyed and replaced. If the cache is
  // full and no non-pinned entry can be evicted, returns NULL and the
  // caller's `value` is destroyed via ops.destroy.
  //
  // Preconditions: l != NULL (passing NULL leaks `value` since the
  // destroy callback lives on l) and value != NULL.
  struct lru_entry* lru_put(struct lru* l,
                            uint64_t hash,
                            const void* probe_key,
                            void* value);

  // Stable value pointer for the entry.
  void* lru_entry_value(const struct lru_entry* e);

  // Refcount-pinning. Pinned entries are skipped during eviction.
  // Acquire/release must be balanced. _locked must be called under the
  // same external mutex that gates lru_get/lru_put/lru_destroy — the
  // mutex is what serializes the 0->1 transition against eviction.
  // Release is lock-free and may run outside that mutex.
  void lru_entry_acquire_locked(struct lru_entry* e);
  void lru_entry_release(struct lru_entry* e);

  // Cumulative event counters. Embedded in lru_stats; also the type
  // the LRU itself stores internally so stats_get is a single
  // struct-assign of the running totals.
  struct lru_counters
  {
    uint64_t hits;
    uint64_t misses;
    uint64_t insertions;
    uint64_t evictions;
    uint64_t replacements; // put with matching key replaced existing
    uint64_t put_failures; // returned NULL because all entries pinned
  };

  struct lru_stats
  {
    struct lru_counters counters;
    uint32_t size;
    uint32_t capacity;
    uint32_t pinned;
    uint32_t max_probe_observed;
  };

  void lru_stats_get(const struct lru* self, struct lru_stats* out);

#ifdef __cplusplus
}
#endif
