#pragma once

#include "damacy.h"
#include "prefetch/prefetch_cache.h"

#ifdef __cplusplus
extern "C"
{
#endif

  struct prefetcher;
  struct damacy_lookahead;

  struct prefetcher_config
  {
    struct damacy_lookahead* lookahead;
    struct prefetch_cache* array_meta_cache;
    struct prefetch_cache* shard_index_cache;
    struct prefetch_cache* chunk_layout_cache;
    uint32_t capacity;
    uint32_t batch_capacity;
  };

  enum prefetcher_sample_state
  {
    PREFETCHER_FREE = 0,
    PREFETCHER_PENDING_ARRAY_META,
    PREFETCHER_PENDING_SHARD_INDEX,
    PREFETCHER_PENDING_CHUNK_LAYOUT,
    PREFETCHER_READY,
    PREFETCHER_ERROR,
  };

  struct prefetcher_stats
  {
    uint64_t submitted;
    uint64_t ready;
    uint64_t errored;
    uint32_t in_flight;
  };

  struct prefetcher* prefetcher_create(const struct prefetcher_config* cfg);
  void prefetcher_destroy(struct prefetcher* p);

  int prefetcher_start(struct prefetcher* p);
  void prefetcher_stop(struct prefetcher* p); // idempotent

  // Blocks until the lookahead is empty and no slots are pending.
  enum damacy_status prefetcher_drain(struct prefetcher* p);

  // uri + h_shards become caller-owned; release via prefetcher_ready_free.
  struct prefetcher_ready
  {
    enum prefetcher_sample_state state; // READY or ERROR
    char* uri;
    struct damacy_aabb aabb;
    uint64_t batch_id;
    struct prefetch_handle h_meta;
    struct prefetch_handle* h_shards;
    uint32_t n_shards;
    struct prefetch_handle h_layout;
  };

  // Returns 1 if a terminal-state slot was popped into *out; 0 if no
  // terminal-state slot is currently available.
  int prefetcher_pop_ready(struct prefetcher* p, struct prefetcher_ready* out);

  void prefetcher_ready_free(struct prefetcher_ready* r);

  // NULL if no in-flight samples; valid until released or destroyed.
  const struct prefetch_gate* prefetcher_batch_gate(struct prefetcher* p,
                                                    uint64_t batch_id);

  // Marks the batch releasable; the entry is reclaimed once the last
  // referencing slot is popped (so the gate can't alias mid-flight).
  void prefetcher_release_batch(struct prefetcher* p, uint64_t batch_id);

  // Entries with max_batch_id < watermark become evictable.
  void prefetcher_advance_watermark(struct prefetcher* p, uint64_t watermark);

  void prefetcher_stats_get(const struct prefetcher* p,
                            struct prefetcher_stats* out);

#ifdef __cplusplus
}
#endif
