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
    uint32_t owner_capacity;
    // Declared upper bound on shards a single sample may intersect. A
    // sample whose AABB intersects more shards is rejected with
    // DAMACY_INVAL (the shard_index cache is sized for this bound, so
    // honoring it keeps cache saturation impossible by construction).
    uint32_t max_shards_per_sample;
  };

  enum prefetcher_result
  {
    PREFETCHER_RESULT_READY = 0,
    PREFETCHER_RESULT_ERROR,
  };

  struct prefetcher_stats
  {
    uint64_t submitted;
    uint64_t ready;
    uint64_t errored;
    uint32_t in_flight;
  };

  struct prefetcher* prefetcher_create(const struct prefetcher_config* cfg);
  // Teardown order: stop the prefetcher, drain the async metadata store, then
  // destroy the prefetcher and caches. Metadata callbacks may complete gates
  // owned by the prefetcher, so the prefetcher memory must outlive that drain.
  void prefetcher_destroy(struct prefetcher* p);

  int prefetcher_start(struct prefetcher* p);
  void prefetcher_stop(struct prefetcher* p); // idempotent

  // Blocks until the lookahead is empty and no slots are pending. Un-popped
  // READY/ERROR slots still occupy capacity, so draining without popping
  // can wedge the next push if capacity fills.
  enum damacy_status prefetcher_drain(struct prefetcher* p);

  // uri + h_shards become caller-owned; release via prefetcher_ready_free.
  struct prefetcher_ready
  {
    enum prefetcher_result result;
    int err_code; // damacy_status when result == PREFETCHER_RESULT_ERROR
    char* uri;
    struct damacy_aabb aabb;
    uint64_t sample_seq;
    struct prefetch_handle h_meta;
    // On ERROR mid-shard-allocation, h_shards may be non-NULL with n_shards
    // == 0; iterate by n_shards, never by (h_shards != NULL).
    struct prefetch_handle* h_shards;
    uint32_t n_shards;
    struct prefetch_handle h_layout;
  };

  struct prefetcher_wave_ticket
  {
    uint64_t sample_seq_begin;
    uint32_t n_samples;
  };

  // Returns 1 if the next sample_seq terminal-state slot was popped into
  // *out; 0 if the contiguous prefix is not currently available.
  int prefetcher_pop_ready(struct prefetcher* p, struct prefetcher_ready* out);

  // Atomically pops the contiguous ready/error prefix starting at the next
  // unconsumed sample sequence, capped by max_samples. Returns 1 if at least
  // one terminal sample was popped.
  int prefetcher_take_ready_wave(struct prefetcher* p,
                                 uint32_t max_samples,
                                 struct prefetcher_wave_ticket* ticket,
                                 struct prefetcher_ready* out);

  uint32_t prefetcher_ready_prefix_count(struct prefetcher* p);

  // Samples pushed by the caller but not yet consumed from the prefetcher
  // ready/error prefix by the planner. This is the push-side backpressure
  // budget, independent of whether those samples are still in lookahead, in
  // the worker's pop/admit window, or occupying prefetcher slots.
  uint64_t prefetcher_unconsumed_count(struct prefetcher* p,
                                       uint64_t pushed_samples);

  uint32_t prefetcher_in_flight(struct prefetcher* p);

  int prefetcher_has_ready(struct prefetcher* p);

  void prefetcher_ready_free(struct prefetcher_ready* r);

  // NULL if no such sample is admitted; valid until the sample is popped and
  // all cache waiters registered against it have resolved.
  const struct prefetch_gate* prefetcher_sample_gate(struct prefetcher* p,
                                                     uint64_t sample_seq);

  // Entries with max owner id < watermark become evictable.
  void prefetcher_advance_watermark(struct prefetcher* p, uint64_t watermark);

  void prefetcher_stats_get(const struct prefetcher* p,
                            struct prefetcher_stats* out);

#ifdef __cplusplus
}
#endif
