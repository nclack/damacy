// Store wrapper that injects deterministic synthetic latency.
#pragma once

#include "damacy.h"

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;

  struct store_latency_stats
  {
    uint64_t ops;
    uint64_t map_ops;
    uint64_t stat_ops;
    uint64_t submit_ops;
    uint64_t submit_dev_ops;
    uint64_t active;
    uint64_t max_active;
    uint64_t total_sleep_ns;
    uint64_t max_sleep_ns;
  };

  // Non-owning: store_destroy() on the returned wrapper frees only the
  // wrapper. The caller keeps ownership of `inner`.
  struct store* store_latency_create(
    struct store* inner,
    const struct damacy_latency_model* latency);

  void store_latency_stats_get(struct store* s, struct store_latency_stats* out);
  void store_latency_stats_reset(struct store* s);

#ifdef __cplusplus
}
#endif
