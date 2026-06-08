#pragma once

#include "damacy.h"
#include "store/store.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct metadata_store_async;
  struct numa_resolved;

  typedef void (*metadata_store_read_cb)(void* user,
                                         enum damacy_status status,
                                         void* data,
                                         size_t len);

  typedef void (*metadata_store_stat_cb)(void* user,
                                         enum store_stat_result status,
                                         uint64_t size);

  struct metadata_store_async* metadata_store_async_create(
    int concurrency,
    const struct numa_resolved* affinity,
    const struct damacy_latency_model* latency);

  void metadata_store_async_destroy(struct metadata_store_async* s);

  struct metadata_store_async_latency_stats
  {
    uint64_t ops;
    uint64_t stat_ops;
    uint64_t submit_ops;
    uint64_t active;
    uint64_t max_active;
    uint64_t total_sleep_ns;
    uint64_t max_sleep_ns;
  };

  struct metadata_store_async_backend_stats
  {
    uint64_t read_jobs;
    uint64_t read_active;
    uint64_t read_max_active;
  };

// Log2-scale histogram of measured submit->completion latency. Bucket i holds
// ops whose latency in ns has floor(log2(ns)) == i (bucket 0 also catches 0 ns,
// the last bucket catches everything above). Percentiles are derived from the
// raw buckets at report time so the estimator can change without an ABI change.
#define METADATA_OP_LATENCY_NBUCKETS DAMACY_METADATA_OP_LATENCY_NBUCKETS
#define METADATA_OP_LATENCY_NKINDS DAMACY_METADATA_OP_LATENCY_NKINDS

  struct metadata_store_async_op_latency_kind
  {
    uint64_t count;
    uint64_t sum_ns;
    uint64_t max_ns;
    uint64_t buckets[METADATA_OP_LATENCY_NBUCKETS];
  };

  // Indexed by enum op_kind: statx, open, read, close.
  struct metadata_store_async_op_latency_stats
  {
    struct metadata_store_async_op_latency_kind
      kinds[METADATA_OP_LATENCY_NKINDS];
  };

  void metadata_store_async_latency_stats_get(
    struct metadata_store_async* s,
    struct metadata_store_async_latency_stats* out);
  void metadata_store_async_latency_stats_reset(struct metadata_store_async* s);

  void metadata_store_async_backend_stats_get(
    struct metadata_store_async* s,
    struct metadata_store_async_backend_stats* out);
  void metadata_store_async_backend_stats_reset(struct metadata_store_async* s);

  void metadata_store_async_op_latency_stats_get(
    struct metadata_store_async* s,
    struct metadata_store_async_op_latency_stats* out);
  void metadata_store_async_op_latency_stats_reset(
    struct metadata_store_async* s);

  int metadata_store_async_read_file(struct metadata_store_async* s,
                                     const char* key,
                                     metadata_store_read_cb cb,
                                     void* user);

  int metadata_store_async_read(struct metadata_store_async* s,
                                const char* key,
                                uint64_t offset,
                                size_t len,
                                metadata_store_read_cb cb,
                                void* user);

  int metadata_store_async_stat(struct metadata_store_async* s,
                                const char* key,
                                metadata_store_stat_cb cb,
                                void* user);

#ifdef __cplusplus
}
#endif
