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
    uint64_t map_ops;
    uint64_t stat_ops;
    uint64_t submit_ops;
    uint64_t submit_dev_ops;
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

  void metadata_store_async_latency_stats_get(
    struct metadata_store_async* s,
    struct metadata_store_async_latency_stats* out);
  void metadata_store_async_latency_stats_reset(struct metadata_store_async* s);

  void metadata_store_async_backend_stats_get(
    struct metadata_store_async* s,
    struct metadata_store_async_backend_stats* out);
  void metadata_store_async_backend_stats_reset(struct metadata_store_async* s);

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
