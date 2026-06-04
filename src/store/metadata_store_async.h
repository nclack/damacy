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
    struct store* store,
    int concurrency,
    const struct numa_resolved* affinity);

  void metadata_store_async_destroy(struct metadata_store_async* s);

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
