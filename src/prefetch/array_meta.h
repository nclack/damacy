#pragma once

#include "prefetch/prefetch_cache.h"

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;
  struct metadata_store_async;

  struct array_meta_fetcher
  {
    struct prefetch_fetcher base;
    struct store* store;
  };

  struct array_meta_async_fetcher
  {
    struct prefetch_async_fetcher base;
    struct metadata_store_async* store;
  };

  void array_meta_fetcher_init(struct array_meta_fetcher* f,
                               struct store* store);

  void array_meta_async_fetcher_init(struct array_meta_async_fetcher* f,
                                     struct metadata_store_async* store);

  extern const struct prefetch_ops array_meta_ops;

#ifdef __cplusplus
}
#endif
