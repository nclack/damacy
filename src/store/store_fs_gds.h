// GPUDirect Storage (cuFile) store. The LRU destroy callback
// deregisters each CUfileHandle_t BEFORE closing its backing fd; that
// ordering is required by cuFile and lives inside one local function.
//
// DAMACY_ENABLE_GDS=OFF links store_fs_gds_stub.c which makes
// store_fs_gds_create return NULL.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;

  struct store_fs_gds_config
  {
    const char* root;
    uint32_t fd_cache_capacity; // 0 → library default
  };

  struct store* store_fs_gds_create(const struct store_fs_gds_config* cfg);

  void store_fs_gds_set_stream(struct store* s, void* stream);

#ifdef __cplusplus
}
#endif
