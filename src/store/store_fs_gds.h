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
  struct lru_stats;

  struct store_fs_gds_config
  {
    const char* root;
    uint32_t fd_cache_capacity; // 0 → library default
  };

  struct store* store_fs_gds_create(const struct store_fs_gds_config* cfg);

  // Adopt `stream` for subsequent store_read_submit_dev calls. The store
  // registers it with cuFile (warming compat-mode per-stream state, see
  // issue #118). Pass NULL to detach. Calling with the currently-adopted
  // stream is a no-op. Callers MUST call store_destroy(s) before
  // cuStreamDestroy(stream) so the cuFile deregister happens first;
  // cuFile rejects cuStreamDestroy on a still-registered stream.
  void store_fs_gds_set_stream(struct store* s, void* stream);

  void store_fs_gds_stats_get(struct store* s, struct lru_stats* out);

#ifdef __cplusplus
}
#endif
