// GPUDirect Storage (cuFile) backend for store_fs. Only compiled when
// DAMACY_ENABLE_GDS is ON. Wires submit_dev into the fs store's vtable
// and owns the per-FD cuFileHandle_t cache (lazy registration).
//
// Lifecycle: cuFileDriverOpen() at store_fs_create; cuFileDriverClose()
// at store_fs destroy. Per-FD handles registered on first device read
// and deregistered when the fs cache slot's platform_file is closed.
#pragma once

#include <stddef.h>
#include <stdint.h>

struct store;
struct store_fs;
struct store_read;
struct store_event;

#ifdef __cplusplus
extern "C"
{
#endif

  // Try to enable GDS on a freshly-created store_fs. Calls
  // cuFileDriverOpen, sets the submit_dev vtable slot on success, and
  // returns 0; returns non-zero (and leaves the store_fs unchanged) if
  // cuFile init fails. Idempotent fallback: callers that don't want
  // hard-fail behavior can ignore the return code.
  int store_fs_gds_init(struct store_fs* fs);

  // Tear down the cuFile driver and any registered handles. Safe to
  // call when GDS was never enabled (no-op).
  void store_fs_gds_destroy(struct store_fs* fs);

#ifdef __cplusplus
}
#endif
