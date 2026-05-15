// GPUDirect Storage (cuFile) backend for store_fs. libcufile is
// dlopen'd lazily at first store_fs_create(enable_gds=1); the handle
// is leaked at process exit (mirrors src/numa/numa.c). Per-FD cuFile
// handles are registered on first device read and deregistered when
// the cache slot's platform_file is closed.
#pragma once

#include <stddef.h>

struct store;
struct store_fs;
struct store_read;
struct store_event;

#ifdef __cplusplus
extern "C"
{
#endif

  // dlopen libcufile + cuFileDriverOpen. Sets fs->gds_driver_opened on
  // success. Returns 0 on success, non-zero on failure.
  int store_fs_gds_init(struct store_fs* fs);

  // No-op when GDS was never enabled.
  void store_fs_gds_destroy(struct store_fs* fs);

  struct store_event store_fs_gds_submit_dev(struct store* s,
                                             const struct store_read* reads,
                                             size_t n);

#ifdef __cplusplus
}
#endif
