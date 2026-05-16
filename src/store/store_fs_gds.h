// GPUDirect Storage (cuFile) backend for store_fs. libcufile is
// dlopen'd lazily at first store_fs_create(enable_gds=1); the handle
// is leaked at process exit (mirrors src/numa/numa.c). Per-FD cuFile
// handles are registered on first device read and deregistered when
// the cache slot's platform_file is closed.
//
// Submissions go through cuFileReadAsync on the wave_pool's stream_h2d:
// the read is stream-ordered with the parse + decode kernels that
// follow on the same stream, so the SLOT_IO → SLOT_READY transition
// is bookkeeping (the actual wait happens via stream events
// downstream). The submit returns a sentinel event whose query / wait
// short-circuit to ready.
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

  // Sentinel returned by store_fs_gds_submit_dev. fs_event_query /
  // fs_event_wait recognize this and treat the event as ready without
  // touching io_queue (which is not used on the GDS submit path).
  // io_queue assigns sequence numbers monotonically from 1; UINT64_MAX
  // is well past anything io_queue can reach.
#define STORE_FS_GDS_SENTINEL_SEQ ((uint64_t)~(uint64_t)0)

  // dlopen libcufile + cuFileDriverOpen. Sets fs->gds_driver_opened on
  // success. Returns 0 on success, non-zero on failure.
  int store_fs_gds_init(struct store_fs* fs);

  // Hand the store the CUstream that cuFileReadAsync should ride on.
  // Caller (wave_pool) owns the stream. The store does not destroy it.
  void store_fs_gds_set_stream(struct store* s, void* stream);

  // No-op when GDS was never enabled.
  void store_fs_gds_destroy(struct store_fs* fs);

  struct store_event store_fs_gds_submit_dev(struct store* s,
                                             const struct store_read* reads,
                                             size_t n);

#ifdef __cplusplus
}
#endif
