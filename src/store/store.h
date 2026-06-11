// Public read-side store interface.
//
// Users create a store (filesystem now; S3 etc later), pass it to the zarr
// reader, and the reader uses it to fetch compressed chunk bytes.
//
// Two access patterns:
//   - store_read_submit / store_event_wait: positional reads for
//     compressed chunk bytes. Submit returns status plus an event and
//     runs on the store's io_queue with configurable concurrency.
//   - store_map / store_unmap: whole-resource read-only view, intended
//     for metadata (zarr.json) where a parser wants a contiguous buffer.
#pragma once

#include "damacy.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;
  struct numa_resolved;

  struct store_fs_config
  {
    const char* root; // root directory for keys; not owned
    int nthreads;     // io_queue worker count (>= 1)
    // NULL (or node<0) skips affinity.
    const struct numa_resolved* affinity;
    // 0 → library default. Size against RLIMIT_NOFILE minus headroom for
    // non-cache fds; the cache holds one fd per entry.
    uint32_t fd_cache_capacity;
    // 0 → library default. Bounds simultaneous read jobs; running out
    // stalls submission and reissues a whole batch, so size for the
    // worst case. Jobs are small (~64 bytes) — err deep.
    uint32_t max_inflight_reads;
  };

  // Create a filesystem-backed store (host-staging only). Returns NULL on
  // failure. For GPUDirect Storage, see store_fs_gds_create in
  // store_fs_gds.h — that's a separate store implementation; this one
  // never references cuFile.
  struct store* store_fs_create(const struct store_fs_config* cfg);

  // Destroy a store created by store_fs_create or any future backend.
  void store_destroy(struct store* s);

  // One positional read against the store. `key` is interpreted by the
  // backend (FS: relative path under root). `dst` must be writable for at
  // least `len` bytes; the caller owns the memory and is responsible for
  // pinning it if it will later be used as a CUDA host-to-device source.
  struct store_read
  {
    const char* key;
    void* dst;
    uint64_t offset;
    size_t len;
  };

  // Non-NULL `impl` owns a backend ref; drive to completion via
  // event_wait or event_query (until ready), else event_discard.
  struct store_event
  {
    uint64_t seq;
    void* impl;
  };

  struct store_submit_result
  {
    enum damacy_status status;
    struct store_event event;
  };

  struct store_event_poll
  {
    enum damacy_status status;
    uint8_t ready;
  };

  // Submit a batch of reads. On success, result.event advances after every
  // read in the batch has completed. Reads run on the store's io_queue.
  struct store_submit_result store_read_submit(struct store* s,
                                               const struct store_read* reads,
                                               size_t n);

  // Like store_read_submit but each `reads[i].dst` is a device pointer.
  // The store routes the read directly into GPU memory when supported
  // (NVIDIA GDS / cuFile).
  struct store_submit_result store_read_submit_dev(
    struct store* s,
    const struct store_read* reads,
    size_t n);

  // 1 if the store can satisfy store_read_submit_dev (cuFile driver
  // initialized for fs stores); 0 otherwise.
  int store_supports_gds(struct store* s);

  // Block until all reads up to and including ev.seq have completed.
  // Reclaims the backend ref in `ev`; do not call event_discard after.
  // Caller must ensure the store's stream remains live and is progressing;
  // a destroyed or permanently-stalled stream will deadlock the wait.
  enum damacy_status store_event_wait(struct store* s, struct store_event ev);

  // Non-blocking variant of store_event_wait. ready=1 means every read up
  // to ev.seq has completed and status is final. ready=1 reclaims the backend
  // ref in `ev`; do not call event_discard after.
  struct store_event_poll store_event_query(struct store* s,
                                            struct store_event ev);

  // Release `ev` without waiting on completion. Required if neither
  // event_wait nor event_query-to-completion is called on it. Safe on
  // any event with NULL impl (zero-initialized events or n==0 submits).
  void store_event_discard(struct store* s, struct store_event ev);

  // Submit + wait. Returns 0 on success.
  int store_read_many(struct store* s,
                      const struct store_read* reads,
                      size_t n);

  // NOT_FOUND is reserved for absent resources; other failures are ERROR
  // so callers don't alias them to "absent."
  enum store_stat_result
  {
    STORE_STAT_OK = 0,
    STORE_STAT_NOT_FOUND,
    STORE_STAT_ERROR,
  };

  enum store_stat_result store_stat(struct store* s,
                                    const char* key,
                                    uint64_t* out);

  // Read-only view of a whole resource. `data`/`len` are public; the
  // remaining fields are backend-private — treat them as opaque.
  struct store_view
  {
    const void* data;
    size_t len;
    void* backend; // backend cookie (e.g. mmap bookkeeping)
  };

  // Map a resource read-only over its full extent. Returns 0 on success
  // and populates *out. Intended for metadata: the buffer must outlive
  // the parser's pointers into it, but parsed POD structs should not
  // retain any references — call store_unmap as soon as parsing is done.
  int store_map(struct store* s, const char* key, struct store_view* out);

  // Release a view returned by store_map. Safe to call on a zeroed view.
  void store_unmap(struct store* s, struct store_view* view);

#ifdef __cplusplus
}
#endif
