// Public read-side store interface.
//
// Users create a store (filesystem now; S3 etc later), pass it to the zarr
// reader, and the reader uses it to fetch compressed chunk bytes.
//
// Two access patterns:
//   - store_read_submit / store_event_wait: positional reads for
//     compressed chunk bytes. Runs on the store's io_queue with
//     configurable concurrency. Each submitted read currently maps 1:1
//     to a `pread` on a cached FD; planner-side coalescing of adjacent
//     reads is on the roadmap (plan.md step 7).
//   - store_map / store_unmap: whole-resource read-only view, intended
//     for metadata (zarr.json) where a parser wants a contiguous buffer.
#pragma once

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
  // pinning it if it will later be used as a CUDA H2D source.
  struct store_read
  {
    const char* key;
    void* dst;
    uint64_t offset;
    size_t len;
  };

  // Completion handle returned by store_read_submit. Wait via
  // store_event_wait; cheap value type.
  struct store_event
  {
    uint64_t seq;
  };

  // Submit a batch of reads. Returns an event whose .seq advances after
  // *every* read in the batch has completed (regardless of the order
  // workers happened to finish them). Reads run on the store's io_queue.
  struct store_event store_read_submit(struct store* s,
                                       const struct store_read* reads,
                                       size_t n);

  // Like store_read_submit but each `reads[i].dst` is a device pointer.
  // The store routes the read directly into GPU memory when supported
  // (NVIDIA GDS / cuFile); otherwise returns an event with seq == 0
  // indicating failure. Callers should query store_supports_gds first
  // and fall back to store_read_submit + an H2D copy when unsupported.
  struct store_event store_read_submit_dev(struct store* s,
                                           const struct store_read* reads,
                                           size_t n);

  // 1 if the store can satisfy store_read_submit_dev (cuFile driver
  // initialized for fs stores); 0 otherwise.
  int store_supports_gds(struct store* s);

  // Block until all reads up to and including ev.seq have completed.
  void store_event_wait(struct store* s, struct store_event ev);

  // Non-blocking variant of store_event_wait. Returns non-zero if every
  // read up to ev.seq has completed.
  int store_event_query(struct store* s, struct store_event ev);

  // Submit + wait. Returns 0 on success.
  int store_read_many(struct store* s,
                      const struct store_read* reads,
                      size_t n);

  // Recover the size in bytes of the resource named by `key`. Returns 0 on
  // success and writes *out. Used by the shard-index cache to find a
  // shard footer.
  int store_stat(struct store* s, const char* key, uint64_t* out);

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
