// Public read-side store interface.
//
// Users create a store (filesystem now; S3 etc later), pass it to the zarr
// reader, and the reader uses it to fetch compressed chunk bytes.
//
// The store is opaque. Internally, store_fs owns an io_queue with
// configurable concurrency for parallel positional reads.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;

  struct store_fs_config
  {
    const char* root; // root directory for keys; not owned
    int nthreads;     // io_queue worker count (0 = synchronous)
    int use_o_direct; // O_DIRECT for compressed reads (advisory)
  };

  // Create a filesystem-backed store. Returns NULL on failure.
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

  // Block until all reads up to and including ev.seq have completed.
  void store_event_wait(struct store* s, struct store_event ev);

  // Submit + wait. Returns 0 on success.
  int store_read_many(struct store* s,
                      const struct store_read* reads,
                      size_t n);

  // Recover the size in bytes of the resource named by `key`. Returns 0 on
  // success and writes *out. Used by zarr_reader to find a shard footer.
  int store_stat(struct store* s, const char* key, uint64_t* out);

#ifdef __cplusplus
}
#endif
