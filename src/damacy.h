// damacy public C API — high-throughput streaming loader for batches
// assembled from many sharded NGFF zarr stores.
//
// See dev/api-design-surface-draft.md for the rationale and discussion;
// dev/api-design-internals-draft.md describes the implementation shape.
//
// Threading model: a single user thread should own a `damacy*` instance
// and drive push/pop/flush on it. damacy_release is safe to call from
// another thread holding a damacy_batch* (a typical PyTorch-DataLoader
// shape: dataloader thread pops, training thread releases).
#pragma once

#include "damacy_limits.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Output dtype of an assembled batch. Sources can vary (per-zarr); the
  // assemble kernel casts each source element to this destination type.
  enum damacy_dtype
  {
    DAMACY_F32,
    DAMACY_BF16,
  };

  enum damacy_status
  {
    DAMACY_OK = 0,
    DAMACY_AGAIN,    // non-blocking call would block (queue full)
    DAMACY_INVAL,    // bad arguments
    DAMACY_NOTFOUND, // uri unresolvable / not a zarr
    DAMACY_DTYPE,    // zarr source dtype has no cast path to cfg.dtype
    DAMACY_RANK,     // sample rank incompatible with zarr rank
    DAMACY_IO,       // read/open failure on a shard file
    DAMACY_DECODE,   // codec parse / decompression failure
    DAMACY_CUDA,     // driver/runtime call failed
    DAMACY_OOM,      // would exceed a configured cap
    DAMACY_SHUTDOWN, // pipeline destroyed or in failed state
  };

  // Human-readable name for a status code; safe for log/error messages.
  const char* damacy_status_str(enum damacy_status s);

  // Half-open [beg, end) interval along one axis, in level-0 voxel indices.
  struct damacy_interval
  {
    int64_t beg, end;
  };

  // Variable-rank AABB. Only dims[0..rank) are read; axis order matches
  // the zarr's stored axis order. rank <= DAMACY_MAX_RANK for sample
  // AABBs. Storage is sized DAMACY_MAX_RANK + 1 to accommodate internal
  // dst AABBs (sample rank + 1 leading batch axis); see chunk_plan.dst.
  struct damacy_aabb
  {
    struct damacy_interval dims[DAMACY_MAX_RANK + 1];
    uint8_t rank;
  };

  // One sample request.
  struct damacy_sample
  {
    const char* uri; // null-terminated; copied internally
    struct damacy_aabb aabb;
  };

  // Caller-owned slice of samples to push.
  struct damacy_sample_slice
  {
    const struct damacy_sample* beg;
    const struct damacy_sample* end;
  };

  // All resource caps fixed at create-time. Nothing grows after this.
  // Output batches are double-buffered (B=2); waves are double-buffered
  // internally. Neither is configurable.
  struct damacy_config
  {
    // Batch geometry
    uint32_t batch_size;        // samples per batch
    uint32_t lookahead_batches; // user-push queue depth (>= 2)
    uint32_t n_io_threads;
    // Workers for host-side blosc1 chunk-header parsing. 0 = serial on
    // the calling thread; total parallelism is n_compute_threads + 1.
    uint32_t n_compute_threads;

    // Streaming buffers (split in half across two wave slots internally)
    uint64_t host_buffer_bytes;   // pinned staging; sized for IO bw
    uint64_t device_buffer_bytes; // device decompress scratch

    // LRU caps (FDs are cached per-key by the fs store; not bounded here).
    uint32_t n_zarrs_meta_cache;
    uint32_t n_shards_meta_cache;

    // Destination dtype of assembled batches. Source zarrs may carry any
    // supported integer or float type; the assemble kernel casts each
    // element to this dtype (RNE float-promote, no overflow handling —
    // precision is bounded by the destination's mantissa). Sources
    // without a cast path error with DAMACY_DTYPE at push.
    enum damacy_dtype dtype;

    // Largest uncompressed chunk size damacy will accept. nvcomp temp
    // scratch is sized for this; chunks exceeding this at planner are
    // rejected with DAMACY_INVAL. 0 means "use
    // DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES (512 KB)"; values
    // exceeding DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES (the kernel-array
    // ceiling, 2 MB) are rejected at create time.
    uint32_t max_chunk_uncompressed_bytes;

    // Hard cap on total GPU memory damacy will allocate for wave-resident
    // buffers and per-batch metadata. damacy_create returns DAMACY_OOM
    // (with a log_error breakdown) if the wave config would exceed this.
    // damacy_push returns DAMACY_OOM if the lazily-sized batch-output
    // pool, computed from the first sample's AABB, would push past it.
    // 0 = no cap (driver OOM, current behaviour).
    uint64_t max_gpu_memory_bytes;

    // -1 captures current CUcontext; >= 0 retains the primary for that
    // device internally and rejects a current context on another device.
    int device;
  };

  struct damacy;
  struct damacy_batch;

  // Create a damacy instance. cfg->device < 0 captures the current
  // CUcontext (DAMACY_INVAL if none); cfg->device >= 0 retains the
  // primary for that device and pushes it on the calling thread.
  enum damacy_status damacy_create(const struct damacy_config* cfg,
                                   struct damacy** out);

  // The CUDA device index this instance is bound to.
  int damacy_get_device(const struct damacy* d);

  // Tear down. Does NOT flush in-flight work; the io_queue is asked to
  // shut down and pending CUDA streams are synchronized before buffers
  // are released. Pending damacy_pop callers (from another thread) wake
  // with DAMACY_SHUTDOWN.
  void damacy_destroy(struct damacy* d);

  struct damacy_push_result
  {
    struct damacy_sample_slice unconsumed;
    enum damacy_status status;
  };

  // Push as many samples as fit. The return value is the unconsumed
  // suffix of the input slice plus a status:
  //   OK        all samples were consumed
  //   AGAIN     the lookahead queue filled mid-slice; caller should pop
  //             a batch (or wait) and retry with the returned suffix
  //   INVAL     bad arguments (samples.beg > samples.end, null d, etc.)
  //   NOTFOUND  could not resolve a uri; result.unconsumed.beg points at it
  //   DTYPE     zarr source dtype has no cast path to cfg.dtype;
  //             result.unconsumed.beg points at the sample
  //   RANK      sample rank incompatible with the resolved zarr's rank
  //   SHUTDOWN  instance is in a failed state or being destroyed
  // On any non-AGAIN error, the offending sample is at
  // result.unconsumed.beg and was NOT consumed.
  struct damacy_push_result damacy_push(struct damacy* d,
                                        struct damacy_sample_slice samples);

  // Block until the next batch is on-device-ready, in push-FIFO order.
  // *out is owned by damacy until damacy_release.
  enum damacy_status damacy_pop(struct damacy* d, struct damacy_batch** out);

  // Return the batch's slot to the pool. Thread-safe; may be called from
  // a thread other than the one that called damacy_pop.
  void damacy_release(struct damacy* d, struct damacy_batch* b);

  // Drain anything currently planned/in-flight, finalize any partial last
  // batch (truncated to the number of complete samples) and ready it for
  // pop. Idempotent. Stream is resumable after flush; subsequent push
  // starts a fresh batch.
  enum damacy_status damacy_flush(struct damacy* d);

  struct damacy_batch_info
  {
    void* device_ptr;                   // dtype-typed, contiguous
    int64_t shape[DAMACY_MAX_RANK + 1]; // [N, ...zarr axes]
    uint8_t rank;                       // includes leading N axis
    enum damacy_dtype dtype;
    void* ready_stream; // CUstream
    uint64_t batch_id;  // monotonic
  };
  // device ordinal is derivable from device_ptr via
  // cuPointerGetAttribute(..., CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, ...)
  // or from ready_stream via cuStreamGetCtx → cuCtxGetDevice.

  void damacy_batch_info(const struct damacy_batch* b,
                         struct damacy_batch_info* out);

  // Cumulative metrics. All counters are stage-cumulative; reset with
  // damacy_stats_reset.
  struct damacy_metric
  {
    const char* name;
    float ms;            // cumulative
    float best_ms;       // best single observation (1e30f = none)
    double input_bytes;  // cumulative bytes consumed by stage
    double output_bytes; // cumulative bytes produced by stage
    uint64_t count;
  };

  struct damacy_stats
  {
    struct damacy_metric plan;
    struct damacy_metric io;
    struct damacy_metric h2d;
    struct damacy_metric decompress;
    // Host wall-clock around the blosc1 chunk-header parse (overlaps
    // the bulk H2D). nvcomp + post-decode kernels share stream_decode
    // so they're no longer separately measurable; both fold into
    // `decompress` above.
    struct damacy_metric decompress_parse;
    struct damacy_metric assemble;
    struct damacy_metric pop_wait_io;
    struct damacy_metric pop_wait_compute;
    struct damacy_metric flush_wait;

    uint64_t zarr_meta_hits, zarr_meta_misses;
    uint64_t shard_idx_hits, shard_idx_misses;
    uint64_t batches_emitted;
    uint64_t batches_truncated;
    uint64_t waves_emitted;
    uint64_t chunks_dispatched;

    // Total GPU bytes currently committed to wave-resident buffers and
    // batch-output pools, counted against max_gpu_memory_bytes. Grows from
    // wave-init to the first damacy_pop (lazy batch pool sizing) and stays
    // flat after that. Useful for surfacing the runtime budget to callers.
    uint64_t gpu_bytes_committed;
  };

  void damacy_stats_get(const struct damacy* d, struct damacy_stats* out);
  void damacy_stats_reset(struct damacy* d);

#ifdef __cplusplus
}
#endif
