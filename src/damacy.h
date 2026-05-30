// damacy public C API — high-throughput streaming loader for batches
// assembled from many sharded NGFF zarr stores.
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

  // Output dtype of an assembled batch.
  enum damacy_dtype
  {
    DAMACY_F32,
    DAMACY_BF16,
  };

  // NUMA placement strategy for pinned-host allocations and worker threads.
  enum damacy_numa_strategy
  {
    DAMACY_NUMA_AUTO = 0, // resolve GPU's host-NUMA node using the driver
    DAMACY_NUMA_DISABLED, // no-op
    DAMACY_NUMA_PIN_TO,   // pin's to an explicit override
                          // (damacy_config.numa_node)
  };

  // GPU Direct storage (GDS) enablement
  enum damacy_gds_mode
  {
    DAMACY_GDS_AUTO = 0, // defer to env DAMACY_GDS_ENALE=1
    DAMACY_GDS_ON,       // enable
    DAMACY_GDS_OFF,      // disable
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
    DAMACY_OOM,      // host allocation failed (calloc/malloc returned null)
    DAMACY_BUDGET,   // a configured cap is too small to satisfy the request
    DAMACY_SHUTDOWN, // pipeline destroyed or in failed state
  };

  // Human-readable name for a status code; safe for log/error messages.
  const char* damacy_status_str(enum damacy_status s);

  // Half-open [beg, end) interval
  struct damacy_interval
  {
    int64_t beg, end;
  };

  // Variable-rank AABB. Only dims[0..rank) are read; axis order matches
  // the zarr's stored axis order.
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

  // Performance + resource-sizing knobs. damacy_create requires explicit
  // numeric values; use damacy_config_validate_with_defaults() to expand
  // defaultable zero/invalid knobs before create.
  struct damacy_tuning
  {
    // Primary GPU budget. Hard cap on total GPU memory damacy will
    // allocate.
    // Required — no default; a value too small for the requested
    // geometry returns DAMACY_BUDGET from damacy_create.
    uint64_t max_gpu_memory_bytes;
    // Default helper: DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES.
    uint32_t max_chunk_uncompressed_bytes;
    // Default helper: DAMACY_DEFAULT_READ_OP_MAX_BYTES.
    uint64_t max_read_op_bytes;
    // Pinned-host slab pool depth, in waves. Default helper:
    // DAMACY_DEFAULT_HOST_BUFFER_WAVES, clamped to
    // [DAMACY_N_WAVES, DAMACY_MAX_HOST_BUFFER_WAVES].
    uint8_t host_buffer_waves;
    // Default helper: DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE. Clamped to
    // 0xFFFFu.
    uint32_t max_chunks_per_wave;
    // Default helper: DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK. Clamped to
    // DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK.
    uint32_t max_substreams_per_chunk;

    // Bulk chunk-read worker threads. Wave IO uses this queue.
    uint32_t n_io_threads;
    // Metadata/prefetch worker threads. Default helper:
    // DAMACY_DEFAULT_PREFETCH_IO_THREADS.
    uint32_t n_prefetch_io_threads;

    uint32_t n_array_meta_cache;
    uint32_t n_shard_index_cache;
    uint32_t n_chunk_layout_cache;

    enum damacy_numa_strategy numa_strategy;
    int numa_node;

    enum damacy_gds_mode enable_gds;
  };

  // Measurement/profiling switches.
  struct damacy_debug_flags
  {
    // Skip decode. Pipeline runs normally but instead of decompressing
    // chunks, an array's fill value is used to fill the batch.
    uint8_t bypass_decode;
  };

  // Configuration of the damacy pipeline
  struct damacy_config
  {
    // Destination dtype of assembled batches.
    enum damacy_dtype dtype;
    // Per-sample output extents (in voxels) along the zarr's axis order — same
    // layout damacy_sample.aabb uses. sample_rank must be in
    // [1, DAMACY_MAX_RANK]; every sample_shape[d] must be > 0.
    int64_t sample_shape[DAMACY_MAX_RANK];
    uint8_t sample_rank;
    uint32_t samples_per_batch;
    // User-push queue depth in samples, not batches. Must cover at least one
    // full output batch. Default helper picks two full batches.
    uint32_t lookahead_samples;

    // -1 captures current CUcontext; >= 0 retains the primary for that
    // device internally and rejects a current context on another device.
    int device;

    struct damacy_tuning tuning;
    struct damacy_debug_flags debug;
  };

  struct damacy;
  struct damacy_batch;

  // Create a damacy instance.
  enum damacy_status damacy_create(const struct damacy_config* cfg,
                                   struct damacy** out);

  // Return a copy of cfg with defaultable numeric knobs expanded to explicit
  // values. This helper preserves enum zero semantics (AUTO modes) and cannot
  // supply required fields such as max_gpu_memory_bytes, samples_per_batch, or
  // sample_shape; damacy_create still validates the returned config.
  struct damacy_config damacy_config_validate_with_defaults(
    const struct damacy_config* cfg);

  // Log (as LOG_INFO) a one-line-per-field description of the geometry damacy
  // would resolve from `cfg` at damacy_create time. Useful when diagnosing "why
  // does damacy use N MB" without standing the instance up.
  //
  // Requires a live CUDA context on the calling thread.
  void damacy_config_describe(const struct damacy_config* cfg);

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
  //   RANK      sample rank incompatible with cfg.sample_rank
  //   SHUTDOWN  instance is in a failed state or being destroyed
  // Store-derived errors (missing uri, unsupported source dtype, per-array
  // rank mismatch, decode failures) surface asynchronously from damacy_pop.
  // On any push-side non-AGAIN error, the offending sample is at
  // result.unconsumed.beg and was not consumed.
  struct damacy_push_result damacy_push(struct damacy* d,
                                        struct damacy_sample_slice samples);

  // Block until the next batch is on-device-ready, in push-FIFO order.
  // *out is owned by damacy until damacy_release.
  enum damacy_status damacy_pop(struct damacy* d, struct damacy_batch** out);

  // Return the batch's slot to the pool. Thread-safe; may be called from
  // a thread other than the one that called damacy_pop.
  void damacy_release(struct damacy* d, struct damacy_batch* b);

  // Deferred release: tell damacy not to reuse the batch's buffer until
  // `event` (a CUevent) has fired. Useful when the consumer kicked off
  // an async D2D copy on a side stream and wants the host to return
  // immediately instead of blocking on cuEventSynchronize before exiting
  // a `with` block.
  //
  // Damacy records the wait on its internal stream_post (which is where
  // assemble writes the slot's output buffer), so the next batch's
  // assemble kernel — which writes to the same buffer — will wait on
  // `event` before launching. No host synchronization is performed; this
  // call returns as soon as the wait is queued and the slot state
  // transitions to FREE.
  //
  // Latency note: stream_post is shared across both batch slots, so the
  // wait gates EVERY subsequent assemble until `event` fires — not just
  // the released slot's next reuse. A long-held consumer event therefore
  // stalls the second slot's assemble too. For maximum overlap, release
  // with an event that completes quickly relative to the consumer's
  // step time.
  //
  // The event must remain valid for the duration of this call; damacy
  // captures it into stream_post's command queue via cuStreamWaitEvent
  // and is done with the handle by return. Passing a NULL event is
  // equivalent to damacy_release.
  //
  // Returns DAMACY_OK on success or DAMACY_CUDA if the driver call fails.
  // In either case the slot is released back to the pool — on the
  // DAMACY_CUDA path the deferred wait was not installed and the slot
  // falls back to immediate release, so the caller knows reuse is not
  // gated on `event` but won't block on a future pop.
  enum damacy_status damacy_release_event(struct damacy* d,
                                          struct damacy_batch* b,
                                          void* event);

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
    // decode: stream_decode work only (nvcomp + status_reduce).
    // post_decode: stream_post work — post-decode kernels + 4B D2H +
    // cross-stream wait on decode_done. A large post_decode avg means
    // stream_post is bottlenecking; a small avg means it overlaps the
    // next wave's decode cleanly.
    struct damacy_metric decode;
    struct damacy_metric post_decode;
    // Stream_decode idle between consecutive waves' decode submissions.
    // Sums to the wave-boundary gap visible in nsys.
    struct damacy_metric decode_gap;
    struct damacy_metric assemble;
    // Time a host_slab_slot sat in SLOT_READY waiting for a WAVE_FREE
    // wave to bind to. Non-zero average → more host_buffer_waves slots
    // would let IO finish further ahead of decode without stalling.
    struct damacy_metric bind_wait;
    struct damacy_metric pop_wait; // user thread blocked on the scheduler cv
    struct damacy_metric flush_wait;

    struct
    {
      uint64_t hits;
      uint64_t misses;
    } array_meta, shard_index, chunk_layout;
    uint64_t batches_emitted;
    uint64_t batches_truncated;
    uint64_t waves_emitted;
    uint64_t chunks_dispatched;
    uint64_t chunks_planned; // total chunk_plans emitted (incl. fills)
    uint64_t chunks_to_load; // non-fill chunks (filter survivors)
    uint64_t reads_issued;   // real read_ops after coalesce
    uint64_t worker_steps;   // scheduler ticks (proxy for worker CPU)

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
