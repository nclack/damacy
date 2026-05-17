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

  // NUMA placement strategy for pinned-host allocations and worker
  // threads. `AUTO` (default) resolves the GPU's host-NUMA node from
  // the driver and pins; `DISABLED` is a full no-op (today's behavior);
  // `PIN_TO` uses the explicit damacy_config.numa_node override.
  enum damacy_numa_strategy
  {
    DAMACY_NUMA_AUTO = 0,
    DAMACY_NUMA_DISABLED,
    DAMACY_NUMA_PIN_TO,
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

    // Primary GPU memory budget. Hard cap on total GPU memory damacy
    // will allocate for wave-resident buffers, the shared decoder
    // scratch, per-wave fanout SOAs, and per-batch metadata. Internal
    // sizing (host slab, device decompress arena, nvcomp temp, …) is
    // derived from this value. 0 selects
    // DAMACY_DEFAULT_MAX_GPU_MEMORY_BYTES. damacy_create returns
    // DAMACY_OOM (with a log_error breakdown) if even the smallest
    // viable wave geometry would exceed this — including the
    // double-buffered batch-output pool the resolver carves out from
    // sample_shape × batch_size × dtype_bpe. Observe-and-grow paths
    // (zstd decoder scratch, per-wave fanout) also enforce this cap at
    // grow time and return DAMACY_OOM if the new size would exceed it.
    uint64_t max_gpu_memory_bytes;

    // Per-sample output extents (in dst voxels) along the zarr's axis
    // order — same layout damacy_sample.aabb uses. The resolver carves
    // out 2 × batch_size × product(sample_shape) × dtype_bpe from
    // max_gpu_memory_bytes before sizing wave-resident buffers, so the
    // batch-output pool is guaranteed to fit. damacy_push validates each
    // sample's aabb extent against this shape and rejects mismatches
    // with DAMACY_INVAL. sample_rank must be in [1, DAMACY_MAX_RANK];
    // every sample_shape[d] must be > 0.
    int64_t sample_shape[DAMACY_MAX_RANK];
    uint8_t sample_rank;

    // Pinned-host slab pool depth, in waves. Each slot holds one wave's
    // compressed bytes; extra slots let IO for upcoming waves complete
    // before the wave struct is free, shrinking the wave-boundary gap on
    // stream_decode. Must be >= DAMACY_N_WAVES (2). 0 selects
    // DAMACY_DEFAULT_HOST_BUFFER_WAVES (3). Each slot costs one
    // dev_compressed_per_wave of pinned host memory.
    uint8_t host_buffer_waves;

    // -1 captures current CUcontext; >= 0 retains the primary for that
    // device internally and rejects a current context on another device.
    int device;

    // NUMA placement plan for pinned-host allocations + io_queue /
    // scheduler worker thread affinity. AUTO resolves the GPU's
    // host-NUMA node via cuDeviceGetAttribute(HOST_NUMA_ID) (with a
    // /sys/bus/pci/devices/<BDF>/numa_node fallback). DISABLED is a
    // no-op; PIN_TO forces `numa_node`. The whole feature is a no-op
    // when libnuma is not linked in or numa_available()<0 at runtime
    // (a single INFO log line announces it once).
    enum damacy_numa_strategy numa_strategy;
    // Only consulted when numa_strategy == DAMACY_NUMA_PIN_TO. Out-of-
    // range values fall back to no-op with a warning.
    int numa_node;

    // Read compressed chunk bytes directly into device memory via
    // NVIDIA GPUDirect Storage (cuFile), skipping the pinned-host
    // staging slab and the bulk H2D copy. libcufile is dlopen'd at
    // store init; if it isn't present on the host, or cuFileDriverOpen
    // fails, damacy_create returns DAMACY_INVAL. The DAMACY_GDS_ENABLE=1
    // environment variable overrides this at damacy_create time when set.
    int enable_gds;
  };

  struct damacy;
  struct damacy_batch;

  // Create a damacy instance. cfg->device < 0 captures the current
  // CUcontext (DAMACY_INVAL if none); cfg->device >= 0 retains the
  // primary for that device and pushes it on the calling thread.
  enum damacy_status damacy_create(const struct damacy_config* cfg,
                                   struct damacy** out);

  // Log a one-line-per-field description of the geometry damacy would
  // resolve from `cfg` at damacy_create time: input max_gpu_memory_bytes
  // plus the derived host_slab_per_wave, dev_decompressed_per_wave,
  // initial nvcomp temp, initial allocation, headroom reserved for the
  // observe-and-grow paths, and remaining slack. Emitted at LOG_INFO
  // through the standard log/log.h dispatcher. Useful when diagnosing
  // "why does damacy use N MB" without standing the instance up.
  //
  // Requires a live CUDA context on the calling thread: the resolver
  // and gpu_budget_predict call into nvcomp's decoder_zstd_query_temp_bytes
  // to size scratch. With no current context the function still
  // returns, but logs a "gpu_budget_predict failed" /
  // "wave_pool_resolve_sizing failed" line and stops short of the
  // per-component breakdown. NULL cfg is safe (logs and returns).
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

    uint64_t zarr_meta_hits, zarr_meta_misses;
    uint64_t shard_idx_hits, shard_idx_misses;
    uint64_t batches_emitted;
    uint64_t batches_truncated;
    uint64_t waves_emitted;
    uint64_t chunks_dispatched;
    uint64_t worker_steps; // scheduler ticks (proxy for worker CPU)

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
