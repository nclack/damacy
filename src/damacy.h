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

  // AUTO defers to env DAMACY_GDS_ENABLE=1; ON/OFF override env. AUTO=0
  // so designated-init callers get the env-controlled default.
  enum damacy_gds_mode
  {
    DAMACY_GDS_AUTO = 0,
    DAMACY_GDS_ON,
    DAMACY_GDS_OFF,
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

  // Performance + resource-sizing knobs. Anything with `0 → DEFAULT_*`
  // resolves to its DAMACY_DEFAULT_* sibling in damacy_limits.h.
  struct damacy_tuning
  {
    // Primary GPU budget. Hard cap on total GPU memory damacy will
    // allocate for wave-resident buffers, the shared decoder scratch,
    // per-wave fanout SOAs, and per-batch metadata. The resolver carves
    // out the batch-output pool (2 × batch_size × product(sample_shape)
    // × dtype_bpe) from this cap before sizing wave-resident buffers.
    // Required — no default; a value too small for the requested
    // geometry returns DAMACY_BUDGET from damacy_create.
    uint64_t max_gpu_memory_bytes;
    // 0 → DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES.
    uint32_t max_chunk_uncompressed_bytes;
    // 0 → DAMACY_DEFAULT_READ_OP_MAX_BYTES.
    uint64_t max_read_op_bytes;
    // Pinned-host slab pool depth, in waves. 0 →
    // DAMACY_DEFAULT_HOST_BUFFER_WAVES. Clamped to
    // [DAMACY_N_WAVES, DAMACY_MAX_HOST_BUFFER_WAVES].
    uint8_t host_buffer_waves;
    // 0 → DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE. Clamped to 0xFFFFu.
    uint32_t max_chunks_per_wave;
    // 0 → DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK. Clamped to
    // DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK.
    uint32_t max_substreams_per_chunk;

    uint32_t n_io_threads;

    uint32_t n_zarrs_meta_cache;
    uint32_t n_shards_meta_cache;

    // AUTO resolves the GPU's host-NUMA node; DISABLED is a no-op;
    // PIN_TO forces `numa_node`.
    enum damacy_numa_strategy numa_strategy;
    int numa_node;
    // GPUDirect Storage: cuFile reads compressed bytes straight to GPU.
    // Requires libcufile.so.0 and a successful cuFileDriverOpen at
    // damacy_create.
    enum damacy_gds_mode enable_gds;
  };

  // Measurement/profiling switches. None of these should be set in
  // production — they intentionally alter pipeline correctness to
  // isolate stages for benchmarking.
  struct damacy_debug_flags
  {
    // Skip decode: planner, IO, and H2D run normally, but every chunk
    // is flipped to is_fill at parse + assemble build. Assemble
    // broadcasts the array's fill_value. CUevents on stream_decode
    // are still recorded so the state machine advances, but the decode
    // and decode_gap metrics are meaningless under this flag.
    uint8_t bypass_decode;
  };

  // All resource caps fixed at create-time. Nothing grows after this.
  // Output batches are double-buffered (B=2); waves are double-buffered
  // internally. Neither is configurable.
  struct damacy_config
  {
    // Destination dtype of assembled batches. Source zarrs may carry
    // any supported integer or float type; the assemble kernel casts
    // each element (RNE float-promote, no overflow handling). Sources
    // without a cast path error with DAMACY_DTYPE at push.
    enum damacy_dtype dtype;
    // Per-sample output extents (in dst voxels) along the zarr's axis
    // order — same layout damacy_sample.aabb uses. The resolver carves
    // out 2 × batch_size × product(sample_shape) × dtype_bpe from
    // tuning.max_gpu_memory_bytes before sizing wave-resident buffers,
    // so the batch-output pool is guaranteed to fit. damacy_push
    // validates each sample's aabb extent against this shape and
    // rejects mismatches with DAMACY_INVAL. sample_rank must be in
    // [1, DAMACY_MAX_RANK]; every sample_shape[d] must be > 0.
    int64_t sample_shape[DAMACY_MAX_RANK];
    uint8_t sample_rank;
    uint32_t batch_size;        // samples per batch
    uint32_t lookahead_batches; // user-push queue depth (>= 2)

    // -1 captures current CUcontext; >= 0 retains the primary for that
    // device internally and rejects a current context on another device.
    int device;

    struct damacy_tuning tuning;
    struct damacy_debug_flags debug;
  };

  struct damacy;
  struct damacy_batch;

  // Create a damacy instance. cfg->device < 0 captures the current
  // CUcontext (DAMACY_INVAL if none); cfg->device >= 0 retains the
  // primary for that device and pushes it on the calling thread.
  enum damacy_status damacy_create(const struct damacy_config* cfg,
                                   struct damacy** out);

  // Log a one-line-per-field description of the geometry damacy would
  // resolve from `cfg` at damacy_create time: max_gpu_memory_bytes plus
  // the derived host_slab_per_wave, dev_decompressed_per_wave, initial
  // nvcomp temp, initial allocation, headroom reserved for the
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
