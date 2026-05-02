// Three-stream throughput driver for chunk-aligned sample crops.
//
// Per batch the pipeline:
//   1. Plans which inner chunks each sample needs (currently a configurable
//      access pattern over the chunk grid).
//   2. Issues store reads onto pinned host staging via the zarr reader's
//      io_queue.
//   3. Async-copies pinned host -> device staging on `reader_stream`.
//   4. Decompresses on `decoder_stream` (waits on reader event).
//   5. Assembles into the batch's output slot on `assembler_stream`
//      (waits on decoder event). v0's "assemble" for chunk-aligned samples
//      is a per-chunk D2D memcpy; a real gather kernel is v1.
//
// `prefetch_depth` controls how many batches are in flight; each holds a
// pinned-host slot, a device-compressed slot, and a device-decompressed
// output slot.
#pragma once

#include "decoder.h"
#include "zarr.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct pipeline;

  enum pipeline_pattern
  {
    PIPELINE_PATTERN_SEQUENTIAL = 0, // raster scan over the chunk grid
    PIPELINE_PATTERN_RANDOM,         // uniform random starting chunk per sample
  };

  struct pipeline_config
  {
    struct store* store;        // not owned (shared with reader)
    struct zarr_reader* reader; // not owned
    struct decoder* decoder;    // not owned
    int batch_size;             // samples per batch
    int chunks_per_sample;      // contiguous along innermost axis
    int prefetch_depth;         // batches in flight (>= 1)
    int device_id;
    enum pipeline_pattern pattern;
    uint64_t seed; // for PATTERN_RANDOM
    // Compressed chunk size upper bound used to size staging rings.
    // Pass 0 to default to inner_chunk_uncompressed_bytes * 2.
    size_t max_compressed_chunk_bytes;
  };

  struct pipeline* pipeline_create(const struct pipeline_config* cfg);
  void pipeline_destroy(struct pipeline* p);

  struct pipeline_batch
  {
    void* device_ptr; // [batch_size, chunks_per_sample, inner_chunk_voxels]
    size_t nbytes;    // total decompressed bytes in this batch
    cudaEvent_t ready_event; // fires when device_ptr is consumable
    uint64_t batch_id;       // monotonically increasing; pass to release
  };

  // Pull the next batch. Blocks until at least the head slot has a queued
  // batch. Returns 0 on success.
  int pipeline_next(struct pipeline* p, struct pipeline_batch* out);

  // Caller signals that they are done with `batch_id`; the slot becomes
  // available for the next prefetch.
  void pipeline_release(struct pipeline* p, uint64_t batch_id);

  struct pipeline_stats
  {
    uint64_t n_batches;
    uint64_t bytes_read_compressed; // sum of compressed bytes pulled from store
    uint64_t bytes_decompressed;    // sum of uncompressed bytes produced
    double wall_seconds;            // since pipeline_create
  };

  void pipeline_stats_get(const struct pipeline* p, struct pipeline_stats* out);

#ifdef __cplusplus
}
#endif
