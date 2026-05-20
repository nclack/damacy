// GPU memory geometry for one wave + the resolver that picks per-wave
// extents from a configured cap.
//
// Pure: no driver allocs (nvcomp temp queries are read-only). The
// runtime committer (struct gpu_budget) is the only object that holds
// state; everything here returns predicted bytes by value.
//
// The runtime grow that consumes this math (decoder_scratch_grow) is
// implemented in wave_budget.c — it shares wave_decoder_caps and
// predict_decoder_scratch_bytes with the resolver — but its only
// caller is wave_pool, which forward-declares it locally.
#pragma once

#include "damacy.h"

#include <stddef.h>
#include <stdint.h>

struct damacy_config;
struct gpu_budget;

// Pool-level prediction (2 waves + shared scratch + batch meta). The
// caller-facing breakdown of "how much GPU memory will damacy
// allocate?", produced by gpu_budget_predict. damacy_create logs this
// and seeds the gpu_budget committer with .total.
struct gpu_budget_breakdown
{
  uint64_t dev_compressed;   // 2× host_slab_per_wave (H2D mirror)
  uint64_t dev_decompressed; // 2× dev_decompressed_per_wave
  uint64_t blosc1_meta;      // 2× per-wave parse + assemble metadata
  // 2× per-wave nvcomp fanout SOA + op arrays, sized off the initial
  // floor. Observe-and-grow may raise this at runtime up to the
  // structural ceiling; the grow path enforces the configured cap.
  uint64_t fanout_soa;
  // 1× (zstd_temp + actual+status), pool-shared, sized off the initial
  // floor. Observe-and-grow applies here too.
  uint64_t nvcomp_temp;
  uint64_t batch_metadata; // 2× cfg.batch_size × sizeof(sample_plan)
  uint64_t total;
};

// Per-wave host/device extents come from wave_pool_resolve_sizing.
// max_chunk_uncompressed_bytes and batch_size come from cfg.
enum damacy_status
gpu_budget_predict(const struct damacy_config* cfg,
                   uint64_t host_slab_per_wave,
                   uint64_t dev_decompressed_per_wave,
                   struct gpu_budget_breakdown* out);

// Per-wave allocation summary: what one wave's wave_init allocates on
// the device, ignoring the pool-shared nvcomp scratch (queried
// separately). All fields are device bytes.
struct wave_alloc_summary
{
  uint64_t dev_compressed;   // dev_compressed alloc (mirrors host slab)
  uint64_t dev_decompressed; // dev_decompressed arena
  uint64_t blosc1_meta;      // d_assemble_chunks + d_blosc1_totals
  uint64_t fanout_soa;       // device fanouts + memcpy op SOA
};

enum damacy_status
wave_predict_bytes(uint32_t max_chunks_per_wave,
                   uint64_t host_slab_bytes,
                   uint64_t dev_decompressed_bytes,
                   struct wave_alloc_summary* out);

// Pool-shared nvcomp scratch (temp + actual-size + status). Sized for
// one wave's worth of substreams since decodes serialize FIFO on
// stream_decode. Returns DAMACY_OK on success; non-OK if a decoder
// query fails.
enum damacy_status
wave_pool_shared_predict_bytes(uint32_t max_chunks_per_wave,
                               uint64_t dev_decompressed_bytes,
                               uint64_t max_chunk_uncompressed_bytes,
                               uint64_t* out_nvcomp_temp);

// Wave geometry resolver. Picks the per-wave host slab and
// dev_decompressed extents that fit inside `max_gpu_memory_bytes` once
// every other component (2× wave, 1× nvcomp scratch, fanout SOAs,
// blosc1 meta, batch_metadata) is accounted for. The resolver insists
// on holding at least one chunk at `max_chunk_uncompressed_bytes` per
// wave; if the smallest viable geometry doesn't fit, returns
// DAMACY_BUDGET with a logged breakdown so the user knows what to raise.
struct wave_pool_sizing
{
  uint64_t host_slab_per_wave;        // pinned-host + dev_compressed mirror
  uint64_t dev_decompressed_per_wave; // dev_decompressed + unshuffle scratch
  // Worst-case post-grow pool footprint at this geometry: assumes both
  // per-wave fanout SOAs and the shared decoder scratch have grown to
  // max_chunks_per_wave * max_substreams_per_chunk. Always
  // <= max_gpu_memory_bytes by resolver construction.
  uint64_t worst_case_total_bytes;
};

enum damacy_status
wave_pool_resolve_sizing(uint32_t max_chunks_per_wave,
                         uint16_t max_substreams_per_chunk,
                         uint64_t max_gpu_memory_bytes,
                         uint64_t max_chunk_uncompressed_bytes,
                         uint32_t batch_size,
                         struct wave_pool_sizing* out);

// Initial substream + per-substream + per-batch caps used at
// decoder_zstd_create time. The substream count is the initial floor
// (observe-and-grow raises it); the per-substream / per-batch values
// bound nvcomp's scratch query.
void
decoder_initial_caps(uint32_t max_chunks_per_wave,
                     uint64_t dev_per_wave,
                     uint64_t max_chunk_uncompressed_bytes,
                     size_t* out_zsubs,
                     size_t* out_zstd_per,
                     size_t* out_total_uncompressed);
