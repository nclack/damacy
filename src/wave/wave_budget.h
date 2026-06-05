// GPU memory geometry for one wave + the resolver that picks per-wave
// extents from a configured cap.
#pragma once

#include "damacy.h"
#include "wave/input_transfer.h"

#include <stddef.h>
#include <stdint.h>

struct damacy_config;
struct gpu_budget;

// Pool-level prediction (2 waves + shared scratch + batch meta).
struct gpu_budget_breakdown
{
  uint64_t dev_compressed;   // compressed-input device staging
  uint64_t dev_decompressed; // 2× dev_decompressed_per_wave
  uint64_t blosc1_meta;      // 2× per-wave parse + assemble metadata
  uint64_t fanout_soa;
  uint64_t nvcomp_temp;
  uint64_t batch_metadata; // 2× cfg.samples_per_batch × sizeof(sample_plan)
  uint64_t total;
};

enum damacy_status
gpu_budget_predict(const struct damacy_config* cfg,
                   const struct input_transfer_resources* input,
                   uint64_t dev_decompressed_per_wave,
                   struct gpu_budget_breakdown* out);

// Per-wave device allocation summary.
struct wave_alloc_summary
{
  uint64_t dev_compressed;   // compressed input staging
  uint64_t dev_decompressed; // dev_decompressed arena
  uint64_t blosc1_meta;      // d_assemble_chunks + d_blosc1_totals
  uint64_t fanout_soa;       // device fanouts + memcpy op SOA
};

enum damacy_status
wave_predict_bytes(uint32_t max_chunks_per_wave,
                   uint64_t input_staging_bytes,
                   uint64_t dev_decompressed_bytes,
                   struct wave_alloc_summary* out);

// Pool-shared nvcomp scratch.
enum damacy_status
wave_pool_shared_predict_bytes(uint32_t max_chunks_per_wave,
                               uint64_t dev_decompressed_bytes,
                               uint64_t max_chunk_uncompressed_bytes,
                               uint64_t* out_nvcomp_temp);

// Wave geometry resolver.
struct wave_pool_sizing
{
  uint64_t input_staging_per_wave;
  uint64_t dev_decompressed_per_wave; // dev_decompressed + unshuffle scratch
  uint64_t worst_case_total_bytes;
};

enum damacy_status
wave_pool_resolve_sizing(uint32_t max_chunks_per_wave,
                         uint32_t max_substreams_per_chunk,
                         uint8_t input_device_staging_buffers,
                         uint64_t max_gpu_memory_bytes,
                         uint64_t max_chunk_uncompressed_bytes,
                         uint32_t samples_per_batch,
                         struct wave_pool_sizing* out);

void
decoder_initial_caps(uint32_t max_chunks_per_wave,
                     uint64_t dev_per_wave,
                     uint64_t max_chunk_uncompressed_bytes,
                     size_t* out_substreams,
                     size_t* out_zstd_per,
                     size_t* out_total_uncompressed);
