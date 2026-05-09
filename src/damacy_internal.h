// Cross-module internal interface to damacy.c. Lets wave/ reach into
// the orchestrator's state (streams, stats, batch pool) without making
// any of these symbols part of the public C API.
#pragma once

#include "batch_pool/batch_pool.h"
#include "damacy.h"
#include "damacy_stats.h"
#include "lookahead/lookahead.h"
#include "wave/wave.h"

#include <cuda.h>
#include <stdint.h>

struct planner;
struct store;
struct threadpool;
struct zarr_meta_cache;
struct zarr_shard_cache;

struct damacy_batch
{
  struct damacy* d;
  uint16_t slot_idx;
  uint64_t batch_id;
};

struct damacy
{
  struct damacy_config cfg;
  enum damacy_status failed_status;
  uint64_t next_batch_id;
  uint64_t page_alignment;
  int cuda_device;
  // -1 = captured caller's ctx; else release at destroy.
  int retained_primary_device;
  // Pushed per-call by ctx_guard when retained.
  CUcontext retained_primary;

  // GPU memory budgeting. Lazy batch-output tensors are summed against
  // gpu_bytes_budget at batch_pool_allocate. 0 budget = no cap.
  uint64_t gpu_bytes_committed;
  uint64_t gpu_bytes_budget;

  struct store* store;
  struct zarr_meta_cache* meta_cache;
  struct zarr_shard_cache* shard_cache;
  struct planner* planner;
  CUstream stream_h2d;
  CUstream stream_compute;
  CUstream stream_zstd;
  CUstream stream_lz4;

  struct threadpool* compute_pool; // host blosc1 parse

  struct damacy_lookahead lookahead;
  struct damacy_batch_pool batch_pool;
  struct damacy_wave waves[2];

  // Sample working set used while planning one batch.
  struct damacy_sample_slot* batch_samples;
  struct damacy_sample* batch_stage;

  struct damacy_batch handle;
  struct damacy_stats stats;
};

// Plan the next batch from the lookahead into the given slot. Defined
// in damacy.c; called from wave/ when kick_new_waves needs a fresh batch.
enum damacy_status damacy_plan_into_slot(struct damacy* self,
                                         uint16_t slot_idx,
                                         uint32_t n_samples);
