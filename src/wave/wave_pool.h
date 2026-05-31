// wave_pool: the aggregate the scheduler drives.
#pragma once

#include "damacy.h"
#include "damacy_limits.h"
#include "render_job/render_job.h"
#include "wave/input_slot.h"
#include "wave/input_transfer.h"
#include "wave/wave.h"

#include <cuda.h>
#include <stdint.h>

struct damacy_batch_pool;
struct damacy_stats;
struct decoder_zstd;
struct gpu_budget;
struct render_job_pool;
struct store;
struct input_transfer_ops;

struct wave_pool
{
  struct damacy_wave waves[DAMACY_N_WAVES];

  // Input staging slots. n_slots >= DAMACY_N_WAVES.
  struct input_slot slots[DAMACY_MAX_HOST_BUFFER_WAVES];
  uint8_t n_slots;

  uint32_t max_chunks_per_wave;
  uint32_t max_substreams_per_wave;

  CUstream stream_input;
  CUstream stream_decode;
  CUstream stream_post;

  // Separate anchors for decode_gap measurement.
  CUevent decode_done_ring[4];
  uint8_t decode_done_ring_idx;

  // Pool-shared zstd decoder.
  struct decoder_zstd* zstd_decoder;

  uint64_t dev_per_wave;
  uint64_t max_chunk_uncompressed_bytes;

  // Borrowed budget tracker.
  struct gpu_budget* budget;

  // Borrowed orchestration dependencies.
  struct damacy_batch_pool* pool;
  struct render_job_pool* render_jobs;
  struct store* store;
  struct damacy_stats* stats;
  enum damacy_dtype dtype;

  const struct input_transfer_ops* input;

  // Bench bypass; see damacy_config.bypass_decode.
  uint8_t bypass_decode;
};

// Create streams, waves, slots, and shared decoder state.
int
wave_pool_init(struct wave_pool* wp,
               struct damacy_batch_pool* pool,
               struct render_job_pool* render_jobs,
               struct store* store,
               struct damacy_stats* stats,
               enum damacy_dtype dtype,
               uint8_t host_buffer_waves,
               uint32_t max_chunks_per_wave,
               uint32_t max_substreams_per_chunk,
               uint64_t input_staging_per_wave,
               uint64_t dev_decompressed_per_wave,
               uint64_t max_chunk_uncompressed_bytes,
               const struct input_transfer_ops* input,
               int bypass_decode,
               struct gpu_budget* budget);

// Destroy owned resources. cuda_skip=1 skips CUDA-owned frees.
void
wave_pool_destroy(struct wave_pool* wp, int cuda_skip);

int
any_wave_in_flight(const struct wave_pool* wp);
int
any_slot_in_flight(const struct wave_pool* wp);
int
any_slot_free(const struct wave_pool* wp);

// Drive both the slot pool and the wave array one step.
enum damacy_status
wave_pool_advance(struct wave_pool* wp, int* changed);

// Handoff state for reserve -> submit -> commit.
struct wave_pool_peel_ticket
{
  int slot_idx;
  uint32_t n_reads;
  struct wave_desc desc;
  uint8_t consumed;
};

struct wave_pool_peel_ticket
wave_pool_peel_reserve(struct wave_pool* wp,
                       uint16_t render_job_idx,
                       enum damacy_status* err);

struct store_event
wave_pool_peel_submit(struct wave_pool* wp,
                      const struct wave_pool_peel_ticket* t);

enum damacy_status
wave_pool_peel_commit(struct wave_pool* wp,
                      struct wave_pool_peel_ticket* t,
                      struct store_event ev,
                      int* changed);

// DAMACY_INVAL on unprobed BLOSC_ZSTD documents the wave-eligibility
// gate's contract; wave_chunks_eligible is the only legitimate caller.
// Exposed for gate-contract testing; internal otherwise.
struct chunk_plan;
struct sample_plan;
enum damacy_status
chunk_substreams_upper_bound(const struct chunk_plan* c,
                             const struct sample_plan* sp,
                             uint32_t* out);
