// wave_pool: the aggregate the scheduler drives. Owns both in-flight
// waves, the pinned-host slab pool, the 3 CUstreams (h2d / decode /
// post), the decode-done anchor ring, the shared zstd decoder, and
// borrowed pointers into the rest of the orchestrator. Lives inline
// inside struct damacy.
//
// The scheduler tick calls wave_pool_advance to drive both state
// machines (slot pool + wave array) one step; kick_peel_into_free_slots
// calls wave_pool_peel to start new IO. The per-stage submission
// helpers (kick_h2d / kick_decode / kick_assemble / finalize_wave) are
// implementation details of advance and live in wave_pool.c.
#pragma once

#include "damacy.h"
#include "damacy_limits.h"
#include "wave/host_slab.h"
#include "wave/wave.h"

#include <cuda.h>
#include <stdint.h>

struct damacy_batch_pool;
struct damacy_stats;
struct decoder_zstd;
struct gpu_budget;
struct store;
struct threadpool;

struct wave_pool
{
  struct damacy_wave waves[DAMACY_N_WAVES];

  // Pinned-host slab pool. n_slots >= DAMACY_N_WAVES; the surplus lets
  // peel + IO for upcoming waves complete before a wave struct is free.
  struct host_slab_slot slots[DAMACY_MAX_HOST_BUFFER_WAVES];
  uint8_t n_slots;

  // stream_decode: nvcomp decode + status_reduce. stream_post:
  // everything past ev.decode_done — gated cross-stream so wave N's
  // tail overlaps wave N+1's decode.
  CUstream stream_h2d;
  CUstream stream_decode;
  CUstream stream_post;

  // Ring of decode-done anchors for decode_gap measurement. wave events
  // are reused across iterations, so cuEventElapsedTime against
  // wave->ev.decode_done would race with the next iteration's record.
  // 4 slots covers the worst case: 2 waves × (current + previous).
  CUevent decode_done_ring[4];
  uint8_t decode_done_ring_idx;

  // Pool-shared zstd decoder. Decodes serialize FIFO on stream_decode
  // (at most one wave's decode in-flight), so a single nvcomp temp +
  // status + actual-sizes allocation suffices for both waves. The
  // decoder's substream cap is observe-and-grow: starts at
  // DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP and bumps to the next power of
  // 2 when a wave's substream count exceeds it. Per-wave fanout SOAs
  // are grown independently — see damacy_wave.fanout_cap.
  struct decoder_zstd* zstd_decoder;

  // Cached at wave_pool_init so the decoder grow path can replay the
  // per-substream + per-batch upper bounds without re-resolving cfg.
  uint64_t dev_per_wave;
  uint64_t max_chunk_uncompressed_bytes;

  // Borrowed budget tracker (owned by struct damacy). Grow paths route
  // through gpu_budget_try_commit so committed/max accounting stays in
  // one place across the orchestrator.
  struct gpu_budget* budget;

  // Borrowed (owned by struct damacy / its members). Set in wave_pool_init
  // and never updated.
  struct damacy_batch_pool* pool;
  struct store* store;
  struct threadpool* compute_pool;
  struct damacy_stats* stats;
  enum damacy_dtype dtype;

  // Opt-in switch: when set, kick_h2d parses blosc1 chunk headers on
  // device via blosc1_parse_kernel instead of on host via
  // blosc1_host_parse. Set by damacy_create from
  // cfg.use_gpu_blosc_parse or the DAMACY_GPU_BLOSC_PARSE env var.
  uint8_t use_gpu_parse;

  // GDS opt-in: peel issues store_read_submit_dev into the slot's
  // device staging buffer; bind aliases wave->dev_compressed to it;
  // submit_bulk_h2d skips the H2D copy. Validated at damacy_create:
  // requires the store to support submit_dev (libcufile.so.0 loadable
  // + cuFileDriverOpen succeeded) and requires use_gpu_parse (no host
  // buffer to parse from). Set by wave_pool_init from the enable_gds
  // parameter.
  uint8_t use_gds;
};

// Create the streams, initialize the wave array, and allocate
// host_buffer_waves pinned-host slabs of slot_cap_bytes each.
// host_slab_per_wave / dev_decompressed_per_wave come from
// wave_pool_resolve_sizing. host_buffer_waves >= DAMACY_N_WAVES;
// `budget` is the orchestrator's gpu_budget tracker — grow paths route
// through gpu_budget_try_commit on it. Returns 0 on success, 1 on
// failure (after self-cleanup).
int
wave_pool_init(struct wave_pool* wp,
               struct damacy_batch_pool* pool,
               struct store* store,
               struct threadpool* compute_pool,
               struct damacy_stats* stats,
               enum damacy_dtype dtype,
               uint8_t host_buffer_waves,
               uint64_t host_slab_per_wave,
               uint64_t dev_decompressed_per_wave,
               uint64_t max_chunk_uncompressed_bytes,
               int enable_gds,
               struct gpu_budget* budget);

// Sync + destroy streams, free per-wave + per-slot pinned host, then
// wave_destroy each wave. cuda_skip=1 leaks GPU + pinned-host resources
// (used when the CUDA context is no longer valid) but releases the
// non-pinned heap.
void
wave_pool_destroy(struct wave_pool* wp, int cuda_skip);

// Pool-level predicates.
int
any_wave_in_flight(const struct wave_pool* wp);
// True if any host_slab_slot is past SLOT_FREE — peel-in-flight, IO
// pending, ready to bind, or bound to a wave. damacy_pop / damacy_flush
// poll this to know when the pipeline has drained.
int
any_slot_in_flight(const struct wave_pool* wp);
// True if at least one host_slab_slot is SLOT_FREE and available for
// the next peel.
int
any_slot_free(const struct wave_pool* wp);

// Drive both the slot pool and the wave array one step:
//   1. SLOT_IO → SLOT_READY when store_event_query succeeds.
//   2. SLOT_READY → bound to a WAVE_FREE wave, then kick_h2d.
//   3. WAVE_H2D: poll bulk_h2d_end → release slot; poll h2d_end → kick
//      decode + assemble, advance to WAVE_ASSEMBLE.
//   4. WAVE_ASSEMBLE: poll asm_end → finalize.
// Returns the first non-OK status encountered; the caller (scheduler
// tick in damacy.c) latches it onto self->failed_status.
enum damacy_status
wave_pool_advance(struct wave_pool* wp);

// Pack chunks from `batch_slot_idx`'s remaining work into a free
// host_slab_slot and submit IO. Returns DAMACY_OK with no-op if no
// host_slab_slot is free or the batch slot has nothing left.
// DAMACY_OOM if a single chunk is larger than the slot capacity.
enum damacy_status
wave_pool_peel(struct wave_pool* wp, uint16_t batch_slot_idx);
