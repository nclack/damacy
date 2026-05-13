// Wave: one in-flight slice of a batch's chunks. Owns the per-wave host
// slab, device decompress arena, blosc1 parse scratch, nvcomp fanout SOAs,
// and the per-stage CUevents the scheduler polls.
//
// wave_pool: aggregate that owns both waves, the two CUstreams the
// scheduler drives (stream_h2d, stream_decode), and borrowed pointers
// into the rest of the orchestrator (batch_pool, store, threadpool,
// stats, failed_status).
// damacy.c holds one inline wave_pool and calls wave_pool_advance /
// wave_pool_peel directly — no per-call ctx building.
//
// Lifetime: wave_pool_init at create-time, wave_pool_destroy at
// destroy-time. cuda_skip=1 leaks GPU resources (used when the CUDA
// context is no longer valid) but releases host memory.
#pragma once

#include "assemble/assemble.h"
#include "damacy.h"
#include "decoder/blosc1.h"
#include "decoder/blosc1_host.h"
#include "decoder/decoder_memcpy.h"
#include "decoder/decoder_zstd.h"
#include "store/store.h"

#include <cuda.h>
#include <stdint.h>

enum wave_state
{
  WAVE_FREE = 0,
  WAVE_IO,
  WAVE_H2D,
  WAVE_ASSEMBLE, // covers decompress + assemble on stream_decode; polled on asm_end
};

struct damacy_wave
{
  enum wave_state state;
  uint16_t batch_pool_slot;    // valid when state != WAVE_FREE
  uint32_t batch_chunk_offset; // index into slot->chunk_plans of first chunk
  uint32_t n_chunks;           // chunks in this wave (1..max_chunks_per_wave)

  // Cursors into the wave's slabs (set at IO submit; read at later stages).
  uint64_t host_used_bytes;

  // Wave-owned device + pinned-host buffers.
  void* host_slab;        // pinned, cfg.host_buffer_bytes / 2
  void* dev_compressed;   // mirrors host_slab on device
  void* dev_decompressed; // arena, cfg.device_buffer_bytes / 2
  uint64_t host_slab_cap;
  uint64_t dev_decompressed_cap;

  // blosc1 host-parse state (all pinned). parse fills these; the fanout
  // / op records H2D onto stream_h2d for codec + post-decode stages on
  // stream_decode.
  struct blosc1_host_chunk* h_chunks;
  struct blosc1_host_scratch scratch;
  struct blosc1_host_fanout h_zstd_fan;
  struct gpu_memcpy_op* h_memcpy_ops;
  struct gpu_shuffle_op* h_unshuffle_ops;
  struct gpu_shuffle_op* h_bitunshuffle_ops;
  struct blosc1_totals* h_blosc1_totals;

  // status_reduce atomicAdds into n_codec_errors; finalize_wave reads.
  struct blosc1_totals* d_blosc1_totals;

  // Device SOA mirror of the host fanout (H2D'd in kick_h2d).
  struct nvcomp_fanout zstd_fan;

  struct gpu_memcpy_op* d_memcpy_ops;
  struct gpu_shuffle_op* d_unshuffle_ops;
  struct gpu_shuffle_op* d_bitunshuffle_ops;

  float parse_ms; // host wall-clock around blosc1_host_parse
  // Same extent as dev_decompressed; (bit)unshuffle staging so the
  // per-block transpose isn't bounded by the 64 KB shared-memory cap.
  void* dev_unshuffle_scratch;

  // Assemble per-wave-chunk metadata (host + device). One record per
  // chunk: arena offset + (sample_idx, chunk_d). Per-sample constants
  // live in the batch slot's d_sample_plans.
  struct assemble_chunk* h_assemble_chunks;
  struct assemble_chunk* d_assemble_chunks;
  uint32_t assemble_max_blocks_per_chunk;
  uint8_t assemble_rank;

  struct store_read* store_reads;
  struct store_event io_event;

  // Per-stage CUevents. ev.h2d_end is the cuStreamWaitEvent target on
  // stream_decode; ev.asm_end is the wave's "done" event polled in
  // wave_pool_advance.
  struct wave_events
  {
    CUevent h2d_start;
    CUevent bulk_h2d_end; // bulk slab H2D done; used for stats.h2d
    CUevent h2d_end;      // + fanout/op H2Ds + d_blosc1_totals zero done
    CUevent decomp_start;
    CUevent decomp_end;
    CUevent asm_start;
    CUevent asm_end;
  } ev;

  // Host-side IO timing: submit→completion wall-clock.
  uint64_t io_t_start_ns;
  uint64_t io_t_end_ns;

  // Per-stage byte totals (filled at peel + compute time, drained at
  // finalize into the cumulative damacy_stats fields).
  uint64_t io_bytes;
  uint64_t decomp_in_bytes;
  uint64_t decomp_out_bytes;
  uint64_t assemble_out_bytes;
};

struct damacy_batch_pool;
struct damacy_stats;
struct store;
struct threadpool;

// Owns the two in-flight waves, the two CUstreams the scheduler drives,
// and borrowed pointers into the rest of the orchestrator. Lives inside
// struct damacy; passed to all wave functions by pointer.
struct wave_pool
{
  struct damacy_wave waves[2];

  // wave_pool owns these — created in wave_pool_init, destroyed in
  // wave_pool_destroy. stream_decode carries nvcomp decode +
  // post-decode (memcpy / (bit)unshuffle) + assemble in FIFO order.
  CUstream stream_h2d;
  CUstream stream_decode;

  // Pool-shared zstd decoder. Decodes serialize FIFO on stream_decode
  // (at most one wave's decode in-flight), so a single nvcomp temp +
  // status + actual-sizes allocation suffices for both waves.
  //
  // Substream-batch cap on this decoder + both waves' fanout SOAs is
  // observe-and-grow: starts at DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP and
  // bumps to the next power of 2 when a wave's substream count exceeds
  // the current cap, capped at DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE.
  struct decoder_zstd* zstd_decoder;

  // Cached at wave_pool_init so wave_pool_grow_zstd_batch can replay
  // the per-substream + per-batch upper bounds without re-resolving cfg.
  uint64_t dev_per_wave;
  uint64_t max_chunk_uncompressed_bytes;

  // Borrowed (owned by struct damacy / its members). Set in wave_pool_init
  // and never updated.
  struct damacy_batch_pool* pool;
  struct store* store;
  struct threadpool* compute_pool;
  struct damacy_stats* stats;
  enum damacy_status* failed_status;
  enum damacy_dtype dtype;
};

// Predicted device-resident byte cost of one wave (wave_init's GPU
// allocs only). gpu_budget/ doubles the per-component sums for the
// 2-wave pool. The shared nvcomp scratch is queried separately via
// wave_pool_shared_predict_bytes (counted once, not 2×).
struct wave_alloc_summary
{
  uint64_t dev_compressed;        // dev_compressed alloc (mirrors host slab)
  uint64_t dev_decompressed;      // dev_decompressed arena
  uint64_t dev_unshuffle_scratch; // matches dev_decompressed extent
  uint64_t blosc1_meta;           // d_assemble_chunks + d_blosc1_totals
  uint64_t fanout_soa;            // device fanouts + memcpy/shuffle op SOAs
};

enum damacy_status
wave_predict_bytes(uint64_t host_slab_bytes,
                   uint64_t dev_decompressed_bytes,
                   struct wave_alloc_summary* out);

// Pool-shared nvcomp scratch (temp + actual-size + status). Sized for
// one wave's worth of substreams since decodes serialize FIFO on
// stream_decode. Returns DAMACY_OK on success; non-OK if a decoder
// query fails.
enum damacy_status
wave_pool_shared_predict_bytes(uint64_t dev_decompressed_bytes,
                               uint64_t max_chunk_uncompressed_bytes,
                               uint64_t* out_nvcomp_temp);

// Returns 0 on success, 1 on failure (after self-cleanup).
int wave_init(struct damacy_wave* wave,
              uint64_t host_slab_bytes,
              uint64_t dev_decompressed_bytes);

void wave_destroy(struct damacy_wave* wave, int cuda_skip);

// Create both streams + initialize both waves and wire borrowed
// pointers. Returns 0 on success, 1 on failure (after self-cleanup so
// the caller can free the enclosing struct).
int wave_pool_init(struct wave_pool* wp,
                   struct damacy_batch_pool* pool,
                   struct store* store,
                   struct threadpool* compute_pool,
                   struct damacy_stats* stats,
                   enum damacy_status* failed_status,
                   enum damacy_dtype dtype,
                   uint64_t host_buffer_bytes,
                   uint64_t device_buffer_bytes,
                   uint64_t max_chunk_uncompressed_bytes);

// Sync + destroy streams, then wave_destroy each wave. cuda_skip=1
// leaks GPU resources but still releases the host heap (used when the
// CUDA context is no longer valid).
void wave_pool_destroy(struct wave_pool* wp, int cuda_skip);

// 4 pinned-host + 4 device allocs for one nvcomp fanout. Returns 0 ok,
// 1 on first failure (logged). Partial-failure cleanup relies on `h`
// and `d` being zero-initialized by the caller — wave_destroy then
// NULL-checks each pointer and frees only what was set.
int fanout_alloc_pinned(struct blosc1_host_fanout* h,
                        struct nvcomp_fanout* d,
                        size_t n);

// Free the 4 pinned-host + 4 device buffers a prior fanout_alloc_pinned
// allocated, zero the SOA structs so they're safe to re-allocate into.
// NULL-safe per pointer (matches wave_destroy's freeing pattern).
void fanout_free_pinned(struct blosc1_host_fanout* h, struct nvcomp_fanout* d);

// H2D the 4 SOA arrays of one fanout in lockstep onto `s`.
enum damacy_status fanout_upload(CUstream s,
                                 const struct nvcomp_fanout* d,
                                 const struct blosc1_host_fanout* h,
                                 size_t n);

// Pool-level predicates over the 2-wave array.
int find_free_wave(const struct wave_pool* wp);
int any_wave_in_flight(const struct wave_pool* wp);

// Drive each in-flight wave forward by one stage (IO retired → H2D,
// h2d_end retired → compute, asm_end retired → finalize). Sets
// *wp->failed_status on driver errors.
enum damacy_status wave_pool_advance(struct wave_pool* wp);

// Pack chunks from `slot_idx`'s remaining work into `wave_idx`, submit
// IO, transition the wave to WAVE_IO. DAMACY_OOM if a single chunk is
// larger than the wave's slabs.
enum damacy_status wave_pool_peel(struct wave_pool* wp,
                                  uint16_t wave_idx,
                                  uint16_t slot_idx);
