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
  WAVE_ASSEMBLE, // covers decompress + assemble on stream_decode; polled on
                 // asm_end
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
  void* host_slab;        // pinned, sized by wave_pool_resolve_sizing
  void* dev_compressed;   // mirrors host_slab on device
  void* dev_decompressed; // arena, sized by wave_pool_resolve_sizing
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
  // Per-wave substream-fanout cap. wave_init seeds it at
  // DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP; kick_h2d grows just this wave's
  // h_zstd_fan + zstd_fan when n_chunks * MAX_BLOCKS exceeds it. The
  // other wave's fanout is independent — its SOA stays untouched during
  // a grow of this wave.
  uint32_t fanout_cap;

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

  // Per-stage CUevents. h2d_end gates stream_decode; decode_done gates
  // stream_post; asm_end is polled in wave_pool_advance.
  struct wave_events
  {
    CUevent h2d_start;
    CUevent bulk_h2d_end;
    CUevent h2d_end;
    CUevent decomp_start;
    CUevent decode_done;
    CUevent decomp_end;
    CUevent asm_start;
    CUevent asm_end;
  } ev;

  // Borrowed slot from wave_pool.decode_done_ring: anchor recorded by
  // the previous kick_decode; drain_wave_metrics measures the gap to
  // this wave's decomp_start.
  CUevent prev_decode_anchor;

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

  // Phase 5 budget enforcement. max_gpu_memory_bytes is the resolved
  // ceiling (legacy default applied) and is non-zero. gpu_bytes_committed
  // is a pointer back into struct damacy so wave_pool and damacy share
  // one accounting variable. Grow paths read both: a grow that pushes
  // committed past the ceiling returns DAMACY_OOM and skips the alloc.
  uint64_t max_gpu_memory_bytes;
  uint64_t* gpu_bytes_committed;

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

// Phase 5 wave geometry resolver. Picks the per-wave host slab and
// dev_decompressed extents that fit inside `max_gpu_memory_bytes` once
// every other component (2× wave, 1× nvcomp scratch, fanout SOAs,
// blosc1 meta, batch_metadata) is accounted for. The resolver insists
// on holding at least one chunk at `max_chunk_uncompressed_bytes` per
// wave; if the smallest viable geometry doesn't fit, returns
// DAMACY_OOM with a logged breakdown so the user knows what to raise.
//
// batch_size is the cfg.batch_size that batch_metadata is sized off.
struct wave_pool_sizing
{
  uint64_t host_slab_per_wave;        // pinned-host + dev_compressed mirror
  uint64_t dev_decompressed_per_wave; // dev_decompressed + unshuffle scratch
  // Worst-case post-grow pool footprint at this geometry: assumes both
  // per-wave fanout SOAs and the shared decoder scratch have grown all
  // the way to DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE. gpu_budget_compute
  // (which uses the initial floor) returns the smaller initial number;
  // the delta is the headroom reserved for observe-and-grow. Always
  // <= max_gpu_memory_bytes by resolver construction.
  uint64_t worst_case_total_bytes;
};

enum damacy_status
wave_pool_resolve_sizing(uint64_t max_gpu_memory_bytes,
                         uint64_t max_chunk_uncompressed_bytes,
                         uint32_t batch_size,
                         struct wave_pool_sizing* out);

// Returns 0 on success, 1 on failure (after self-cleanup).
int
wave_init(struct damacy_wave* wave,
          uint64_t host_slab_bytes,
          uint64_t dev_decompressed_bytes);

void
wave_destroy(struct damacy_wave* wave, int cuda_skip);

// Create both streams + initialize both waves and wire borrowed
// pointers. host_slab_per_wave / dev_decompressed_per_wave come from
// wave_pool_resolve_sizing. max_gpu_memory_bytes + gpu_bytes_committed
// drive Phase 5 grow-time budget enforcement; gpu_bytes_committed must
// point at a uint64_t the caller owns. Returns 0 on success, 1 on
// failure (after self-cleanup so the caller can free the enclosing
// struct).
int
wave_pool_init(struct wave_pool* wp,
               struct damacy_batch_pool* pool,
               struct store* store,
               struct threadpool* compute_pool,
               struct damacy_stats* stats,
               enum damacy_status* failed_status,
               enum damacy_dtype dtype,
               uint64_t host_slab_per_wave,
               uint64_t dev_decompressed_per_wave,
               uint64_t max_chunk_uncompressed_bytes,
               uint64_t max_gpu_memory_bytes,
               uint64_t* gpu_bytes_committed);

// Sync + destroy streams, then wave_destroy each wave. cuda_skip=1
// leaks GPU resources but still releases the host heap (used when the
// CUDA context is no longer valid).
void
wave_pool_destroy(struct wave_pool* wp, int cuda_skip);

// 4 pinned-host + 4 device allocs for one nvcomp fanout. Returns 0 ok,
// 1 on first failure (logged). Partial-failure cleanup relies on `h`
// and `d` being zero-initialized by the caller — wave_destroy then
// NULL-checks each pointer and frees only what was set.
int
fanout_alloc_pinned(struct blosc1_host_fanout* h,
                    struct nvcomp_fanout* d,
                    size_t n);

// Free the 4 pinned-host + 4 device buffers a prior fanout_alloc_pinned
// allocated, zero the SOA structs so they're safe to re-allocate into.
// NULL-safe per pointer (matches wave_destroy's freeing pattern).
void
fanout_free_pinned(struct blosc1_host_fanout* h, struct nvcomp_fanout* d);

// H2D the 4 SOA arrays of one fanout in lockstep onto `s`.
enum damacy_status
fanout_upload(CUstream s,
              const struct nvcomp_fanout* d,
              const struct blosc1_host_fanout* h,
              size_t n);

// Pool-level predicates over the 2-wave array.
int
find_free_wave(const struct wave_pool* wp);
int
any_wave_in_flight(const struct wave_pool* wp);

// Drive each in-flight wave forward by one stage (IO retired → H2D,
// h2d_end retired → compute, asm_end retired → finalize). Sets
// *wp->failed_status on driver errors.
enum damacy_status
wave_pool_advance(struct wave_pool* wp);

// Pack chunks from `slot_idx`'s remaining work into `wave_idx`, submit
// IO, transition the wave to WAVE_IO. DAMACY_OOM if a single chunk is
// larger than the wave's slabs.
enum damacy_status
wave_pool_peel(struct wave_pool* wp, uint16_t wave_idx, uint16_t slot_idx);
