// One in-flight slice of a batch's chunks. Owns the per-wave host slab,
// device decompress arena, blosc1 parse scratch, nvcomp fanout SOAs, and
// the per-stage CUevents the scheduler polls. Two waves are kept in
// flight by the orchestrator (damacy.c).
//
// Lifetime: wave_init at create-time, wave_destroy at destroy-time.
// cuda_skip=1 leaks GPU resources (used when the CUDA context is no
// longer valid) but releases host memory.
#pragma once

#include "assemble/assemble.h"
#include "damacy.h"
#include "damacy_limits.h"
#include "decoder/blosc1.h"
#include "decoder/blosc1_host.h"
#include "decoder/decoder_lz4.h"
#include "decoder/decoder_memcpy.h"
#include "decoder/decoder_zstd.h"
#include "decoder/shuffle.h"
#include "store/store.h"

#include <cuda.h>
#include <stdint.h>

enum wave_state
{
  WAVE_FREE = 0,
  WAVE_IO,
  WAVE_H2D,
  WAVE_ASSEMBLE, // covers decompress + assemble (one stream, one event)
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
  // / op records H2D onto stream_h2d for codec + post-decode stages.
  struct blosc1_host_chunk* h_chunks;
  struct blosc1_host_scratch scratch;
  struct blosc1_host_fanout h_zstd_fan;
  struct blosc1_host_fanout h_lz4_fan;
  struct gpu_memcpy_op* h_memcpy_ops;
  struct gpu_shuffle_op* h_unshuffle_ops;
  struct gpu_shuffle_op* h_bitunshuffle_ops;
  struct blosc1_totals* h_blosc1_totals;

  // status_reduce atomicAdds into n_codec_errors; finalize_wave reads.
  struct blosc1_totals* d_blosc1_totals;

  // Device SOA mirrors of the host fanout (H2D'd in kick_h2d).
  struct nvcomp_fanout zstd_fan;
  struct nvcomp_fanout lz4_fan;

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
  // stream_compute; ev.asm_end is the wave's "done" event polled in
  // advance_waves.
  struct wave_events
  {
    CUevent h2d_start;
    CUevent bulk_h2d_end; // bulk slab H2D done; used for stats.h2d
    CUevent h2d_end;      // + fanout/op H2Ds + d_blosc1_totals zero done
    CUevent decomp_start;
    CUevent zstd_done;
    CUevent lz4_done;
    CUevent post_start; // stream_compute resumes after waiting on codec streams
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

  // Per-wave decoders: nvCOMP scratch is not safe to share across
  // in-flight waves.
  struct decoder_zstd* zstd_decoder;
  struct decoder_lz4* lz4_decoder;
};

// Returns 0 on success, 1 on failure (after self-cleanup).
int wave_init(struct damacy_wave* wave,
              uint64_t host_slab_bytes,
              uint64_t dev_decompressed_bytes,
              uint8_t max_bpe,
              uint64_t max_chunk_uncompressed_bytes);

void wave_destroy(struct damacy_wave* wave, int cuda_skip);

// 4 pinned-host + 4 device allocs for one nvcomp fanout. Returns 0 ok,
// 1 on first failure (logged).
int fanout_alloc_pinned(struct blosc1_host_fanout* h,
                        struct nvcomp_fanout* d,
                        size_t n);

// H2D the 4 SOA arrays of one fanout in lockstep onto `s`.
enum damacy_status fanout_upload(CUstream s,
                                 const struct nvcomp_fanout* d,
                                 const struct blosc1_host_fanout* h,
                                 size_t n);

// Pool-level predicates over a 2-wave array.
int find_free_wave(const struct damacy_wave waves[2]);
int any_wave_in_flight(const struct damacy_wave waves[2]);
