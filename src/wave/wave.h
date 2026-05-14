// Wave: one in-flight slice of a batch's chunks. Owns its device
// decompress arena, blosc1 parse scratch, nvcomp fanout SOAs, and the
// per-stage CUevents the scheduler polls.
//
// The pinned-host slab that feeds the bulk H2D lives in a separate
// pool of host_slab_slots (see wave/host_slab.h) so peel + IO for the
// next wave can overlap with the current wave's decode. The aggregate
// that owns both waves + the slot pool + streams + decoder lives in
// wave/wave_pool.h.
#pragma once

#include "assemble/assemble.h"
#include "damacy.h"
#include "damacy_limits.h"
#include "decoder/blosc1.h"
#include "decoder/blosc1_host.h"
#include "decoder/decoder_memcpy.h"

#include <cuda.h>
#include <stdint.h>

enum wave_state
{
  WAVE_FREE = 0,
  WAVE_H2D,      // bound to a slot; kick_h2d submitted, polling bulk_h2d_end
                 // (slot release) and h2d_end (state transition)
  WAVE_ASSEMBLE, // covers decompress + assemble on stream_decode; polled on
                 // asm_end
};

struct damacy_wave
{
  enum wave_state state;
  // Index into wave_pool.slots while bound, -1 otherwise. Set at bind;
  // cleared when bulk_h2d_end fires and the slot returns to the pool.
  int8_t bound_slot;
  // Copied from the bound slot at bind. The first three are read by
  // build_blosc1_host_chunks / kick_h2d / finalize / log paths; the
  // host_slab pointer aliases the slot's pinned buffer for the lifetime
  // of the bulk H2D.
  uint16_t batch_pool_slot;
  uint32_t batch_chunk_offset;
  uint32_t n_chunks;
  uint64_t host_used_bytes;
  void* host_slab; // borrowed from slot; NULL when not bound

  // Wave-owned device buffers (per-wave; not pooled).
  void* dev_compressed;   // mirror of slot bytes on device
  void* dev_decompressed; // decode arena
  uint64_t dev_decompressed_cap;

  // blosc1 host-parse state (all pinned). parse fills these; the fanout
  // / op records H2D onto stream_h2d for codec + post-decode stages on
  // stream_decode.
  struct blosc1_host_chunk* h_chunks;
  struct blosc1_host_scratch scratch;
  struct blosc1_host_fanout h_zstd_fan;
  struct gpu_memcpy_op* h_memcpy_ops;
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

  float parse_ms; // host wall-clock around blosc1_host_parse

  // Assemble per-wave-chunk metadata (host + device). One record per
  // chunk: arena offset + (sample_idx, chunk_d). Per-sample constants
  // live in the batch slot's d_sample_plans.
  struct assemble_chunk* h_assemble_chunks;
  struct assemble_chunk* d_assemble_chunks;
  uint32_t assemble_max_blocks_per_chunk;
  uint8_t assemble_rank;

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

  // Host-side IO timing (copied from slot at bind).
  uint64_t io_t_start_ns;
  uint64_t io_t_end_ns;

  // Per-stage byte totals (filled at bind + compute time, drained at
  // finalize into the cumulative damacy_stats fields).
  uint64_t io_bytes;
  uint64_t decomp_in_bytes;
  uint64_t decomp_out_bytes;
  uint64_t assemble_out_bytes;
};

// Returns 0 on success, 1 on failure (after self-cleanup). The wave
// allocates dev_compressed sized to the slot capacity so bulk H2D can
// copy a full slot byte-for-byte.
int
wave_init(struct damacy_wave* wave,
          uint64_t slot_cap_bytes,
          uint64_t dev_decompressed_bytes);

// cuda_skip=1 leaks GPU + pinned-host resources (used when the CUDA
// context is no longer valid) but releases the non-pinned heap.
void
wave_destroy(struct damacy_wave* wave, int cuda_skip);
