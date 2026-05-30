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
#include "decoder/blosc1.h"
#include "decoder/blosc1_parse.h"
#include "decoder/decoder_memcpy.h"
#include "wave/fanout.h" // struct nvcomp_fanout_host

#include <cuda.h>
#include <stdint.h>

enum wave_state
{
  WAVE_FREE = 0,
  WAVE_H2D,  // bound to a slot; kick_h2d submitted, polling bulk_h2d_end
             // (slot release) and h2d_end (state transition)
  WAVE_POST, // decode submitted on stream_decode and post/assemble submitted
             // on stream_post; polled on asm_end
};

struct damacy_wave
{
  enum wave_state state;
  // Index into wave_pool.slots while bound, -1 otherwise. Set at bind;
  // cleared when compressed input is consumed and the slot returns to the pool.
  int8_t bound_slot;
  // Copied from the bound slot at bind. The first three are read by
  // kick_h2d / finalize / log paths. host_slab aliases the slot's pinned
  // buffer on the host-staging path; on GDS the bound slot owns the active
  // compressed device buffer until h2d_end.
  uint16_t render_job_idx;
  uint16_t batch_pool_slot;
  uint32_t batch_chunk_offset;
  uint32_t n_chunks;
  uint64_t host_used_bytes;
  void* host_slab; // borrowed from slot; NULL when not bound

  // Wave-owned device buffers (per-wave; not pooled).
  // dev_compressed is the ACTIVE pointer the parse / decode kernels
  // read from. On the host-staging path it equals dev_compressed_owned
  // (this wave allocates and copies into it). On the GDS path it
  // aliases the bound slot's dev_buf (cuFile writes directly into
  // slot memory); dev_compressed_owned is NULL — the slot owns the
  // device staging buffer instead.
  void* dev_compressed;
  void* dev_compressed_owned;
  void* dev_decompressed; // decode arena
  uint64_t dev_decompressed_cap;

  // GPU parse inputs + counter buffers. h_parse_chunks is pinned host
  // (cap DAMACY_MAX_CHUNKS_PER_WAVE); d_parse_chunks is the device mirror
  // H2D'd alongside the bulk slab. d_n_zstd / d_n_memcpy are device
  // counters seeded with the host-emitted slot counts; the kernels
  // atomicAdd on top. The 12-byte (n_zstd, n_memcpy, parse_err) result
  // lands in h_parse_counters via D2H before h2d_end fires.
  //
  // h_blosc_chunk_indices / d_block_chunk_map are inputs to the two
  // parse kernels; built by build_gpu_parse_chunks. d_is_memcpyed is a
  // per-chunk bitset Kernel A sets and Kernel B reads. n_blosc_zstd_*
  // and n_host_* are transient per-wave counts owned by the GPU parse
  // path (live for the duration of one kick_h2d).
  struct gpu_parse_chunk* h_parse_chunks;
  struct gpu_parse_chunk* d_parse_chunks;
  uint32_t* h_blosc_chunk_indices;
  uint32_t* d_blosc_chunk_indices;
  uint32_t* h_block_chunk_map;
  uint32_t* d_block_chunk_map;
  uint32_t* d_is_memcpyed;
  uint32_t n_blosc_zstd_chunks;
  uint32_t n_blosc_zstd_blocks;
  uint32_t n_host_memcpy; // host-emitted memcpy ops (FILL/NONE/ZSTD pre-fill)
  uint32_t n_host_zstd;   // host-emitted whole-chunk zstd entries
  uint32_t* d_n_zstd;
  uint32_t* d_n_memcpy;
  uint32_t* d_parse_err;
  uint32_t* h_parse_counters; // [n_zstd, n_memcpy, parse_err]

  // Host-pre-filled SOA prefix for the zstd fanout (whole-chunk ZSTD
  // entries), uploaded to zstd_fan before Kernel B atomic-adds per-
  // block entries on top.
  struct nvcomp_fanout_host h_zstd_fan;
  struct gpu_memcpy_op* h_memcpy_ops;

  // status_reduce atomicAdds into n_codec_errors; finalize_wave reads.
  struct blosc1_totals* d_blosc1_totals;
  struct blosc1_totals* h_blosc1_totals;

  // Device SOA mirror of the host fanout (H2D'd in kick_h2d).
  struct nvcomp_fanout zstd_fan;
  // Per-wave substream-fanout cap. wave_init seeds it at
  // DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP; kick_h2d grows just this wave's
  // h_zstd_fan + zstd_fan when n_chunks * MAX_BLOCKS exceeds it. The
  // other wave's fanout is independent — its SOA stays untouched during
  // a grow of this wave.
  uint32_t fanout_cap;

  struct gpu_memcpy_op* d_memcpy_ops;

  // Assemble per-wave-chunk metadata (host + device). One record per
  // chunk: arena offset + (sample_idx, chunk_d). Per-sample constants
  // live in the render job's d_sample_plans.
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

  float io_ms;

  // Per-stage byte totals (filled at bind + compute time, drained at
  // finalize into the cumulative damacy_stats fields).
  uint64_t io_bytes;
  uint64_t decomp_in_bytes;
  uint64_t decomp_out_bytes;
  uint64_t assemble_out_bytes;
};

// Returns 0 on success, 1 on failure (after self-cleanup). On the
// host-staging path (enable_gds = 0) the wave allocates dev_compressed
// sized to the slot capacity so bulk H2D can copy a full slot
// byte-for-byte. On the GDS path (enable_gds = 1) the slot owns the
// device staging buffer; bind_slot_to_wave aliases wave->dev_compressed
// to slot->dev_buf at bind time.
int
wave_init(struct damacy_wave* wave,
          uint32_t max_chunks_per_wave,
          uint32_t max_substreams_per_wave,
          uint64_t slot_cap_bytes,
          uint64_t dev_decompressed_bytes,
          int enable_gds);

// cuda_skip=1 leaks GPU + pinned-host resources (used when the CUDA
// context is no longer valid) but releases the non-pinned heap.
void
wave_destroy(struct damacy_wave* wave, int cuda_skip);
