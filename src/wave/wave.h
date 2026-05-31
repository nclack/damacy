#pragma once

#include "assemble/assemble.h"
#include "decoder/blosc1.h"
#include "decoder/blosc1_parse.h"
#include "decoder/decoder_memcpy.h"
#include "wave/fanout.h"

#include <cuda.h>
#include <stdint.h>

struct input_slot;

enum wave_state
{
  WAVE_FREE = 0,
  WAVE_INPUT,
  WAVE_POST,
};

struct damacy_wave
{
  enum wave_state state;
  int8_t bound_slot;
  uint16_t render_job_idx;
  uint16_t batch_pool_slot;
  uint32_t batch_chunk_offset;
  uint32_t n_chunks;
  uint64_t input_used_bytes;
  void* host_input;

  // dev_compressed points at the active compressed input. It may be the
  // wave-owned buffer or a bound input slot buffer.
  void* dev_compressed;
  void* dev_compressed_owned;
  void* dev_decompressed;
  uint64_t dev_decompressed_cap;

  struct gpu_parse_chunk* h_parse_chunks;
  struct gpu_parse_chunk* d_parse_chunks;
  uint32_t* h_blosc_chunk_indices;
  uint32_t* d_blosc_chunk_indices;
  uint32_t* h_block_chunk_map;
  uint32_t* d_block_chunk_map;
  uint32_t* d_is_memcpyed;
  uint32_t n_blosc_zstd_chunks;
  uint32_t n_blosc_zstd_blocks;
  uint32_t n_host_memcpy;
  uint32_t n_host_zstd;
  uint32_t* d_n_zstd;
  uint32_t* d_n_memcpy;
  uint32_t* d_parse_err;
  uint32_t* h_parse_counters;

  struct nvcomp_fanout_host h_zstd_fan;
  struct gpu_memcpy_op* h_memcpy_ops;

  struct blosc1_totals* d_blosc1_totals;
  struct blosc1_totals* h_blosc1_totals;

  struct nvcomp_fanout zstd_fan;
  uint32_t fanout_cap;

  struct gpu_memcpy_op* d_memcpy_ops;

  struct assemble_chunk* h_assemble_chunks;
  struct assemble_chunk* d_assemble_chunks;
  uint32_t assemble_max_blocks_per_chunk;
  uint8_t assemble_rank;

  struct wave_events
  {
    CUevent input_start;
    CUevent input_transfer_done;
    CUevent input_parse_done;
    CUevent decomp_start;
    CUevent decode_done;
    CUevent decomp_end;
    CUevent asm_start;
    CUevent asm_end;
  } ev;

  CUevent prev_decode_anchor;

  float io_ms;

  uint64_t io_bytes;
  uint64_t decomp_in_bytes;
  uint64_t decomp_out_bytes;
  uint64_t assemble_out_bytes;
};

int
wave_init(struct damacy_wave* wave,
          uint32_t max_chunks_per_wave,
          uint32_t max_substreams_per_wave,
          uint64_t input_staging_device_bytes,
          uint64_t dev_decompressed_bytes);

// cuda_skip=1 leaks GPU + pinned-host resources (used when the CUDA
// context is no longer valid) but releases the non-pinned heap.
void
wave_destroy(struct damacy_wave* wave, int cuda_skip);

void
wave_bind_input_slot(struct damacy_wave* wave,
                     int slot_idx,
                     const struct input_slot* slot,
                     void* dev_compressed);

void
wave_unbind_input_slot(struct damacy_wave* wave);
