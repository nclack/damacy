// Assemble kernel: scatter from per-chunk decompressed buffers in a
// device arena into the output batch tensor, casting source elements to
// the destination dtype on the fly.
//
// The kernel iterates the union of chunks for each sample (a single
// rectangle per sample, U[d] = N[d] * S[d]) and culls voxels outside
// the sample's tight AABB at write time. Per-sample constants live in
// `struct sample_plan` (planner.h); per-wave-chunk records carry the
// arena offset and the chunk's grid position. Source dtype is read off
// the sample_plan; destination dtype is fixed for the launch.
#pragma once

#include "damacy.h" // enum damacy_dtype (destination)
#include "damacy_limits.h"
#include "planner/planner.h"

#include <stddef.h>
#include <stdint.h>

typedef struct CUstream_st* CUstream;

#ifdef __cplusplus
extern "C"
{
#endif

  // Per-chunk shuffle mode. Tells the assemble kernel how the
  // decompressed bytes are laid out within the chunk's blosc blocks.
  enum assemble_shuffle_mode
  {
    ASSEMBLE_SHUFFLE_NONE = 0,
    ASSEMBLE_SHUFFLE_BYTE = 1, // blosc byte-shuffle
    ASSEMBLE_SHUFFLE_BIT = 2,  // blosc bit-shuffle
  };

  // Per-chunk record materialized at wave dispatch time. shuffle fields
  // are filled by the GPU parse (NONE for non-blosc codecs).
  // is_fill=1 marks an absent source chunk: the kernel broadcasts the
  // sample's fill_value across the chunk's region; src_base_byte_off and
  // shuffle_* are ignored.
  struct assemble_chunk
  {
    uint64_t src_base_byte_off;        // arena byte offset of chunk start
    uint32_t shuffle_blocksize;        // blosc block size (bytes); 0 if NONE
    uint16_t sample_idx_in_batch;      // index into d_samples[]
    uint8_t shuffle_mode;              // enum assemble_shuffle_mode
    uint8_t shuffle_typesize;          // 1, 2, 4, 8; 0 if NONE
    uint8_t is_fill;                   // 1 = broadcast sample.fill_value
    uint32_t chunk_d[DAMACY_MAX_RANK]; // chunk grid position within sample
  };

  // Launch the assemble kernel on `stream`. Inputs (device-resident
  // unless noted):
  //   rank           — spatial rank shared by all chunks in the wave
  //                    (1..DAMACY_MAX_RANK supported; ranks 1..8 use a
  //                    compile-time-templated kernel, higher ranks use
  //                    the runtime-rank fallback)
  //   d_samples      — sample_plan[] for the batch slot (size n_samples).
  //                    Each carries src_dtype (enum dtype) used to pick
  //                    the typed read+cast at thread time.
  //   n_samples      — number of samples in the batch slot
  //   d_chunks       — assemble_chunk[] for the wave (size n_chunks)
  //   n_chunks       — number of chunks in the wave
  //   max_blocks_per_chunk — gridDim.x; max over the wave's chunks of
  //                          ∏ ceil_div(S[d], T[d]). Pre-computed host
  //                          side; chunks with fewer blocks early-return
  //                          on the surplus.
  //   arena_base / output_base — base device pointers
  //   dst_dtype      — destination dtype (DAMACY_F32 or DAMACY_BF16);
  //                    sets the kernel's write type
  //
  // Returns 0 on success, non-zero on launch error.
  int assemble_launch(CUstream stream,
                      uint8_t rank,
                      const struct sample_plan* d_samples,
                      uint32_t n_samples,
                      const struct assemble_chunk* d_chunks,
                      uint32_t n_chunks,
                      uint32_t max_blocks_per_chunk,
                      const void* arena_base,
                      void* output_base,
                      enum damacy_dtype dst_dtype);

  // Compute blocks-per-chunk for a sample given its dims and the kernel's
  // block-tile shape. Host-side helper used by damacy.c when packing the
  // wave's metadata. Returns 0 for unsupported ranks.
  uint32_t assemble_blocks_per_chunk(uint8_t rank,
                                     const struct sample_dim* dims);

#ifdef __cplusplus
}
#endif
