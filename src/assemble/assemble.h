// Assemble kernel: scatter from per-chunk decompressed buffers in a
// device arena into the output batch tensor.
//
// The kernel iterates the union of chunks for each sample (a single
// rectangle per sample, U[d] = N[d] * S[d]) and culls voxels outside
// the sample's tight AABB at write time. Per-sample constants live in
// `struct sample_plan` (planner.h); per-wave-chunk records carry the
// arena offset and the chunk's grid position.
#pragma once

#include "damacy_limits.h"
#include "planner/planner.h"

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Per-chunk record materialized at wave dispatch time.
  struct assemble_chunk
  {
    uint64_t src_base_byte_off;        // arena byte offset of chunk start
    uint16_t sample_idx_in_batch;      // index into d_samples[]
    uint32_t chunk_d[DAMACY_MAX_RANK]; // chunk grid position within sample
  };

  // Launch the assemble kernel on `stream`. Inputs (device-resident
  // unless noted):
  //   rank           — spatial rank shared by all chunks in the wave
  //                    (1..DAMACY_MAX_RANK supported; ranks 1..8 use a
  //                    compile-time-templated kernel, higher ranks use
  //                    the runtime-rank fallback)
  //   d_samples      — sample_plan[] for the batch slot (size n_samples)
  //   n_samples      — number of samples in the batch slot
  //   d_chunks       — assemble_chunk[] for the wave (size n_chunks)
  //   n_chunks       — number of chunks in the wave
  //   max_blocks_per_chunk — gridDim.x; max over the wave's chunks of
  //                          ∏ ceil_div(S[d], T[d]). Pre-computed host
  //                          side; chunks with fewer blocks early-return
  //                          on the surplus.
  //   arena_base / output_base — base device pointers
  //   bpe            — bytes per element (1, 2, 4, or 8)
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
                      uint32_t bpe);

  // Compute blocks-per-chunk for a sample given its dims and the kernel's
  // block-tile shape. Host-side helper used by damacy.c when packing the
  // wave's metadata. Returns 0 for unsupported ranks.
  uint32_t assemble_blocks_per_chunk(uint8_t rank,
                                     const struct sample_dim* dims);

#ifdef __cplusplus
}
#endif
