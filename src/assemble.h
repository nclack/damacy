// Assemble kernel: bytewise gather from per-chunk decompressed buffers
// in a device arena into the output batch tensor.
//
// One launch handles n_chunks chunks. Each chunk has a window (same
// shape in src and dst); the kernel walks elements in the window,
// computing src (chunk-local) and dst (output-tensor) byte offsets via
// per-axis strides. dtype is encoded via bpe (bytes per element).
//
// Step 4: invoked synchronously per batch by damacy.c. Step 5+ shares
// the compute stream with decompress.
#pragma once

#include "limits.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Per-chunk descriptor consumed by the assemble kernel. The kernel
  // reads dims[0..rank); src/dst base offsets already account for the
  // chunk's position in the arena and the sample's slot in the output.
  struct assemble_chunk
  {
    uint64_t src_base_byte_off;           // first window element in arena
    uint64_t dst_base_byte_off;           // first window element in output
    uint32_t win[DAMACY_MAX_RANK];        // window extent per axis
    int64_t src_strides[DAMACY_MAX_RANK]; // chunk row-major strides (elements)
    int64_t dst_strides[DAMACY_MAX_RANK]; // output strides for spatial axes
    uint32_t rank;
  };

  // Launch the assemble kernel on `stream`. `chunks_dev` is a
  // device-resident array of n_chunks descriptors; arena_base /
  // output_base are device pointers. Returns 0 on success.
  int assemble_launch(cudaStream_t stream,
                      const struct assemble_chunk* chunks_dev,
                      uint32_t n_chunks,
                      uint32_t max_window_elements,
                      const void* arena_base,
                      void* output_base,
                      uint32_t bpe);

#ifdef __cplusplus
}
#endif
