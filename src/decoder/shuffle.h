#pragma once

#include "decoder/blosc1.h"

#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Reverse blosc1 byte-shuffle. Per op, processes nblocks_full blocks
  // of `op.blocksize` bytes each. blocksize is read per-op so a launch
  // can mix ops with different blocksizes.
  //
  // The kernel reads each block from `d_buf` (the decompressed slot in
  // dev_decompressed) and writes the transposed result back to
  // d_buf. To avoid blocksize > 64 KB shmem caps and to handle the
  // non-in-place transpose, the caller supplies a scratch buffer
  // covering the wave's dev_decompressed extent. The kernel internally
  // copies each block to its corresponding offset in scratch (under
  // __syncthreads), then reads from scratch and scatters to d_buf.
  int gpu_unshuffle_launch(CUstream stream,
                           const struct gpu_shuffle_op* d_ops,
                           uint32_t n_ops,
                           const void* dev_decompressed_base,
                           void* scratch_base);

#ifdef __cplusplus
}
#endif
