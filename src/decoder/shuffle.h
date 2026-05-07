#pragma once

#include "decoder/blosc1.h"

#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Reverse blosc1 byte-shuffle in place. Per op, processes nblocks_full
  // blocks of `blocksize` bytes each (partial last block via tail_nbytes
  // is unsupported here; spike fixtures don't use it).
  //
  // All ops in one launch must share the same blocksize; the launcher
  // sizes shared memory for that blocksize.
  int gpu_unshuffle_launch(CUstream stream,
                           const struct gpu_shuffle_op* d_ops,
                           uint32_t n_ops,
                           uint32_t blocksize);

#ifdef __cplusplus
}
#endif
