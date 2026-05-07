#pragma once

#include "decoder/blosc1.h"

#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Reverse blosc1 bit-shuffle in place. Same constraints as
  // gpu_unshuffle_launch: all ops in one launch must share the same
  // blocksize; no partial-tail handling. blocksize/typesize must be a
  // multiple of 8 (bitshuffle's own precondition).
  int gpu_bitunshuffle_launch(CUstream stream,
                              const struct gpu_shuffle_op* d_ops,
                              uint32_t n_ops,
                              uint32_t blocksize);

#ifdef __cplusplus
}
#endif
