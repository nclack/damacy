#pragma once

#include "decoder/blosc1.h"

#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Reverse blosc1 bit-shuffle. Per-op blocksize; uses the same per-wave
  // scratch model as gpu_unshuffle_launch. blocksize/typesize must be a
  // multiple of 8 (bitshuffle's own precondition).
  int gpu_bitunshuffle_launch(CUstream stream,
                              const struct gpu_shuffle_op* d_ops,
                              uint32_t n_ops,
                              const void* dev_decompressed_base,
                              void* scratch_base);

#ifdef __cplusplus
}
#endif
