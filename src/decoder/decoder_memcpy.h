#pragma once

#include <stdint.h>

// Forward-declare CUstream so non-CUDA consumers (the host parse module)
// can use gpu_memcpy_op without pulling in cuda.h. Matches cuda.h's
// definition; the launch implementation lives in a .cu and #include's
// the real header.
typedef struct CUstream_st* CUstream;

#ifdef __cplusplus
extern "C"
{
#endif

  struct gpu_memcpy_op
  {
    const void* d_src;
    void* d_dst;
    uint32_t nbytes;
  };

  // Launch one CUDA block per op; each block does a coalesced byte copy
  // of `nbytes` from d_src to d_dst. d_ops is a device pointer.
  int decoder_memcpy_launch(CUstream stream,
                            const struct gpu_memcpy_op* d_ops,
                            uint32_t n_ops);

#ifdef __cplusplus
}
#endif
