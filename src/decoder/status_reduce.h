// Tiny reduce kernel that counts non-zero entries in an int-sized status
// array (typically nvcomp's per-batch d_statuses) and atomicAdds the count
// into a uint32 device counter. Decoupled from any specific codec so the
// shared-host orchestration path can fold both nvcomp Zstd and LZ4 status
// arrays into one error tally.
#pragma once

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // d_statuses points to `n` int32-sized status values (nvcompStatus_t is
  // an enum that fits in int). d_error_counter is a single uint32 in
  // device memory; the kernel atomicAdds the non-zero count without
  // first zeroing — caller is responsible for clearing the counter
  // upstream (typically as part of the same memset that clears
  // blosc1_totals before parse).
  int decoder_status_reduce_launch(CUstream stream,
                                   const int* d_statuses,
                                   uint32_t* d_error_counter,
                                   uint32_t n);

#ifdef __cplusplus
}
#endif
