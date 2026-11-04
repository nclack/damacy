#pragma once

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  int decoder_status_reduce_launch(CUstream stream,
                                   const int* d_statuses,
                                   const size_t* d_actual_sizes,
                                   const size_t* d_expected_sizes,
                                   uint32_t* d_error_counter,
                                   uint32_t n);

#ifdef __cplusplus
}
#endif
