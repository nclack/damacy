// NVTX wrappers and named CUDA streams.
#pragma once

#include <cuda.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef DAMACY_NVTX_ENABLED
#define DAMACY_NVTX_ENABLED 0
#endif

  void damacy_nvtx_range_push(const char* name);
  // printf-style variant; bounded by an internal thread-local buffer
  // so call sites can encode wave index without owning a formatter.
  void damacy_nvtx_range_pushf(const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;
  void damacy_nvtx_range_pop(void);
  void damacy_nvtx_mark(const char* name);
  void damacy_nvtx_stream_name(CUstream s, const char* name);

#ifdef __cplusplus
}
#endif
