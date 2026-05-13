// Phase 6 observability: NVTX wrappers + named CUDA streams.
//
// Compiled into call-site no-ops when DAMACY_NVTX_ENABLED=0 (the
// inline-in-header path below), so unbuilt timeline calls cost nothing
// in release builds where Nsight isn't attached.
//
// All ranges land in a single named domain ("damacy") so Nsight Systems
// renders them on their own swimlane separate from CUDA's own ranges.
//
// Stream naming uses nvtxNameCuStreamA (from nvtx3/nvToolsExtCuda.h);
// portable across CUDA 12.x / 13.x and doesn't require the driver's
// CUstreamAttrValue plumbing.
#pragma once

#include <cuda.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef DAMACY_NVTX_ENABLED
#define DAMACY_NVTX_ENABLED 0
#endif

#if DAMACY_NVTX_ENABLED

  void damacy_nvtx_init(void);
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

#else // DAMACY_NVTX_ENABLED == 0

static inline void
damacy_nvtx_init(void)
{
}
static inline void
damacy_nvtx_range_push(const char* name)
{
  (void)name;
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
static inline void
damacy_nvtx_range_pushf(const char* fmt, ...)
{
  (void)fmt;
}
static inline void
damacy_nvtx_range_pop(void)
{
}
static inline void
damacy_nvtx_mark(const char* name)
{
  (void)name;
}
static inline void
damacy_nvtx_stream_name(CUstream s, const char* name)
{
  (void)s;
  (void)name;
}

#endif // DAMACY_NVTX_ENABLED

// Scoped range: pushes on entry, pops on exit. Use:
//   DAMACY_NVTX_RANGE("peel_wave") { ... }
// The loop runs exactly once; the second iteration's increment pops
// the range. Works under both ON and OFF flavors because the macro
// resolves to the respective push/pop above (no-op when disabled).
#define DAMACY_NVTX_RANGE(name)                                                \
  for (int _nvtx_done = (damacy_nvtx_range_push(name), 0); !_nvtx_done;        \
       _nvtx_done = 1, damacy_nvtx_range_pop())

#ifdef __cplusplus
}
#endif
