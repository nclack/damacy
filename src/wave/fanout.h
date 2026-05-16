// Pinned-host + device SOA fanout used by the zstd decoder batch path.
// Owns alloc / free / upload / grow primitives; wave/ packages a pair
// of these per wave and grows them independently when a wave's
// substream count outpaces the current cap.
#pragma once

#include "damacy.h"         // enum damacy_status
#include "decoder/blosc1.h" // struct nvcomp_fanout

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

struct gpu_budget;

// Host-resident mirror of nvcomp_fanout. Pointer slots hold device
// addresses; fanout_upload pushes the four arrays to the device SOA.
struct nvcomp_fanout_host
{
  const void** comp_ptrs;
  size_t* comp_sizes;
  void** decomp_ptrs;
  size_t* decomp_buf_sizes;
};

// Next power of 2 ≥ v, with a floor of 1. SIZE_MAX wraps to 0; callers
// bound `v` against a real ceiling before calling. Used by fanout_grow
// and the decoder-scratch grow path to round substream caps so the
// grow path doesn't fire on every wave with a few-substream increment.
size_t
fanout_next_pow2(size_t v);

// 4 pinned-host + 4 device allocs for one nvcomp fanout. Returns 0 ok,
// 1 on first failure (logged). Partial-failure cleanup relies on `h`
// and `d` being zero-initialized by the caller; the matching
// fanout_free walks each pointer NULL-safely.
int
fanout_alloc_pinned(struct nvcomp_fanout_host* h,
                    struct nvcomp_fanout* d,
                    size_t n);

// Free the 4 pinned-host + 4 device buffers a prior fanout_alloc_pinned
// allocated, zero the SOA structs so they're safe to re-allocate into.
// NULL-safe per pointer.
void
fanout_free_pinned(struct nvcomp_fanout_host* h, struct nvcomp_fanout* d);

// H2D the 4 SOA arrays in lockstep onto `s`.
enum damacy_status
fanout_upload(CUstream s,
              const struct nvcomp_fanout* d,
              const struct nvcomp_fanout_host* h,
              size_t n);

// Grow the fanout SOA pair to fit `need` substreams. *cap is the
// current capacity (in substreams); updated on success. `need` is
// rounded up to the next power of 2 and clamped at
// DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE. budget is checked + committed
// for the device-resident bytes delta; on OOM the SOA is left
// unchanged and DAMACY_OOM is returned. On CUDA failure the SOA is
// zeroed (so wave_destroy stays safe) and the committed bytes are
// rolled back.
//
// Caller responsibility: ensure no stream is reading the to-be-freed
// SOA when calling — fanout_grow does NOT synchronize. The wave-pool
// invariant is that the OWNING wave's fanout is only touched while
// that wave is between bind and kick_h2d's fanout_upload, so the
// other wave's stream_h2d work is independent.
enum damacy_status
fanout_grow(struct nvcomp_fanout_host* h,
            struct nvcomp_fanout* d,
            uint32_t* cap,
            size_t need,
            struct gpu_budget* budget);
