#pragma once

#include "damacy.h"         // enum damacy_status
#include "decoder/blosc1.h" // struct decode_fanout

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

struct gpu_budget;

// Host-resident mirror of decode_fanout. Pointer slots hold device
// addresses; fanout_upload pushes the four arrays to the device SOA.
struct decode_fanout_host
{
  const void** comp_ptrs;
  size_t* comp_sizes;
  void** decomp_ptrs;
  size_t* decomp_buf_sizes;
  struct gpu_memcpy_op* memcpy_ops;
  uint32_t* block_chunk_map;
};

// Next power of 2 ≥ v, with a floor of 1. SIZE_MAX wraps to 0; callers
// bound `v` against a real ceiling before calling. Used by fanout_grow
// and the decoder-scratch grow path to round substream caps so the
// grow path doesn't fire on every wave with a few-substream increment.
size_t
fanout_next_pow2(size_t v);

// Pinned-host and device allocations for one nvcomp fanout. Returns 0 ok,
// 1 on first failure (logged). Partial-failure cleanup relies on `h`
// and `d` being zero-initialized by the caller; the matching
// fanout_free walks each pointer NULL-safely.
int
fanout_alloc_pinned(struct decode_fanout_host* h,
                    struct decode_fanout* d,
                    size_t n);

// Free the pinned-host and device buffers a prior fanout_alloc_pinned
// allocated, zero the SOA structs so they're safe to re-allocate into.
// NULL-safe per pointer.
void
fanout_free_pinned(struct decode_fanout_host* h, struct decode_fanout* d);

// Copy the 4 SOA arrays in lockstep onto `s`.
enum damacy_status
fanout_upload(CUstream s,
              const struct decode_fanout* d,
              const struct decode_fanout_host* h,
              size_t n);

// Grow the fanout SOA pair to fit `need` substreams. *cap is the
// current capacity (in substreams); updated on success. `need` is
// rounded up to the next power of 2 and clamped at
// max_substreams_per_wave. budget is checked + committed for the
// device-resident bytes delta; on OOM the SOA is left unchanged and
// DAMACY_OOM is returned. On CUDA failure the SOA is zeroed (so
// wave_destroy stays safe) and the committed bytes are rolled back.
//
// Caller responsibility: ensure no stream is reading the to-be-freed
// SOA when calling — fanout_grow does NOT synchronize. The wave-pool
// invariant is that the OWNING wave's fanout is only touched while
// that wave is between bind and kick_input's fanout_upload, so the
// other wave's input-stream work is independent.
enum damacy_status
fanout_grow(struct decode_fanout_host* h,
            struct decode_fanout* d,
            uint32_t* cap,
            size_t need,
            uint32_t max_substreams_per_wave,
            struct gpu_budget* budget);
