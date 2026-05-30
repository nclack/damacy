// Pinned-host slab slot — one wave's worth of compressed bytes plus
// the store_read records that drove the IO. A pool of these slots
// (>= DAMACY_N_WAVES) lets peel + IO for upcoming waves complete
// before a wave struct is free, shrinking the wave-boundary gap on
// stream_decode.
//
// Lifecycle:
//   FREE → PEELING (peel reserve) → IO (peel commit, after async submit)
//        → READY (store_event_query) → BUSY (bind to wave)
//        → FREE (bulk_h2d_end on stream_h2d)
#pragma once

#include "platform/platform.h"
#include "store/store.h"

#include <stdint.h>

enum slot_state
{
  SLOT_FREE = 0,
  SLOT_PEELING,
  SLOT_IO,
  SLOT_READY,
  SLOT_BUSY,
};

struct host_slab_slot
{
  enum slot_state state;
  // Pinned host buffer used by the host-staging IO path. NULL when the
  // slot was allocated as device-only (GDS). Cap is the slot's logical
  // capacity in either case; reads use cap to gate chunk packing.
  void* buf;
  // Device buffer used by the GDS path; cuFile reads land here
  // directly. NULL when the slot is host-only.
  void* dev_buf;
  uint64_t cap;
  uint64_t used_bytes;

  uint16_t render_job_idx;
  uint16_t batch_pool_slot;
  uint32_t batch_chunk_offset;
  uint32_t n_chunks;

  struct store_read* store_reads; // capacity = wave_pool.max_chunks_per_wave
  struct store_event io_event;
  uint8_t is_fill_wave; // 1 if every chunk in this slot is a fill chunk
                        // (no IO submitted); polling skips io_event entirely.

  // tic'd at peel-submit; toc at SLOT_IO→SLOT_READY captures io_ms
  // AND advances last_ns so the bind-time toc measures bind-wait.
  struct platform_clock io_clock;
  float io_ms;
  uint64_t io_bytes;
};

// Allocate one slot's pinned buffer + store_reads array. `host_cap` is
// the pinned host buffer size (0 = skip; GDS-only slots). `dev_cap`
// is the device buffer size (0 = skip; host-staging-only slots). At
// least one must be non-zero. Returns 0 on success, 1 on failure
// (after self-cleanup). cuda_skip=1 in slot_destroy leaks the pinned
// buffer (used when the CUDA context is no longer valid) but releases
// the heap.
int
slot_init(struct host_slab_slot* slot,
          uint32_t max_chunks_per_wave,
          uint64_t host_cap,
          uint64_t dev_cap);

void
slot_destroy(struct host_slab_slot* slot, int cuda_skip);

// SLOT_BUSY → SLOT_FREE. Called when compressed input is consumed
// (bulk_h2d_end for host staging, h2d_end for GDS) or from the failure
// cleanup path.
void
slot_release(struct host_slab_slot* slot);

// Linear scans across `slots[0..n)`. find_* returns the index of the
// first matching slot, or -1 if none. any_* returns 1/0.
//
// host_slab_any_in_flight returns 1 for any non-FREE state, including
// the transient SLOT_PEELING window (peel reserve → peel commit). This
// is intentional: damacy_pop relies on it to know a peel may produce
// future ready batches even when no wave has actually launched yet.
// Any new non-FREE state added later must keep this property.
int
host_slab_find_free(const struct host_slab_slot* slots, uint8_t n);
int
host_slab_find_ready(const struct host_slab_slot* slots, uint8_t n);
int
host_slab_any_in_flight(const struct host_slab_slot* slots, uint8_t n);
int
host_slab_any_free(const struct host_slab_slot* slots, uint8_t n);
