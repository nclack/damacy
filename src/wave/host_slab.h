// Pinned-host slab slot — one wave's worth of compressed bytes plus
// the store_read records that drove the IO. A pool of these slots
// (>= DAMACY_N_WAVES) lets peel + IO for upcoming waves complete
// before a wave struct is free, shrinking the wave-boundary gap on
// stream_decode.
//
// Lifecycle:
//   FREE  → peel writes bytes + submits IO → IO
//   IO    → store_event_query succeeds     → READY
//   READY → bind to a free wave            → BUSY
//   BUSY  → bulk_h2d_end fires on stream_h2d → FREE
#pragma once

#include "store/store.h"

#include <stdint.h>

enum slot_state
{
  SLOT_FREE = 0,
  SLOT_IO,
  SLOT_READY,
  SLOT_BUSY,
};

struct host_slab_slot
{
  enum slot_state state;
  void* buf; // pinned host (cuMemAllocHost)
  uint64_t cap;
  uint64_t used_bytes;

  uint16_t batch_pool_slot;
  uint32_t batch_chunk_offset;
  uint32_t n_chunks;

  struct store_read* store_reads; // capacity DAMACY_MAX_CHUNKS_PER_WAVE
  struct store_event io_event;
  uint8_t is_fill_wave; // 1 if every chunk in this slot is a fill chunk
                        // (no IO submitted); polling skips io_event entirely.

  uint64_t io_t_start_ns; // peel-submit wall-clock
  uint64_t io_t_end_ns;   // SLOT_IO → SLOT_READY transition
  uint64_t io_bytes;
};

// Allocate one slot's pinned buffer + store_reads array. Returns 0 on
// success, 1 on failure (after self-cleanup). cuda_skip=1 in
// slot_destroy leaks the pinned buffer (used when the CUDA context is
// no longer valid) but releases the heap.
int
slot_init(struct host_slab_slot* slot, uint64_t cap);

void
slot_destroy(struct host_slab_slot* slot, int cuda_skip);

// SLOT_BUSY → SLOT_FREE. Called when bulk_h2d_end fires on stream_h2d
// (the host bytes have been consumed by the GPU copy) or from the
// failure cleanup path.
void
slot_release(struct host_slab_slot* slot);

// Linear scans across `slots[0..n)`. find_* returns the index of the
// first matching slot, or -1 if none. any_* returns 1/0.
int
host_slab_find_free(const struct host_slab_slot* slots, uint8_t n);
int
host_slab_find_ready(const struct host_slab_slot* slots, uint8_t n);
int
host_slab_any_in_flight(const struct host_slab_slot* slots, uint8_t n);
int
host_slab_any_free(const struct host_slab_slot* slots, uint8_t n);
