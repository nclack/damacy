#pragma once

#include <stdatomic.h>
#include <stdint.h>

static inline void
wave_pool_observe_blosc_nblocks(_Atomic(uint16_t)* slot, uint16_t nblocks)
{
  if (!slot || nblocks == 0)
    return;
  uint16_t cur = atomic_load_explicit(slot, memory_order_relaxed);
  while (nblocks > cur) {
    if (atomic_compare_exchange_weak_explicit(
          slot, &cur, nblocks, memory_order_relaxed, memory_order_relaxed))
      return;
  }
}
