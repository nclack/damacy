#pragma once

#include <stdatomic.h>
#include <stdint.h>

static inline void
atomic_u16_observe_max(_Atomic(uint16_t)* slot, uint16_t value)
{
  if (!slot || value == 0)
    return;
  uint16_t cur = atomic_load_explicit(slot, memory_order_relaxed);
  while (value > cur) {
    /* release/acquire pair: any thread that observes a max also observes
     * the writes the producer made before the CAS. */
    if (atomic_compare_exchange_weak_explicit(
          slot, &cur, value, memory_order_release, memory_order_relaxed))
      return;
  }
}
