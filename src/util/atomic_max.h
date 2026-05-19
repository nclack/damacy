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
    /* monotonic max; no dependent loads, relaxed is sufficient */
    if (atomic_compare_exchange_weak_explicit(
          slot, &cur, value, memory_order_relaxed, memory_order_relaxed))
      return;
  }
}
