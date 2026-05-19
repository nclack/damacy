#pragma once

// Observe-and-retain-max for atomic u16; multi-producer /
// single-or-multi-consumer; producer side uses release on the winning
// CAS, consumer side pairs with acquire on the load.

#include <stdatomic.h>
#include <stdint.h>

static inline void
atomic_u16_observe_max(_Atomic(uint16_t)* slot, uint16_t value)
{
  if (!slot || value == 0)
    return;
  uint16_t cur = atomic_load_explicit(slot, memory_order_relaxed);
  while (value > cur) {
    // CAS success = release: publishes the new max to any subsequent
    // acquire-load on `slot`. Failure-path load = relaxed: it's a pure
    // retry-reload with no ordering requirement (C11 also forbids
    // release/acq_rel on the failure order). The consumer side
    // (e.g. chunk_zsubs_upper_bound) uses memory_order_acquire.
    if (atomic_compare_exchange_weak_explicit(
          slot, &cur, value, memory_order_release, memory_order_relaxed))
      return;
  }
}
