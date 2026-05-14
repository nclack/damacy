// NUMA-aware pinned-host placement + thread affinity.
//
// Three jobs:
//   1. Resolve the GPU's nearest host-NUMA node (CUDA driver attr →
//      sysfs fallback).
//   2. numa_scope_enter / numa_scope_exit — temporarily pin the calling
//      thread to that node so first-touch of cuMemAllocHost pages lands
//      on the right node.
//   3. numa_apply_thread_affinity — pin a worker thread permanently
//      (io_queue, scheduler) to the node's CPU set.
//
// All entry points are NULL-safe and graceful no-ops when libnuma is
// absent at build time (DAMACY_NUMA off) or NUMA is unavailable at
// runtime (numa_available() < 0, single-node, etc.).
#pragma once

#include <cuda.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum numa_strategy
  {
    NUMA_AUTO = 0, // resolve GPU node + apply
    NUMA_DISABLED, // no-op (no resolution, no affinity)
    NUMA_PIN_TO,   // use the explicit numa_node override
  };

  // Resolved NUMA placement plan. node < 0 means "no pinning"; produced
  // by numa_init when strategy is disabled, the platform lacks libnuma,
  // numa_available() < 0, the driver attr returns -1, or the resolved
  // node has no CPUs. cpu_mask is opaque; numa_apply_thread_affinity
  // consumes it.
  struct numa_resolved
  {
    int node;
    // Saved cpu-mask blob. Sized to fit cpu_set_t; treated as opaque by
    // callers. 1024 bits covers up to 1024 logical CPUs which is more
    // than today's flagship sockets.
    uint8_t cpu_mask[128];
  };

  // Resolve the GPU's host-NUMA node and populate `out`. Logs once at
  // INFO if libnuma is missing or numa_available() < 0; that log line
  // is emitted from the first call and silenced thereafter.
  void numa_init(enum numa_strategy strategy,
                 int override_node,
                 CUdevice cu_device,
                 struct numa_resolved* out);

  // Temporarily pin the calling thread to the resolved node's CPU set,
  // saving the prior mask in `saved`. Pair with numa_scope_exit. No-op
  // (and writes a zero `saved`) when `r->node < 0` or libnuma is off.
  void numa_scope_enter(const struct numa_resolved* r, uint8_t saved[128]);

  // Restore the affinity captured by numa_scope_enter. Safe to call when
  // numa_scope_enter was a no-op (recognizes the zero `saved`).
  void numa_scope_exit(const uint8_t saved[128]);

  // Permanently set the current thread's CPU affinity to the resolved
  // node's CPU set. Intended to run from worker-thread entry. No-op
  // when `r->node < 0` or libnuma is off. Logs the thread's resulting
  // mask at TRACE for diagnosis.
  void numa_apply_thread_affinity(const struct numa_resolved* r,
                                  const char* thread_label);

#ifdef __cplusplus
}
#endif
