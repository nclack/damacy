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
// All entry points are NULL-safe and graceful no-ops when NUMA is
// unavailable (platform_numa_available()==0), the host is single-node,
// or the GPU's host-NUMA node can't be resolved.
#pragma once

#include "damacy.h"
#include "platform/numa.h"

#include <cuda.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Resolved NUMA placement plan. node < 0 means "no pinning"; produced
  // by numa_init when strategy is disabled, NUMA is unavailable, the
  // driver attr returns -1, or the resolved node has no CPUs.
  struct numa_resolved
  {
    int node;
    struct platform_cpu_mask cpu_mask;
  };

  // Resolve the GPU's host-NUMA node and populate `out`. Logs once at
  // INFO if NUMA is unavailable; that log line is silenced thereafter.
  void numa_init(enum damacy_numa_strategy strategy,
                 int override_node,
                 CUdevice cu_device,
                 struct numa_resolved* out);

  // Temporarily pin the calling thread to the resolved node's CPU set,
  // saving the prior mask in `*saved`. Pair with numa_scope_exit. No-op
  // (and writes an empty `*saved`) when `r->node < 0`.
  void numa_scope_enter(const struct numa_resolved* r,
                        struct platform_cpu_mask* saved);

  // Restore the affinity captured by numa_scope_enter. Safe to call when
  // numa_scope_enter was a no-op (recognizes the empty mask).
  void numa_scope_exit(const struct platform_cpu_mask* saved);

  // Permanently set the current thread's CPU affinity to the resolved
  // node's CPU set. Intended to run from worker-thread entry. No-op
  // when `r->node < 0`. Logs the thread's resulting mask at TRACE for
  // diagnosis.
  void numa_apply_thread_affinity(const struct numa_resolved* r,
                                  const char* thread_label);

#ifdef __cplusplus
}
#endif
