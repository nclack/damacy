#include "log/log.h"
#include "numa/numa.h"

#include <string.h>

void
numa_scope_enter(const struct numa_resolved* r, struct platform_cpu_mask* saved)
{
  memset(saved, 0, sizeof(*saved));
  if (!r || r->node < 0)
    return;
  if (platform_thread_affinity_get(saved) != 0) {
    log_warn("numa: thread_affinity_get failed; scope is best-effort");
    return;
  }
  if (platform_thread_affinity_set(&r->cpu_mask) != 0) {
    log_warn("numa: thread_affinity_set(node=%d) failed for scope", r->node);
    // Clear saved so the matching exit doesn't restore garbage.
    memset(saved, 0, sizeof(*saved));
  }
}

void
numa_scope_exit(const struct platform_cpu_mask* saved)
{
  if (platform_cpu_mask_is_empty(saved))
    return;
  if (platform_thread_affinity_set(saved) != 0)
    log_warn("numa: thread_affinity_set(restore) failed");
}

void
numa_apply_thread_affinity(const struct numa_resolved* r,
                           const char* thread_label)
{
  if (!r || r->node < 0)
    return;
  if (platform_thread_affinity_set(&r->cpu_mask) != 0) {
    log_warn("numa: thread_affinity_set(node=%d) failed for %s",
             r->node,
             thread_label ? thread_label : "thread");
    return;
  }
  struct platform_cpu_mask got;
  if (platform_thread_affinity_get(&got) == 0) {
    int first = -1, last = -1, count = 0;
    if (platform_cpu_mask_describe(&got, &first, &last, &count) == 0)
      log_trace("numa: pinned %s to node=%d cpus=[%d..%d] (%d cores)",
                thread_label ? thread_label : "thread",
                r->node,
                first,
                last,
                count);
  }
}
