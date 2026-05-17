#include "numa/numa.h"

#include "log/log.h"
#include "platform/numa.h"

#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

// One-shot guard so we only log "NUMA unavailable" once across the
// process lifetime — repeated damacy_create calls shouldn't spam.
static _Atomic int g_numa_unavailable_logged = 0;

// Linux sysfs fallback for CUDA drivers that don't expose
// CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID. Resolves the GPU's BDF and reads
// /sys/bus/pci/devices/<bdf>/numa_node. The kernel writes -1 there for
// "no NUMA"; we propagate that as a miss.
static int
resolve_from_sysfs(CUdevice cu_dev)
{
  char bus[32] = { 0 };
  if (cuDeviceGetPCIBusId(bus, (int)sizeof(bus), cu_dev) != CUDA_SUCCESS)
    return -1;
  // bus is like "00000000:01:00.0"; lower-case it for sysfs.
  for (char* p = bus; *p; ++p)
    if (*p >= 'A' && *p <= 'Z')
      *p += 32;
  char path[256];
  int n =
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", bus);
  if (n <= 0 || (size_t)n >= sizeof(path))
    return -1;
  FILE* f = fopen(path, "r");
  if (!f)
    return -1;
  int node = -1;
  if (fscanf(f, "%d", &node) != 1)
    node = -1;
  fclose(f);
  return node;
}

// CUDA driver attribute path. Returns -1 if the attribute is
// unsupported, reports "no NUMA", or returns a node id beyond
// platform_numa_max_node() (some misconfigured hosts report stale ids).
static int
resolve_from_driver(CUdevice cu_dev)
{
  int node = -1;
#ifdef CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
  if (cuDeviceGetAttribute(&node, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cu_dev) !=
      CUDA_SUCCESS)
    return -1;
  if (node >= 0 && node > platform_numa_max_node())
    return -1;
#else
  (void)cu_dev;
#endif
  return node;
}

void
numa_init(enum damacy_numa_strategy strategy,
          int override_node,
          CUdevice cu_device,
          struct numa_resolved* out)
{
  memset(out, 0, sizeof(*out));
  out->node = -1;

  if (strategy == DAMACY_NUMA_DISABLED) {
    log_info("numa: strategy=disabled — no pinning");
    return;
  }
  if (!platform_numa_available()) {
    int expected = 0;
    if (atomic_compare_exchange_strong(
          &g_numa_unavailable_logged, &expected, 1))
      log_info("numa: unavailable on this host — NUMA placement disabled");
    return;
  }

  int max_node = platform_numa_max_node();
  int node = -1;
  const char* source = "?";
  if (strategy == DAMACY_NUMA_PIN_TO) {
    // PIN_TO is an explicit user override; honor it on single-node hosts
    // too so node=0 isn't silently dropped to no-op. Out-of-range still
    // falls through to the no-op path with a clear warning.
    node = override_node;
    source = "override";
    if (node < 0 || node > max_node) {
      log_warn("numa: pin_to node=%d out of range [0..%d] — no pinning",
               node,
               max_node);
      return;
    }
  } else {
    // AUTO: nothing useful to pin against on a single-node box.
    if (max_node <= 0) {
      log_info("numa: max_node=%d — single-node host, no pinning", max_node);
      return;
    }
    node = resolve_from_driver(cu_device);
    source = "cuda";
    if (node < 0) {
      node = resolve_from_sysfs(cu_device);
      source = "sysfs";
    }
    if (node < 0) {
      log_info("numa: could not resolve GPU host-NUMA node — no pinning");
      return;
    }
  }

  if (platform_numa_node_cpu_mask(node, &out->cpu_mask) != 0) {
    log_warn("numa: node=%d has no CPUs in its cpumask — no pinning", node);
    return;
  }
  out->node = node;
  log_info(
    "numa: enabled — pinning to host-NUMA node=%d (source=%s)", node, source);
}

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
