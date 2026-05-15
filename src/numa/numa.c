// _GNU_SOURCE must come before any libc header that conditionally pulls
// in cpu_set_t / pthread_setaffinity_np / CPU_*.
#define _GNU_SOURCE

#include "numa/numa.h"

#include "log/log.h"

#include <stdio.h>
#include <string.h>

#if defined(__linux__)

#include <assert.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>

// Minimal redeclaration of libnuma's `struct bitmask`. We dlopen
// libnuma instead of linking it so a single binary ships across hosts
// with and without numactl installed; that means we cannot include
// <numa.h>. The (size, maskp) pair has been stable since libnuma 2.0
// — older fields don't exist and newer ones (if any) come after.
struct numa_bitmask
{
  unsigned long size;
  unsigned long* maskp;
};

typedef int (*pfn_numa_available)(void);
typedef int (*pfn_numa_max_node)(void);
typedef struct numa_bitmask* (*pfn_numa_allocate_cpumask)(void);
typedef void (*pfn_numa_free_cpumask)(struct numa_bitmask*);
typedef int (*pfn_numa_node_to_cpus)(int, struct numa_bitmask*);

// numa_resolved.cpu_mask is a raw 128-byte blob — make sure cpu_set_t
// fits. glibc's default CPU_SETSIZE is 1024 bits = 128 bytes; if a
// future toolchain expands it, callers would silently truncate.
static_assert(sizeof(cpu_set_t) <= 128,
              "cpu_set_t larger than numa_resolved.cpu_mask");

// Resolved libnuma symbol table. `handle` doubles as the lazy-load
// state machine: NULL = not yet attempted, (void*)-1 = dlopen/dlsym
// failed (don't retry), anything else = a live handle with the function
// pointers populated.
#define LIBNUMA_FAILED ((void*)-1)
static struct
{
  void* handle;
  pfn_numa_available available;
  pfn_numa_max_node max_node;
  pfn_numa_allocate_cpumask allocate_cpumask;
  pfn_numa_free_cpumask free_cpumask;
  pfn_numa_node_to_cpus node_to_cpus;
} g_libnuma;

// One-shot guard so we only log "libnuma unavailable" once across the
// process lifetime — repeated damacy_create calls shouldn't spam.
static int g_numa_unavailable_logged = 0;

// dlopen libnuma.so.1 and resolve the 5 symbols we need. Returns 1 on
// full success, 0 on any failure (which is then memoized via the
// LIBNUMA_FAILED sentinel so the next call short-circuits). Not thread-
// safe vs. the very first call; numa_init is invoked from damacy_create
// before any workers spin up.
static int
try_load_libnuma(void)
{
  if (g_libnuma.handle == LIBNUMA_FAILED)
    return 0;
  if (g_libnuma.handle != NULL)
    return 1;

  void* h = dlopen("libnuma.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (!h) {
    g_libnuma.handle = LIBNUMA_FAILED;
    return 0;
  }

  g_libnuma.available = (pfn_numa_available)dlsym(h, "numa_available");
  g_libnuma.max_node = (pfn_numa_max_node)dlsym(h, "numa_max_node");
  g_libnuma.allocate_cpumask =
    (pfn_numa_allocate_cpumask)dlsym(h, "numa_allocate_cpumask");
  g_libnuma.free_cpumask = (pfn_numa_free_cpumask)dlsym(h, "numa_free_cpumask");
  g_libnuma.node_to_cpus = (pfn_numa_node_to_cpus)dlsym(h, "numa_node_to_cpus");

  if (!g_libnuma.available || !g_libnuma.max_node ||
      !g_libnuma.allocate_cpumask || !g_libnuma.free_cpumask ||
      !g_libnuma.node_to_cpus) {
    dlclose(h);
    g_libnuma.handle = LIBNUMA_FAILED;
    return 0;
  }

  g_libnuma.handle = h;
  return 1;
}

// Probe libnuma at runtime. Returns 1 if usable, 0 if libnuma can't be
// loaded or the kernel doesn't support NUMA (numa_available() < 0).
// Logs a single INFO message on first failure.
static int
runtime_numa_ok(void)
{
  if (!try_load_libnuma()) {
    if (!g_numa_unavailable_logged) {
      log_info("numa: libnuma.so.1 not loadable — NUMA placement disabled");
      g_numa_unavailable_logged = 1;
    }
    return 0;
  }
  if (g_libnuma.available() < 0) {
    if (!g_numa_unavailable_logged) {
      log_info("numa: numa_available()<0 — NUMA placement disabled");
      g_numa_unavailable_logged = 1;
    }
    return 0;
  }
  return 1;
}

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

// CUDA driver attribute path. Returns -1 if the attribute is unsupported
// or reports "no NUMA".
static int
resolve_from_driver(CUdevice cu_dev)
{
  int node = -1;
#ifdef CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID
  if (cuDeviceGetAttribute(&node, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, cu_dev) !=
      CUDA_SUCCESS)
    return -1;
#else
  (void)cu_dev;
#endif
  return node;
}

// Build a cpu_set_t for `node` and serialize into `out` (caller-owned
// 128-byte blob). Returns 0 on success, non-zero if the node has no
// CPUs or numa_node_to_cpus fails.
static int
build_node_cpu_mask(int node, uint8_t out[128])
{
  struct numa_bitmask* bm = g_libnuma.allocate_cpumask();
  if (!bm)
    return 1;
  if (g_libnuma.node_to_cpus(node, bm) != 0) {
    g_libnuma.free_cpumask(bm);
    return 1;
  }
  cpu_set_t cs;
  CPU_ZERO(&cs);
  int popcnt = 0;
  const unsigned long* words = bm->maskp;
  const unsigned long max_bits = bm->size;
  for (unsigned long b = 0; b < max_bits && b < CPU_SETSIZE; ++b) {
    if (words[b / (8 * sizeof(unsigned long))] &
        (1UL << (b % (8 * sizeof(unsigned long))))) {
      CPU_SET((int)b, &cs);
      ++popcnt;
    }
  }
  g_libnuma.free_cpumask(bm);
  if (popcnt == 0)
    return 1;
  memcpy(out, &cs, sizeof(cs) < 128 ? sizeof(cs) : 128);
  return 0;
}

void
numa_init(enum numa_strategy strategy,
          int override_node,
          CUdevice cu_device,
          struct numa_resolved* out)
{
  memset(out, 0, sizeof(*out));
  out->node = -1;

  if (strategy == NUMA_DISABLED) {
    log_info("numa: strategy=disabled — no pinning");
    return;
  }
  if (!runtime_numa_ok())
    return;
  // Single-node box: nothing to pin against. numa_max_node() returns the
  // highest configured node id; 0 means just node 0 exists. Treat as
  // no-op so single-socket CI runs don't pretend to pin.
  if (g_libnuma.max_node() <= 0) {
    log_info("numa: max_node=%d — single-node host, no pinning",
             g_libnuma.max_node());
    return;
  }

  int node = -1;
  const char* source = "?";
  if (strategy == NUMA_PIN_TO) {
    node = override_node;
    source = "override";
    if (node < 0 || node > g_libnuma.max_node()) {
      log_warn("numa: pin_to node=%d out of range [0..%d] — no pinning",
               node,
               g_libnuma.max_node());
      return;
    }
  } else {
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

  if (build_node_cpu_mask(node, out->cpu_mask) != 0) {
    log_warn("numa: node=%d has no CPUs in its cpumask — no pinning", node);
    return;
  }
  out->node = node;
  log_info("numa: resolved GPU host-NUMA node=%d (source=%s)", node, source);
}

void
numa_scope_enter(const struct numa_resolved* r, uint8_t saved[128])
{
  memset(saved, 0, 128);
  if (!r || r->node < 0)
    return;
  // Capture current affinity before overriding so we can restore.
  cpu_set_t cur;
  CPU_ZERO(&cur);
  if (pthread_getaffinity_np(pthread_self(), sizeof(cur), &cur) != 0) {
    log_warn("numa: pthread_getaffinity_np failed; scope is best-effort");
    return;
  }
  memcpy(saved, &cur, sizeof(cur) < 128 ? sizeof(cur) : 128);
  cpu_set_t target;
  memcpy(&target, r->cpu_mask, sizeof(target));
  if (pthread_setaffinity_np(pthread_self(), sizeof(target), &target) != 0) {
    log_warn("numa: pthread_setaffinity_np(node=%d) failed for scope", r->node);
    // Clear saved so the matching exit doesn't restore garbage.
    memset(saved, 0, 128);
  }
}

void
numa_scope_exit(const uint8_t saved[128])
{
  if (!saved)
    return;
  // All-zeros == no-op (we never wrote a real cpu_set_t into `saved`).
  int any = 0;
  for (size_t i = 0; i < 128; ++i)
    if (saved[i]) {
      any = 1;
      break;
    }
  if (!any)
    return;
  cpu_set_t cur;
  memcpy(&cur, saved, sizeof(cur));
  if (pthread_setaffinity_np(pthread_self(), sizeof(cur), &cur) != 0)
    log_warn("numa: pthread_setaffinity_np(restore) failed");
}

void
numa_apply_thread_affinity(const struct numa_resolved* r,
                           const char* thread_label)
{
  if (!r || r->node < 0)
    return;
  cpu_set_t target;
  memcpy(&target, r->cpu_mask, sizeof(target));
  if (pthread_setaffinity_np(pthread_self(), sizeof(target), &target) != 0) {
    log_warn("numa: pthread_setaffinity_np(node=%d) failed for %s",
             r->node,
             thread_label ? thread_label : "thread");
    return;
  }
  // Log the resolved CPUs for the thread; useful for verifying single-
  // socket / SLURM-managed runs leave the right shape.
  cpu_set_t got;
  CPU_ZERO(&got);
  if (pthread_getaffinity_np(pthread_self(), sizeof(got), &got) == 0) {
    int first = -1, last = -1, count = 0;
    for (int i = 0; i < CPU_SETSIZE; ++i)
      if (CPU_ISSET(i, &got)) {
        if (first < 0)
          first = i;
        last = i;
        ++count;
      }
    log_trace("numa: pinned %s to node=%d cpus=[%d..%d] (%d cores)",
              thread_label ? thread_label : "thread",
              r->node,
              first,
              last,
              count);
  }
}

#else /* !__linux__ */

// Non-Linux: libnuma is Linux-only and dlopen-able only there. Compile
// the control surface as no-ops so call sites stay #ifdef-free.

static int g_numa_off_logged = 0;

void
numa_init(enum numa_strategy strategy,
          int override_node,
          CUdevice cu_device,
          struct numa_resolved* out)
{
  (void)strategy;
  (void)override_node;
  (void)cu_device;
  memset(out, 0, sizeof(*out));
  out->node = -1;
  if (!g_numa_off_logged) {
    log_info("numa: non-Linux platform — no pinning");
    g_numa_off_logged = 1;
  }
}

void
numa_scope_enter(const struct numa_resolved* r, uint8_t saved[128])
{
  (void)r;
  memset(saved, 0, 128);
}

void
numa_scope_exit(const uint8_t saved[128])
{
  (void)saved;
}

void
numa_apply_thread_affinity(const struct numa_resolved* r,
                           const char* thread_label)
{
  (void)r;
  (void)thread_label;
}

#endif /* __linux__ */
