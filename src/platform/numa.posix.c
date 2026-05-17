// _GNU_SOURCE must come before any libc header that conditionally pulls
// in cpu_set_t / pthread_setaffinity_np / CPU_*.
#define _GNU_SOURCE

#include "platform/numa.h"
#include "platform/platform.h"

#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <string.h>

// Minimal redeclaration of libnuma's `struct bitmask`. We dlopen libnuma
// instead of linking it so a single binary ships across hosts with and
// without numactl installed; that means we cannot include <numa.h>. The
// (size, maskp) pair has been stable since libnuma 2.0 — older fields
// don't exist and newer ones (if any) come after.
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

static_assert(sizeof(cpu_set_t) <=
                sizeof(((struct platform_cpu_mask*)0)->bytes),
              "cpu_set_t larger than platform_cpu_mask blob");

// Lazy-load state machine: NULL = not yet attempted, LIBNUMA_FAILED =
// dlopen/dlsym failed (don't retry), anything else = a live handle with
// the function pointers populated. Atomic because platform_numa_available
// may be called from concurrent damacy_create instances.
#define LIBNUMA_FAILED ((void*)-1)
static struct
{
  _Atomic(void*) handle;
  pfn_numa_available numa_available;
  pfn_numa_max_node numa_max_node;
  pfn_numa_allocate_cpumask numa_allocate_cpumask;
  pfn_numa_free_cpumask numa_free_cpumask;
  pfn_numa_node_to_cpus numa_node_to_cpus;
} g_libnuma;

#define DLSYM_BIND(handle, table, sym)                                         \
  (*(void**)&(table).sym = platform_dlsym((handle), #sym))

// Concurrent first-callers may each dlopen; one wins the CAS publish,
// losers dlclose their copy.
static int
try_load_libnuma(void)
{
  void* cur = atomic_load_explicit(&g_libnuma.handle, memory_order_acquire);
  if (cur == LIBNUMA_FAILED)
    return 0;
  if (cur != NULL)
    return 1;

  void* h = platform_dlopen("libnuma.so.1");
  if (!h) {
    atomic_store_explicit(
      &g_libnuma.handle, LIBNUMA_FAILED, memory_order_release);
    return 0;
  }

  DLSYM_BIND(h, g_libnuma, numa_available);
  DLSYM_BIND(h, g_libnuma, numa_max_node);
  DLSYM_BIND(h, g_libnuma, numa_allocate_cpumask);
  DLSYM_BIND(h, g_libnuma, numa_free_cpumask);
  DLSYM_BIND(h, g_libnuma, numa_node_to_cpus);

  if (!g_libnuma.numa_available || !g_libnuma.numa_max_node ||
      !g_libnuma.numa_allocate_cpumask || !g_libnuma.numa_free_cpumask ||
      !g_libnuma.numa_node_to_cpus) {
    platform_dlclose(h);
    atomic_store_explicit(
      &g_libnuma.handle, LIBNUMA_FAILED, memory_order_release);
    return 0;
  }

  void* expected = NULL;
  if (!atomic_compare_exchange_strong_explicit(&g_libnuma.handle,
                                               &expected,
                                               h,
                                               memory_order_acq_rel,
                                               memory_order_acquire)) {
    platform_dlclose(h);
  }
  return 1;
}

int
platform_numa_available(void)
{
  if (!try_load_libnuma())
    return 0;
  return g_libnuma.numa_available() >= 0 ? 1 : 0;
}

int
platform_numa_max_node(void)
{
  if (!platform_numa_available())
    return -1;
  return g_libnuma.numa_max_node();
}

int
platform_numa_node_cpu_mask(int node, struct platform_cpu_mask* out)
{
  memset(out, 0, sizeof(*out));
  if (!platform_numa_available())
    return 1;
  if (node < 0 || node > g_libnuma.numa_max_node())
    return 1;
  struct numa_bitmask* bm = g_libnuma.numa_allocate_cpumask();
  if (!bm)
    return 1;
  if (g_libnuma.numa_node_to_cpus(node, bm) != 0) {
    g_libnuma.numa_free_cpumask(bm);
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
  g_libnuma.numa_free_cpumask(bm);
  if (popcnt == 0)
    return 1;
  memcpy(out->bytes, &cs, sizeof(cs));
  return 0;
}

int
platform_thread_affinity_get(struct platform_cpu_mask* out)
{
  memset(out, 0, sizeof(*out));
  cpu_set_t cs;
  CPU_ZERO(&cs);
  if (pthread_getaffinity_np(pthread_self(), sizeof(cs), &cs) != 0)
    return 1;
  memcpy(out->bytes, &cs, sizeof(cs));
  return 0;
}

int
platform_thread_affinity_set(const struct platform_cpu_mask* mask)
{
  cpu_set_t cs;
  memcpy(&cs, mask->bytes, sizeof(cs));
  return pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) == 0 ? 0 : 1;
}

int
platform_cpu_mask_is_empty(const struct platform_cpu_mask* mask)
{
  if (!mask)
    return 1;
  for (size_t i = 0; i < sizeof(mask->bytes); ++i)
    if (mask->bytes[i])
      return 0;
  return 1;
}

int
platform_cpu_mask_describe(const struct platform_cpu_mask* mask,
                           int* first,
                           int* last,
                           int* count)
{
  if (!mask || platform_cpu_mask_is_empty(mask))
    return 1;
  cpu_set_t cs;
  memcpy(&cs, mask->bytes, sizeof(cs));
  int f = -1, l = -1, c = 0;
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    if (CPU_ISSET(i, &cs)) {
      if (f < 0)
        f = i;
      l = i;
      ++c;
    }
  }
  if (first)
    *first = f;
  if (last)
    *last = l;
  if (count)
    *count = c;
  return 0;
}
