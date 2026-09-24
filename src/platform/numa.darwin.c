// macOS does not expose Linux NUMA-node or CPU-mask affinity controls.
#include "platform/numa.h"
#include <stddef.h>
#include <string.h>

int
platform_numa_available(void)
{
  return 0;
}
int
platform_numa_max_node(void)
{
  return -1;
}
int
platform_numa_node_cpu_mask(int node, struct platform_cpu_mask* out)
{
  (void)node;
  if (out)
    memset(out, 0, sizeof(*out));
  return 1;
}
int
platform_thread_affinity_get(struct platform_cpu_mask* out)
{
  if (out)
    memset(out, 0, sizeof(*out));
  return 1;
}
int
platform_thread_affinity_set(const struct platform_cpu_mask* mask)
{
  (void)mask;
  return 1;
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
  (void)mask;
  if (first)
    *first = -1;
  if (last)
    *last = -1;
  if (count)
    *count = 0;
  return 1;
}
