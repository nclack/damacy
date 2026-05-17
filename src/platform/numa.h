#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Opaque; do not inspect. 128 bytes fits glibc cpu_set_t (1024 bits).
  struct platform_cpu_mask
  {
    uint8_t bytes[128];
  };

  int platform_numa_available(void);
  int platform_numa_max_node(void);

  // Out-mask zero-init on failure so callers can ignore the return.
  int platform_numa_node_cpu_mask(int node, struct platform_cpu_mask* out);
  int platform_thread_affinity_get(struct platform_cpu_mask* out);

  // Existing affinity is left as-is on failure.
  int platform_thread_affinity_set(const struct platform_cpu_mask* mask);

  // NULL-safe.
  int platform_cpu_mask_is_empty(const struct platform_cpu_mask* mask);
  int platform_cpu_mask_describe(const struct platform_cpu_mask* mask,
                                 int* first,
                                 int* last,
                                 int* count);

#ifdef __cplusplus
}
#endif
