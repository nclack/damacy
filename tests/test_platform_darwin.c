#include "expect.h"
#include "platform/numa.h"
#include "platform/platform.h"

#include <stdint.h>
#include <string.h>
#include <unistd.h>

static int
test_memory_and_threads(void)
{
  EXPECT(platform_page_size() == (size_t)sysconf(_SC_PAGESIZE));
  EXPECT(platform_page_alignment() == platform_page_size());
  EXPECT(platform_default_thread_count() > 0);
  void* ptr = platform_aligned_alloc(64, 128);
  EXPECT(ptr && (uintptr_t)ptr % 64 == 0);
  platform_aligned_free(ptr);
  return 0;
}

static int
test_unavailable_affinity(void)
{
  EXPECT(!platform_numa_available());
  EXPECT(platform_numa_max_node() == -1);
  struct platform_cpu_mask mask;
  memset(&mask, 0xff, sizeof mask);
  EXPECT(platform_numa_node_cpu_mask(0, &mask) != 0);
  EXPECT(platform_cpu_mask_is_empty(&mask));
  memset(&mask, 0xff, sizeof mask);
  EXPECT(platform_thread_affinity_get(&mask) != 0);
  EXPECT(platform_cpu_mask_is_empty(&mask));
  EXPECT(platform_thread_affinity_set(&mask) != 0);
  int first = 0, last = 0, count = 1;
  EXPECT(platform_cpu_mask_describe(&mask, &first, &last, &count) != 0);
  EXPECT(first == -1 && last == -1 && count == 0);
  return 0;
}

int
main(void)
{
  RUN(test_memory_and_threads);
  RUN(test_unavailable_affinity);
  return 0;
}
