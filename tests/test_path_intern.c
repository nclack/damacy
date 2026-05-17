#include "expect.h"
#include "util/path_intern.h"

#include <stdio.h>
#include <string.h>

static int
test_pointer_identity(void)
{
  struct path_intern pi = { 0 };
  const char* a = path_intern_acquire(&pi, "shard/0");
  const char* b = path_intern_acquire(&pi, "shard/0");
  EXPECT(a != NULL);
  EXPECT(a == b);
  EXPECT(strcmp(a, "shard/0") == 0);
  path_intern_release(&pi, "shard/0");
  path_intern_release(&pi, "shard/0");
  path_intern_free(&pi);
  return 0;
}

static int
test_distinct_strings(void)
{
  struct path_intern pi = { 0 };
  const char* a = path_intern_acquire(&pi, "shard/A");
  const char* b = path_intern_acquire(&pi, "shard/B");
  EXPECT(a && b);
  EXPECT(a != b);
  EXPECT(strcmp(a, "shard/A") == 0);
  EXPECT(strcmp(b, "shard/B") == 0);
  EXPECT(path_intern_acquire(&pi, "shard/A") == a);
  EXPECT(path_intern_acquire(&pi, "shard/B") == b);
  path_intern_free(&pi);
  return 0;
}

// N is sized to force several rehashes past the 16-bucket initial cap,
// exercising the "pointers stay valid across rehash" invariant.
static int
test_rehash_stability(void)
{
  enum
  {
    N = 200
  };
  struct path_intern pi = { 0 };
  const char* ptrs[N];
  char buf[32];
  for (int i = 0; i < N; ++i) {
    snprintf(buf, sizeof buf, "shard/%d", i);
    ptrs[i] = path_intern_acquire(&pi, buf);
    EXPECT(ptrs[i] != NULL);
  }
  for (int i = 0; i < N; ++i) {
    snprintf(buf, sizeof buf, "shard/%d", i);
    EXPECT(strcmp(ptrs[i], buf) == 0);
    EXPECT(path_intern_acquire(&pi, buf) == ptrs[i]);
  }
  path_intern_free(&pi);
  return 0;
}

static int
test_release_evicts_at_zero(void)
{
  struct path_intern pi = { 0 };
  const char* p1 = path_intern_acquire(&pi, "alpha");
  path_intern_acquire(&pi, "alpha");
  EXPECT(p1 != NULL);
  EXPECT(pi.n == 1);
  path_intern_release(&pi, "alpha");
  EXPECT(pi.n == 1);
  path_intern_release(&pi, "alpha");
  EXPECT(pi.n == 0);
  // Pointer identity across the eviction gap is not required.
  const char* p2 = path_intern_acquire(&pi, "alpha");
  EXPECT(p2 != NULL);
  EXPECT(strcmp(p2, "alpha") == 0);
  path_intern_release(&pi, "alpha");
  path_intern_free(&pi);
  return 0;
}

// Eviction of a colliding entry must not orphan later probes (the
// backward-shift deletion invariant).
static int
test_release_preserves_probe_chains(void)
{
  enum
  {
    N = 40
  };
  struct path_intern pi = { 0 };
  char buf[16];
  for (int i = 0; i < N; ++i) {
    snprintf(buf, sizeof buf, "k%d", i);
    EXPECT(path_intern_acquire(&pi, buf) != NULL);
  }
  for (int i = 0; i < N; i += 2) {
    snprintf(buf, sizeof buf, "k%d", i);
    path_intern_release(&pi, buf);
  }
  for (int i = 1; i < N; i += 2) {
    snprintf(buf, sizeof buf, "k%d", i);
    const char* p = path_intern_acquire(&pi, buf);
    EXPECT(p != NULL);
    EXPECT(strcmp(p, buf) == 0);
    path_intern_release(&pi, buf);
  }
  path_intern_free(&pi);
  return 0;
}

static int
test_reset_ignores_refcount(void)
{
  struct path_intern pi = { 0 };
  path_intern_acquire(&pi, "x");
  path_intern_acquire(&pi, "x");
  path_intern_acquire(&pi, "y");
  EXPECT(pi.n == 2);
  path_intern_reset(&pi);
  EXPECT(pi.n == 0);
  EXPECT(pi.cap > 0);
  const char* p = path_intern_acquire(&pi, "x");
  EXPECT(p != NULL);
  EXPECT(strcmp(p, "x") == 0);
  path_intern_free(&pi);
  return 0;
}

static int
test_null_inputs(void)
{
  struct path_intern pi = { 0 };
  EXPECT(path_intern_acquire(NULL, "x") == NULL);
  EXPECT(path_intern_acquire(&pi, NULL) == NULL);
  path_intern_release(NULL, "x");
  path_intern_release(&pi, NULL);
  path_intern_free(&pi);
  return 0;
}

static int
test_free_zero_init(void)
{
  struct path_intern pi = { 0 };
  path_intern_free(&pi);
  path_intern_free(NULL);
  path_intern_reset(NULL);
  return 0;
}

int
main(void)
{
  RUN(test_pointer_identity);
  RUN(test_distinct_strings);
  RUN(test_rehash_stability);
  RUN(test_release_evicts_at_zero);
  RUN(test_release_preserves_probe_chains);
  RUN(test_reset_ignores_refcount);
  RUN(test_null_inputs);
  RUN(test_free_zero_init);
  return 0;
}
