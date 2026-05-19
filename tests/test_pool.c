#include "expect.h"
#include "platform/platform.h"
#include "util/pool.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct thing
{
  uint64_t a;
  uint64_t b;
  uint64_t c;
};

static int
test_basic_alloc_free(void)
{
  struct pool* p = pool_create(sizeof(struct thing), 4);
  EXPECT(p);
  EXPECT(pool_capacity(p) == 4);
  EXPECT(pool_in_use(p) == 0);

  struct thing* a = (struct thing*)pool_alloc(p);
  struct thing* b = (struct thing*)pool_alloc(p);
  EXPECT(a);
  EXPECT(b);
  EXPECT(a != b);
  EXPECT(a->a == 0 && a->b == 0 && a->c == 0);
  EXPECT(pool_in_use(p) == 2);

  a->a = 0xdeadbeef;
  pool_free(p, a);
  EXPECT(pool_in_use(p) == 1);

  struct thing* a2 = (struct thing*)pool_alloc(p);
  EXPECT(a2);
  EXPECT(a2->a == 0);
  pool_free(p, a2);
  pool_free(p, b);
  EXPECT(pool_in_use(p) == 0);

  pool_destroy(p);
  return 0;
}

static int
test_exhaustion_returns_null(void)
{
  struct pool* p = pool_create(sizeof(struct thing), 3);
  EXPECT(p);

  void* slots[3] = { 0 };
  for (int i = 0; i < 3; ++i) {
    slots[i] = pool_alloc(p);
    EXPECT(slots[i]);
  }
  EXPECT(pool_alloc(p) == NULL);
  EXPECT(pool_in_use(p) == 3);

  pool_free(p, slots[1]);
  void* recycled = pool_alloc(p);
  EXPECT(recycled == slots[1]);

  EXPECT(pool_alloc(p) == NULL);
  for (int i = 0; i < 3; ++i)
    pool_free(p, slots[i]);
  pool_destroy(p);
  return 0;
}

static int
test_destroy_null_safe(void)
{
  pool_destroy(NULL);
  pool_free(NULL, NULL);
  EXPECT(pool_alloc(NULL) == NULL);
  EXPECT(pool_in_use(NULL) == 0);
  EXPECT(pool_capacity(NULL) == 0);
  return 0;
}

static int
test_create_rejects_bad_args(void)
{
  EXPECT(pool_create(0, 8) == NULL);
  EXPECT(pool_create(sizeof(void*), 0) == NULL);
  struct pool* p = pool_create(1, 4);
  EXPECT(p);
  void* a = pool_alloc(p);
  EXPECT(a);
  pool_free(p, a);
  pool_destroy(p);
  return 0;
}

#define N_CONTENTION_THREADS 8
#define N_CONTENTION_ITERS 4096

struct contention_arg
{
  struct pool* p;
  int rc;
};

static void*
contention_worker(void* vctx)
{
  struct contention_arg* a = (struct contention_arg*)vctx;
  for (int i = 0; i < N_CONTENTION_ITERS; ++i) {
    void* slot = pool_alloc(a->p);
    if (!slot)
      continue;
    memset(slot, (unsigned char)i, sizeof(struct thing));
    pool_free(a->p, slot);
  }
  a->rc = 0;
  return NULL;
}

static int
test_contention(void)
{
  struct pool* p = pool_create(sizeof(struct thing), N_CONTENTION_THREADS / 2);
  EXPECT(p);

  pthread_t threads[N_CONTENTION_THREADS];
  struct contention_arg args[N_CONTENTION_THREADS] = { 0 };
  for (int i = 0; i < N_CONTENTION_THREADS; ++i) {
    args[i].p = p;
    args[i].rc = 1;
    EXPECT(pthread_create(&threads[i], NULL, contention_worker, &args[i]) == 0);
  }
  for (int i = 0; i < N_CONTENTION_THREADS; ++i) {
    EXPECT(pthread_join(threads[i], NULL) == 0);
    EXPECT(args[i].rc == 0);
  }
  EXPECT(pool_in_use(p) == 0);
  pool_destroy(p);
  return 0;
}

int
main(void)
{
  RUN(test_basic_alloc_free);
  RUN(test_exhaustion_returns_null);
  RUN(test_destroy_null_safe);
  RUN(test_create_rejects_bad_args);
  RUN(test_contention);
  return 0;
}
