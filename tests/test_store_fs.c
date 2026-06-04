// Bound + pinning assertions for the store_fs LRU fd cache.

#include "expect.h"
#include "store/store.h"
#include "store/store_fs.h"
#include "util/lru.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CAP 4u
#define N_KEYS 12u

static int
write_byte(const char* path, char c)
{
  FILE* f = fopen(path, "wb");
  if (!f)
    return 1;
  size_t n = fwrite(&c, 1, 1, f);
  fclose(f);
  return n == 1 ? 0 : 1;
}

static int
setup_files(const char* root, size_t n)
{
  for (size_t i = 0; i < n; ++i) {
    char p[256];
    snprintf(p, sizeof p, "%s/k%zu", root, i);
    if (write_byte(p, (char)('a' + (i % 26))))
      return 1;
  }
  return 0;
}

static int
test_bounded_eviction(void)
{
  char root[] = "/tmp/damacy_store_fs_bound_XXXXXX";
  EXPECT(mkdtemp(root));
  EXPECT(setup_files(root, N_KEYS) == 0);

  struct store_fs_config sc = {
    .root = root,
    .nthreads = 1,
    .fd_cache_capacity = CAP,
  };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct store_fs* fs = (struct store_fs*)store;

  for (size_t i = 0; i < N_KEYS; ++i) {
    char key[32];
    snprintf(key, sizeof key, "k%zu", i);
    struct lru_entry* pin = NULL;
    EXPECT(store_fs_acquire(fs, key, &pin) != NULL);
    store_fs_release(fs, pin);
  }

  struct lru_stats stats;
  store_fs_stats_get(fs, &stats);
  EXPECT(stats.capacity == CAP);
  EXPECT(stats.size == CAP);
  EXPECT(stats.counters.insertions == N_KEYS);
  EXPECT(stats.counters.evictions == N_KEYS - CAP);
  EXPECT(stats.counters.put_failures == 0);
  EXPECT(stats.pinned == 0);

  store_destroy(store);
  return 0;
}

static int
test_hit_path(void)
{
  char root[] = "/tmp/damacy_store_fs_hit_XXXXXX";
  EXPECT(mkdtemp(root));
  EXPECT(setup_files(root, 1) == 0);

  struct store_fs_config sc = {
    .root = root,
    .nthreads = 1,
    .fd_cache_capacity = CAP,
  };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct store_fs* fs = (struct store_fs*)store;

  enum
  {
    REPEATS = 5
  };
  for (size_t i = 0; i < REPEATS; ++i) {
    struct lru_entry* pin = NULL;
    EXPECT(store_fs_acquire(fs, "k0", &pin) != NULL);
    store_fs_release(fs, pin);
  }

  struct lru_stats stats;
  store_fs_stats_get(fs, &stats);
  EXPECT(stats.counters.insertions == 1);
  EXPECT(stats.counters.hits == REPEATS - 1);
  EXPECT(stats.counters.misses == 1);

  store_destroy(store);
  return 0;
}

static int
test_pin_saturation(void)
{
  char root[] = "/tmp/damacy_store_fs_pin_XXXXXX";
  EXPECT(mkdtemp(root));
  EXPECT(setup_files(root, CAP + 1) == 0);

  struct store_fs_config sc = {
    .root = root,
    .nthreads = 1,
    .fd_cache_capacity = CAP,
  };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct store_fs* fs = (struct store_fs*)store;

  struct lru_entry* pins[CAP] = { 0 };
  for (size_t i = 0; i < CAP; ++i) {
    char key[32];
    snprintf(key, sizeof key, "k%zu", i);
    EXPECT(store_fs_acquire(fs, key, &pins[i]) != NULL);
  }

  {
    struct lru_entry* extra = NULL;
    EXPECT(store_fs_acquire(fs, "k4", &extra) == NULL);
    EXPECT(extra == NULL);
  }

  struct lru_stats stats;
  store_fs_stats_get(fs, &stats);
  EXPECT(stats.pinned == CAP);
  EXPECT(stats.counters.put_failures == 1);

  store_fs_release(fs, pins[0]);
  {
    struct lru_entry* fresh = NULL;
    EXPECT(store_fs_acquire(fs, "k4", &fresh) != NULL);
    store_fs_release(fs, fresh);
  }

  for (size_t i = 1; i < CAP; ++i)
    store_fs_release(fs, pins[i]);

  store_destroy(store);
  return 0;
}

// CAP > THREADS so acquire never blocks on pin saturation.
#define CONTEND_KEYS 32u
#define CONTEND_THREADS 4
#define CONTEND_CAP (CONTEND_THREADS + 1u)
#define CONTEND_ITERS 4000

struct contend_args
{
  struct store_fs* fs;
  atomic_int ok_count;
};

static void*
contend_worker(void* arg)
{
  struct contend_args* w = (struct contend_args*)arg;
  for (int i = 0; i < CONTEND_ITERS; ++i) {
    char key[32];
    snprintf(key, sizeof key, "k%u", (unsigned)(i % CONTEND_KEYS));
    struct lru_entry* pin = NULL;
    if (!store_fs_acquire(w->fs, key, &pin) || !pin)
      return NULL;
    atomic_fetch_add(&w->ok_count, 1);
    store_fs_release(w->fs, pin);
  }
  return NULL;
}

static int
test_lock_free_release_under_contention(void)
{
  char root[] = "/tmp/damacy_store_fs_contend_XXXXXX";
  EXPECT(mkdtemp(root));
  EXPECT(setup_files(root, CONTEND_KEYS) == 0);

  struct store_fs_config sc = {
    .root = root,
    .nthreads = 1,
    .fd_cache_capacity = CONTEND_CAP,
  };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct store_fs* fs = (struct store_fs*)store;

  struct contend_args w = { .fs = fs };
  atomic_init(&w.ok_count, 0);

  pthread_t threads[CONTEND_THREADS];
  for (int t = 0; t < CONTEND_THREADS; ++t)
    EXPECT(pthread_create(&threads[t], NULL, contend_worker, &w) == 0);
  for (int t = 0; t < CONTEND_THREADS; ++t)
    EXPECT(pthread_join(threads[t], NULL) == 0);

  EXPECT(atomic_load(&w.ok_count) == CONTEND_THREADS * CONTEND_ITERS);

  struct lru_stats stats;
  store_fs_stats_get(fs, &stats);
  EXPECT(stats.counters.evictions > 0);
  EXPECT(stats.pinned == 0);

  store_destroy(store);
  return 0;
}

static int
test_read_job_stats(void)
{
  char root[] = "/tmp/damacy_store_fs_read_stats_XXXXXX";
  EXPECT(mkdtemp(root));
  EXPECT(setup_files(root, 1) == 0);

  struct store_fs_config sc = {
    .root = root,
    .nthreads = 1,
    .fd_cache_capacity = CAP,
  };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct store_fs* fs = (struct store_fs*)store;

  for (int i = 0; i < 3; ++i) {
    char dst = 0;
    struct store_read read = {
      .key = "k0",
      .dst = &dst,
      .offset = 0,
      .len = sizeof dst,
    };
    EXPECT(store_read_many(store, &read, 1) == 0);
    EXPECT(dst == 'a');
  }

  struct store_fs_io_stats stats;
  store_fs_io_stats_get(fs, &stats);
  EXPECT(stats.read_jobs == 3);
  EXPECT(stats.read_active == 0);
  EXPECT(stats.read_max_active >= 1);

  store_fs_io_stats_reset(fs);
  store_fs_io_stats_get(fs, &stats);
  EXPECT(stats.read_jobs == 0);
  EXPECT(stats.read_active == 0);
  EXPECT(stats.read_max_active == 0);

  store_destroy(store);
  return 0;
}

static int
test_async_read_error_reports_io(void)
{
  char root[] = "/tmp/damacy_store_fs_error_XXXXXX";
  EXPECT(mkdtemp(root));
  char bad_dir[256];
  snprintf(bad_dir, sizeof bad_dir, "%s/bad_dir", root);
  EXPECT(mkdir(bad_dir, 0700) == 0);

  struct store_fs_config sc = {
    .root = root,
    .nthreads = 1,
    .fd_cache_capacity = CAP,
  };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  char dst[4] = { 0 };
  struct store_read read = {
    .key = "bad_dir",
    .dst = dst,
    .offset = 0,
    .len = sizeof dst,
  };
  struct store_submit_result submit = store_read_submit(store, &read, 1);
  EXPECT(submit.status == DAMACY_OK);

  struct store_event_poll poll = { 0 };
  for (int i = 0; i < 1000 && !poll.ready; ++i) {
    poll = store_event_query(store, submit.event);
    if (!poll.ready)
      usleep(1000);
  }
  EXPECT(poll.ready);
  EXPECT(poll.status == DAMACY_IO);

  store_destroy(store);
  return 0;
}

int
main(void)
{
  RUN(test_bounded_eviction);
  RUN(test_hit_path);
  RUN(test_pin_saturation);
  RUN(test_lock_free_release_under_contention);
  RUN(test_read_job_stats);
  RUN(test_async_read_error_reports_io);
  return 0;
}
