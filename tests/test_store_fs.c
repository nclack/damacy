// Bound + pinning assertions for the store_fs LRU fd cache.

#include "expect.h"
#include "store/store.h"
#include "store/store_fs.h"
#include "util/hash.h"
#include "util/lru.h"

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
    EXPECT(store_fs_acquire(fs, key, hash_fnv1a_str(key), &pin) != NULL);
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
    EXPECT(store_fs_acquire(fs, "k0", hash_fnv1a_str("k0"), &pin) != NULL);
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
    EXPECT(store_fs_acquire(fs, key, hash_fnv1a_str(key), &pins[i]) != NULL);
  }

  {
    struct lru_entry* extra = NULL;
    EXPECT(store_fs_acquire(fs, "k4", hash_fnv1a_str("k4"), &extra) == NULL);
    EXPECT(extra == NULL);
  }

  struct lru_stats stats;
  store_fs_stats_get(fs, &stats);
  EXPECT(stats.pinned == CAP);
  EXPECT(stats.counters.put_failures == 1);

  store_fs_release(fs, pins[0]);
  {
    struct lru_entry* fresh = NULL;
    EXPECT(store_fs_acquire(fs, "k4", hash_fnv1a_str("k4"), &fresh) != NULL);
    store_fs_release(fs, fresh);
  }

  for (size_t i = 1; i < CAP; ++i)
    store_fs_release(fs, pins[i]);

  store_destroy(store);
  return 0;
}

int
main(void)
{
  RUN(test_bounded_eviction);
  RUN(test_hit_path);
  RUN(test_pin_saturation);
  return 0;
}
