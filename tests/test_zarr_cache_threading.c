// Thread-safety test for zarr_meta_cache and zarr_shard_cache.
// Spawns N workers that hammer the same set of URIs concurrently; under
// ThreadSanitizer this surfaces any unsynchronized writes inside
// lru_get / lru_put / lru_stats_get. The bug we caught regressed when
// damacy.c's plan_run released scheduler_lock and exposed the cache to
// concurrent access for the first time.
//
// The test is correctness-by-construction: if it completes without a
// TSan report and the asserted hit/miss counts match, it passes. The
// only failure mode without TSan is a crash from corrupted LRU links.
//
// Run under tsan via:
//   cmake -B build-tsan -DDAMACY_TSAN=ON && cmake --build build-tsan
//   ctest --test-dir build-tsan -R test_zarr_cache_threading

#include "fixture.h"
#include "store/store.h"
#include "zarr/zarr_meta_cache.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_cache.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define N_URIS 4
#define N_THREADS 4
#define N_ITERATIONS 2000

static const char* MINIMAL_ZARR_JSON =
  "{"
  "\"zarr_format\":3,"
  "\"node_type\":\"array\","
  "\"shape\":[64,1024],"
  "\"data_type\":\"uint16\","
  "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
  "\"chunk_shape\":[64,512]}},"
  "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
  "\"separator\":\"/\"}},"
  "\"fill_value\":0,"
  "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
  "\"chunk_shape\":[32,128],"
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
  "{\"name\":\"zstd\",\"configuration\":{\"level\":3,\"checksum\":false}}],"
  "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
  "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
  "\"index_location\":\"end\"}}]"
  "}";

struct worker_args
{
  struct zarr_meta_cache* meta_cache;
  const char* uris[N_URIS];
  atomic_int ok_count;
};

static void*
meta_worker(void* arg)
{
  struct worker_args* w = (struct worker_args*)arg;
  for (int i = 0; i < N_ITERATIONS; ++i) {
    const char* uri = w->uris[i % N_URIS];
    struct zarr_metadata m = { 0 };
    if (zarr_meta_cache_get(w->meta_cache, uri, &m) == DAMACY_OK) {
      // Also exercise the layout side: it goes through the same LRU
      // promotion path on the hit branch and pokes the entry's mutex.
      struct chunk_layout cl = { 0 };
      (void)zarr_meta_cache_layout_get(w->meta_cache, uri, &cl);
      atomic_fetch_add(&w->ok_count, 1);
    }
  }
  return NULL;
}

// stats_get races against the worker threads — this is what
// damacy_stats_get was doing in the original hang and what surfaced
// the LRU corruption when iterating list_head from the consumer.
static void*
stats_reader(void* arg)
{
  struct worker_args* w = (struct worker_args*)arg;
  for (int i = 0; i < N_ITERATIONS; ++i) {
    struct zarr_meta_cache_stats st;
    zarr_meta_cache_stats_get(w->meta_cache, &st);
  }
  return NULL;
}

static int
test_meta_cache_concurrent(void)
{
  char tmpl[] = "/tmp/damacy_cache_tsan_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  const char* names[N_URIS] = { "zarr_a", "zarr_b", "zarr_c", "zarr_d" };
  for (int i = 0; i < N_URIS; ++i) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", root, names[i]);
    EXPECT(mkdir(path, 0755) == 0);
    snprintf(path, sizeof path, "%s/%s/zarr.json", root, names[i]);
    EXPECT(fixture_write_file(path, MINIMAL_ZARR_JSON) == 0);
  }

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  // Cache cap > N_URIS so we exercise the hit path (which mutates LRU
  // order via list_promote) most of the time. A short prewarm seeds the
  // entries on the main thread.
  struct zarr_meta_cache* c = zarr_meta_cache_create(store, NULL, 16);
  EXPECT(c);
  for (int i = 0; i < N_URIS; ++i) {
    struct zarr_metadata m = { 0 };
    EXPECT(zarr_meta_cache_get(c, names[i], &m) == DAMACY_OK);
  }

  struct worker_args w = { .meta_cache = c };
  for (int i = 0; i < N_URIS; ++i)
    w.uris[i] = names[i];
  atomic_init(&w.ok_count, 0);

  pthread_t threads[N_THREADS + 1];
  for (int t = 0; t < N_THREADS; ++t)
    EXPECT(pthread_create(&threads[t], NULL, meta_worker, &w) == 0);
  EXPECT(pthread_create(&threads[N_THREADS], NULL, stats_reader, &w) == 0);
  for (int t = 0; t < N_THREADS + 1; ++t)
    EXPECT(pthread_join(threads[t], NULL) == 0);

  EXPECT(atomic_load(&w.ok_count) == N_THREADS * N_ITERATIONS);

  zarr_meta_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Shard cache had the same bug pattern: lru_get unguarded. This exercises
// shard_cache_get on a tiny synthetic shard so the get path hits the LRU
// promotion. Real shard probe needs valid shard index bytes; we use the
// fixture helper that synthesizes a minimal valid shard index.
struct shard_worker_args
{
  struct zarr_shard_cache* shard_cache;
  const struct zarr_metadata* meta;
  const char* uri;
  atomic_int ok_count;
};

static void*
shard_worker(void* arg)
{
  struct shard_worker_args* w = (struct shard_worker_args*)arg;
  uint64_t coord[2] = { 0, 0 };
  for (int i = 0; i < N_ITERATIONS; ++i) {
    const struct zarr_shard_entry* entries = NULL;
    uint64_t n_entries = 0;
    struct zarr_shard_pin pin = { 0 };
    if (zarr_shard_cache_get(
          w->shard_cache, w->uri, w->meta, coord, &pin, &entries, &n_entries) ==
        DAMACY_OK) {
      // Touch the pinned entries — the pin must protect against
      // concurrent eviction here. Result is discarded; the read just
      // needs to happen so TSan sees the access.
      volatile uint64_t sink = (n_entries && entries) ? entries[0].offset : 0;
      (void)sink;
      atomic_fetch_add(&w->ok_count, 1);
      zarr_shard_cache_release(w->shard_cache, pin);
    }
  }
  return NULL;
}

static void*
shard_stats_reader(void* arg)
{
  struct shard_worker_args* w = (struct shard_worker_args*)arg;
  for (int i = 0; i < N_ITERATIONS; ++i) {
    struct zarr_shard_cache_stats st;
    zarr_shard_cache_stats_get(w->shard_cache, &st);
  }
  return NULL;
}

static int
test_shard_cache_concurrent(void)
{
  char tmpl[] = "/tmp/damacy_shard_tsan_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  // Single sharded zarr with a minimal valid shard at c/0/0 (the
  // first-shard coord for the chunk_grid in MINIMAL_ZARR_JSON).
  char path[512];
  snprintf(path, sizeof path, "%s/zarr_a", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/zarr_a/zarr.json", root);
  EXPECT(fixture_write_file(path, MINIMAL_ZARR_JSON) == 0);
  snprintf(path, sizeof path, "%s/zarr_a/c", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/zarr_a/c/0", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/zarr_a/c/0/0", root);
  // 4 inner chunks per shard for MINIMAL_ZARR_JSON (shard 64x512, inner
  // 32x128 → 2×4 = 8). Actually 64/32 * 512/128 = 2 * 4 = 8.
  uint64_t offsets[8] = { 0, 16, 32, 48, 64, 80, 96, 112 };
  uint64_t nbytes[8] = { 16, 16, 16, 16, 16, 16, 16, 16 };
  EXPECT(fixture_write_synthetic_shard(path, 128, offsets, nbytes, 8) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct zarr_meta_cache* mc = zarr_meta_cache_create(store, NULL, 16);
  EXPECT(mc);
  struct zarr_metadata meta = { 0 };
  EXPECT(zarr_meta_cache_get(mc, "zarr_a", &meta) == DAMACY_OK);

  struct zarr_shard_cache* sc_cache = zarr_shard_cache_create(store, NULL, 16);
  EXPECT(sc_cache);

  // Prewarm the entry so workers hit the cache path.
  uint64_t coord[2] = { 0, 0 };
  const struct zarr_shard_entry* entries = NULL;
  uint64_t n_entries = 0;
  struct zarr_shard_pin pin = { 0 };
  EXPECT(zarr_shard_cache_get(
           sc_cache, "zarr_a", &meta, coord, &pin, &entries, &n_entries) ==
         DAMACY_OK);
  zarr_shard_cache_release(sc_cache, pin);

  struct shard_worker_args w = { .shard_cache = sc_cache,
                                 .meta = &meta,
                                 .uri = "zarr_a" };
  atomic_init(&w.ok_count, 0);

  pthread_t threads[N_THREADS + 1];
  for (int t = 0; t < N_THREADS; ++t)
    EXPECT(pthread_create(&threads[t], NULL, shard_worker, &w) == 0);
  EXPECT(pthread_create(&threads[N_THREADS], NULL, shard_stats_reader, &w) ==
         0);
  for (int t = 0; t < N_THREADS + 1; ++t)
    EXPECT(pthread_join(threads[t], NULL) == 0);

  EXPECT(atomic_load(&w.ok_count) == N_THREADS * N_ITERATIONS);

  zarr_shard_cache_destroy(sc_cache);
  zarr_meta_cache_destroy(mc);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Meta cache: working set > cache capacity, every miss evicts. Verifies
// the copy-by-value contract — a stale pointer to an evicted entry
// would be a UAF here.
#define META_EVICT_CAP 2
#define META_EVICT_URIS 8

struct meta_evict_args
{
  struct zarr_meta_cache* meta_cache;
  const char* uris[META_EVICT_URIS];
  atomic_int ok_count;
};

static void*
meta_evict_worker(void* arg)
{
  struct meta_evict_args* w = (struct meta_evict_args*)arg;
  for (int i = 0; i < N_ITERATIONS; ++i) {
    const char* uri = w->uris[i % META_EVICT_URIS];
    struct zarr_metadata m = { 0 };
    if (zarr_meta_cache_get(w->meta_cache, uri, &m) == DAMACY_OK) {
      if (m.rank == 2 && m.shape[0] == 64)
        atomic_fetch_add(&w->ok_count, 1);
    }
  }
  return NULL;
}

static int
test_meta_cache_eviction_concurrent(void)
{
  char tmpl[] = "/tmp/damacy_cache_evict_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  const char* names[META_EVICT_URIS] = { "z0", "z1", "z2", "z3",
                                         "z4", "z5", "z6", "z7" };
  for (int i = 0; i < META_EVICT_URIS; ++i) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", root, names[i]);
    EXPECT(mkdir(path, 0755) == 0);
    snprintf(path, sizeof path, "%s/%s/zarr.json", root, names[i]);
    EXPECT(fixture_write_file(path, MINIMAL_ZARR_JSON) == 0);
  }

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct zarr_meta_cache* c =
    zarr_meta_cache_create(store, NULL, META_EVICT_CAP);
  EXPECT(c);

  struct meta_evict_args w = { .meta_cache = c };
  for (int i = 0; i < META_EVICT_URIS; ++i)
    w.uris[i] = names[i];
  atomic_init(&w.ok_count, 0);

  pthread_t threads[N_THREADS];
  for (int t = 0; t < N_THREADS; ++t)
    EXPECT(pthread_create(&threads[t], NULL, meta_evict_worker, &w) == 0);
  for (int t = 0; t < N_THREADS; ++t)
    EXPECT(pthread_join(threads[t], NULL) == 0);

  EXPECT(atomic_load(&w.ok_count) == N_THREADS * N_ITERATIONS);

  zarr_meta_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Shard cache: same shape, but the entries are a borrow whose lifetime
// the pin protects. cap > N_THREADS so an unpinned slot is always
// available to evict; cap < working set so eviction fires on every cold
// rotation. Without pin/release a worker would race a concurrent
// eviction reusing the entries allocation.
static const char* SHARD_EVICT_ZARR_JSON =
  "{"
  "\"zarr_format\":3,"
  "\"node_type\":\"array\","
  "\"shape\":[64,4096],"
  "\"data_type\":\"uint16\","
  "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
  "\"chunk_shape\":[64,512]}},"
  "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
  "\"separator\":\"/\"}},"
  "\"fill_value\":0,"
  "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
  "\"chunk_shape\":[32,128],"
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
  "{\"name\":\"zstd\",\"configuration\":{\"level\":3,\"checksum\":false}}],"
  "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
  "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
  "\"index_location\":\"end\"}}]"
  "}";

#define SHARD_EVICT_N 8
#define SHARD_EVICT_CAP (N_THREADS + 1)

struct shard_evict_args
{
  struct zarr_shard_cache* shard_cache;
  const struct zarr_metadata* meta;
  const char* uri;
  atomic_int ok_count;
};

static void*
shard_evict_worker(void* arg)
{
  struct shard_evict_args* w = (struct shard_evict_args*)arg;
  for (int i = 0; i < N_ITERATIONS; ++i) {
    uint64_t coord[2] = { 0, (uint64_t)(i % SHARD_EVICT_N) };
    const struct zarr_shard_entry* entries = NULL;
    uint64_t n_entries = 0;
    struct zarr_shard_pin pin = { 0 };
    if (zarr_shard_cache_get(
          w->shard_cache, w->uri, w->meta, coord, &pin, &entries, &n_entries) ==
        DAMACY_OK) {
      volatile uint64_t sink = 0;
      for (uint64_t j = 0; j < n_entries; ++j)
        sink += entries[j].offset + entries[j].nbytes;
      (void)sink;
      atomic_fetch_add(&w->ok_count, 1);
      zarr_shard_cache_release(w->shard_cache, pin);
    }
  }
  return NULL;
}

static int
test_shard_cache_pin_under_eviction(void)
{
  char tmpl[] = "/tmp/damacy_shard_evict_tsan_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  char path[512];
  snprintf(path, sizeof path, "%s/zarr_a", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/zarr_a/zarr.json", root);
  EXPECT(fixture_write_file(path, SHARD_EVICT_ZARR_JSON) == 0);
  snprintf(path, sizeof path, "%s/zarr_a/c", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/zarr_a/c/0", root);
  EXPECT(mkdir(path, 0755) == 0);
  uint64_t offsets[8] = { 0, 16, 32, 48, 64, 80, 96, 112 };
  uint64_t nbytes[8] = { 16, 16, 16, 16, 16, 16, 16, 16 };
  for (int k = 0; k < SHARD_EVICT_N; ++k) {
    snprintf(path, sizeof path, "%s/zarr_a/c/0/%d", root, k);
    EXPECT(fixture_write_synthetic_shard(path, 128, offsets, nbytes, 8) == 0);
  }

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct zarr_meta_cache* mc = zarr_meta_cache_create(store, NULL, 16);
  EXPECT(mc);
  struct zarr_metadata meta = { 0 };
  EXPECT(zarr_meta_cache_get(mc, "zarr_a", &meta) == DAMACY_OK);

  struct zarr_shard_cache* sc_cache =
    zarr_shard_cache_create(store, NULL, SHARD_EVICT_CAP);
  EXPECT(sc_cache);

  struct shard_evict_args w = { .shard_cache = sc_cache,
                                .meta = &meta,
                                .uri = "zarr_a" };
  atomic_init(&w.ok_count, 0);

  pthread_t threads[N_THREADS];
  for (int t = 0; t < N_THREADS; ++t)
    EXPECT(pthread_create(&threads[t], NULL, shard_evict_worker, &w) == 0);
  for (int t = 0; t < N_THREADS; ++t)
    EXPECT(pthread_join(threads[t], NULL) == 0);

  EXPECT(atomic_load(&w.ok_count) == N_THREADS * N_ITERATIONS);

  struct zarr_shard_cache_stats st;
  zarr_shard_cache_stats_get(sc_cache, &st);
  EXPECT(st.counters.evictions > 0);

  zarr_shard_cache_destroy(sc_cache);
  zarr_meta_cache_destroy(mc);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_meta_cache_concurrent);
  RUN(test_meta_cache_eviction_concurrent);
  RUN(test_shard_cache_concurrent);
  RUN(test_shard_cache_pin_under_eviction);
  log_info("all tests passed");
  return 0;
}
