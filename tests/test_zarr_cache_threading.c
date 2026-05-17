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
  struct zarr_meta_cache* c = zarr_meta_cache_create(store, 16);
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
    if (zarr_shard_cache_get(
          w->shard_cache, w->uri, w->meta, coord, &entries, &n_entries) ==
        DAMACY_OK)
      atomic_fetch_add(&w->ok_count, 1);
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

  struct zarr_meta_cache* mc = zarr_meta_cache_create(store, 16);
  EXPECT(mc);
  struct zarr_metadata meta = { 0 };
  EXPECT(zarr_meta_cache_get(mc, "zarr_a", &meta) == DAMACY_OK);

  struct zarr_shard_cache* sc_cache = zarr_shard_cache_create(store, 16);
  EXPECT(sc_cache);

  // Prewarm the entry so workers hit the cache path.
  uint64_t coord[2] = { 0, 0 };
  const struct zarr_shard_entry* entries = NULL;
  uint64_t n_entries = 0;
  EXPECT(zarr_shard_cache_get(
           sc_cache, "zarr_a", &meta, coord, &entries, &n_entries) ==
         DAMACY_OK);

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

// Eviction-pressure: working set > cache capacity, so every miss
// evicts. Validates the copy-by-value lifetime contract on _get — a
// concurrent _get on URI B may evict the entry for URI A while thread
// T is still using its returned metadata. With copy semantics each
// thread owns its bytes, so TSan reports no race on the meta payload.
#define EVICT_CAP 2
#define EVICT_URIS 8

struct evict_worker_args
{
  struct zarr_meta_cache* meta_cache;
  const char* uris[EVICT_URIS];
  atomic_int ok_count;
};

static void*
evict_worker(void* arg)
{
  struct evict_worker_args* w = (struct evict_worker_args*)arg;
  for (int i = 0; i < N_ITERATIONS; ++i) {
    const char* uri = w->uris[i % EVICT_URIS];
    struct zarr_metadata m = { 0 };
    if (zarr_meta_cache_get(w->meta_cache, uri, &m) == DAMACY_OK) {
      // Touch the copy after the call returns; a stale pointer into an
      // evicted entry would be a UAF here.
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

  const char* names[EVICT_URIS] = { "z0", "z1", "z2", "z3",
                                    "z4", "z5", "z6", "z7" };
  for (int i = 0; i < EVICT_URIS; ++i) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", root, names[i]);
    EXPECT(mkdir(path, 0755) == 0);
    snprintf(path, sizeof path, "%s/%s/zarr.json", root, names[i]);
    EXPECT(fixture_write_file(path, MINIMAL_ZARR_JSON) == 0);
  }

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  // Capacity below the working set: misses force eviction.
  struct zarr_meta_cache* c = zarr_meta_cache_create(store, EVICT_CAP);
  EXPECT(c);

  struct evict_worker_args w = { .meta_cache = c };
  for (int i = 0; i < EVICT_URIS; ++i)
    w.uris[i] = names[i];
  atomic_init(&w.ok_count, 0);

  pthread_t threads[N_THREADS];
  for (int t = 0; t < N_THREADS; ++t)
    EXPECT(pthread_create(&threads[t], NULL, evict_worker, &w) == 0);
  for (int t = 0; t < N_THREADS; ++t)
    EXPECT(pthread_join(threads[t], NULL) == 0);

  EXPECT(atomic_load(&w.ok_count) == N_THREADS * N_ITERATIONS);

  zarr_meta_cache_destroy(c);
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
  log_info("all tests passed");
  return 0;
}
