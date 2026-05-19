// Integration test for zarr_meta_cache. Spins up a tmpdir-backed fs
// store, writes a minimal valid sharded-zstd zarr.json, and exercises
// hit/miss/eviction behaviour.

#include "fixture.h"
#include "store/store.h"
#include "util/path_intern.h"
#include "zarr/zarr_meta_cache.h"
#include "zarr/zarr_metadata.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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

// Non-sharded zarr v3: top-level [bytes, zstd] codec list, no
// sharding_indexed wrapper. Each chunk is its own file.
static const char* UNSHARDED_ZSTD_ZARR_JSON =
  "{"
  "\"zarr_format\":3,"
  "\"node_type\":\"array\","
  "\"shape\":[64,1024],"
  "\"data_type\":\"uint16\","
  "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
  "\"chunk_shape\":[32,128]}},"
  "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
  "\"separator\":\"/\"}},"
  "\"fill_value\":0,"
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
  "{\"name\":\"zstd\",\"configuration\":{\"level\":3}}]"
  "}";

// Sharded zarr v3 with index_location: "start".
static const char* SHARDED_START_ZARR_JSON =
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
  "\"index_location\":\"start\"}}]"
  "}";

static int
test_meta_cache(void)
{
  char tmpl[] = "/tmp/damacy_meta_cache_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  // Layout: <root>/foo/zarr.json and <root>/bar/zarr.json
  char path[512];
  snprintf(path, sizeof path, "%s/foo", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/zarr.json", root);
  EXPECT(fixture_write_file(path, MINIMAL_ZARR_JSON) == 0);

  snprintf(path, sizeof path, "%s/bar", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/bar/zarr.json", root);
  EXPECT(fixture_write_file(path, MINIMAL_ZARR_JSON) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct zarr_meta_cache* c = zarr_meta_cache_create(store, NULL, 4);
  EXPECT(c);

  // First get: miss → load → return.
  struct zarr_metadata m1 = { 0 };
  EXPECT(zarr_meta_cache_get(c, "foo", &m1) == DAMACY_OK);
  EXPECT(m1.rank == 2);
  EXPECT(m1.shape[0] == 64 && m1.shape[1] == 1024);
  EXPECT(m1.inner_chunk_shape[0] == 32 && m1.inner_chunk_shape[1] == 128);
  EXPECT(m1.shard_shape[0] == 64 && m1.shard_shape[1] == 512);
  EXPECT(m1.sharded == 1);

  struct zarr_meta_cache_stats st;
  zarr_meta_cache_stats_get(c, &st);
  EXPECT(st.counters.hits == 0);
  EXPECT(st.counters.misses == 1);
  EXPECT(st.size == 1);

  // Second get same uri: hit. Returned bytes must match.
  struct zarr_metadata m1b = { 0 };
  EXPECT(zarr_meta_cache_get(c, "foo", &m1b) == DAMACY_OK);
  EXPECT(memcmp(&m1b, &m1, sizeof m1) == 0);
  zarr_meta_cache_stats_get(c, &st);
  EXPECT(st.counters.hits == 1);
  EXPECT(st.counters.misses == 1);

  // Different uri: another miss; both entries cached.
  struct zarr_metadata m2 = { 0 };
  EXPECT(zarr_meta_cache_get(c, "bar", &m2) == DAMACY_OK);
  zarr_meta_cache_stats_get(c, &st);
  EXPECT(st.size == 2);

  // Missing uri: NOTFOUND.
  struct zarr_metadata mx = { 0 };
  EXPECT(zarr_meta_cache_get(c, "doesnotexist", &mx) == DAMACY_NOTFOUND);

  zarr_meta_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Non-sharded zarr v3 should parse with sharded=0 and
// inner_chunk_shape == shard_shape.
static int
test_meta_cache_unsharded(void)
{
  char tmpl[] = "/tmp/damacy_meta_unsharded_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  char path[512];
  snprintf(path, sizeof path, "%s/foo", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/zarr.json", root);
  EXPECT(fixture_write_file(path, UNSHARDED_ZSTD_ZARR_JSON) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct zarr_meta_cache* c = zarr_meta_cache_create(store, NULL, 4);
  EXPECT(c);

  struct zarr_metadata m = { 0 };
  EXPECT(zarr_meta_cache_get(c, "foo", &m) == DAMACY_OK);
  EXPECT(m.rank == 2);
  EXPECT(m.shape[0] == 64 && m.shape[1] == 1024);
  EXPECT(m.shard_shape[0] == 32 && m.shard_shape[1] == 128);
  EXPECT(m.inner_chunk_shape[0] == 32 && m.inner_chunk_shape[1] == 128);
  EXPECT(m.sharded == 0);
  EXPECT(m.inner_codec.id == CODEC_ZSTD);

  zarr_meta_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Sharded zarr v3 with index_location: "start" should parse with
// index_location_end == 0.
static int
test_meta_cache_index_start(void)
{
  char tmpl[] = "/tmp/damacy_meta_index_start_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  char path[512];
  snprintf(path, sizeof path, "%s/foo", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/zarr.json", root);
  EXPECT(fixture_write_file(path, SHARDED_START_ZARR_JSON) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct zarr_meta_cache* c = zarr_meta_cache_create(store, NULL, 4);
  EXPECT(c);

  struct zarr_metadata m = { 0 };
  EXPECT(zarr_meta_cache_get(c, "foo", &m) == DAMACY_OK);
  EXPECT(m.sharded == 1);
  EXPECT(m.index_location_end == 0);

  zarr_meta_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Drives the cache through a path_intern so the ownership assert fires
// on non-interned keys; verifies that re-interning the same string
// produces a hit (same pointer) and a different string a miss.
static int
test_meta_cache_pointer_identity(void)
{
  char tmpl[] = "/tmp/damacy_meta_pid_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  char path[512];
  snprintf(path, sizeof path, "%s/foo", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/zarr.json", root);
  EXPECT(fixture_write_file(path, MINIMAL_ZARR_JSON) == 0);
  snprintf(path, sizeof path, "%s/bar", root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/bar/zarr.json", root);
  EXPECT(fixture_write_file(path, MINIMAL_ZARR_JSON) == 0);

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);

  struct path_intern uris = { 0 };
  struct zarr_meta_cache* c = zarr_meta_cache_create(store, &uris, 4);
  EXPECT(c);

  const char* foo_a = path_intern_acquire(&uris, "foo");
  const char* foo_b = path_intern_acquire(&uris, "foo");
  const char* bar = path_intern_acquire(&uris, "bar");
  EXPECT(foo_a && foo_b && bar);
  EXPECT(foo_a == foo_b);
  EXPECT(foo_a != bar);

  struct zarr_metadata m = { 0 };
  EXPECT(zarr_meta_cache_get(c, foo_a, &m) == DAMACY_OK);
  struct zarr_meta_cache_stats st;
  zarr_meta_cache_stats_get(c, &st);
  EXPECT(st.counters.misses == 1);
  EXPECT(st.size == 1);

  EXPECT(zarr_meta_cache_get(c, foo_b, &m) == DAMACY_OK);
  zarr_meta_cache_stats_get(c, &st);
  EXPECT(st.counters.hits == 1);
  EXPECT(st.counters.misses == 1);
  EXPECT(st.size == 1);

  EXPECT(zarr_meta_cache_get(c, bar, &m) == DAMACY_OK);
  zarr_meta_cache_stats_get(c, &st);
  EXPECT(st.counters.misses == 2);
  EXPECT(st.size == 2);

  path_intern_release(&uris, foo_a);
  path_intern_release(&uris, foo_b);
  path_intern_release(&uris, bar);
  zarr_meta_cache_destroy(c);
  path_intern_free(&uris);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

// Verifies the observer slot advances monotonically through
// layout_set, ignores smaller nblocks, ignores nblocks=0 (short-circuit
// in atomic_u16_observe_max), and tolerates a NULL observer.
static int
test_meta_cache_blosc_nblocks_observer(void)
{
  char tmpl[] = "/tmp/damacy_meta_observer_XXXXXX";
  char* root = mkdtemp(tmpl);
  EXPECT(root);

  const char* names[] = { "a", "b", "c", "d" };
  for (size_t i = 0; i < sizeof names / sizeof names[0]; ++i) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", root, names[i]);
    EXPECT(mkdir(path, 0755) == 0);
    snprintf(path, sizeof path, "%s/%s/zarr.json", root, names[i]);
    EXPECT(fixture_write_file(path, MINIMAL_ZARR_JSON) == 0);
  }

  struct store_fs_config sc = { .root = root, .nthreads = 1 };
  struct store* store = store_fs_create(&sc);
  EXPECT(store);
  struct zarr_meta_cache* c = zarr_meta_cache_create(store, NULL, 8);
  EXPECT(c);

  _Atomic(uint16_t) slot;
  atomic_store_explicit(&slot, 1u, memory_order_relaxed);
  zarr_meta_cache_set_blosc_nblocks_observer(c, &slot);

  // layout_set is first-writer-wins per entry, so each step uses a
  // distinct URI.
  for (size_t i = 0; i < sizeof names / sizeof names[0]; ++i) {
    struct zarr_metadata m = { 0 };
    EXPECT(zarr_meta_cache_get(c, names[i], &m) == DAMACY_OK);
  }

  {
    struct chunk_layout cl = { .codec_id = (uint8_t)CODEC_BLOSC_ZSTD,
                               .typesize = 4,
                               .blocksize = 256,
                               .nbytes = 1024,
                               .nblocks = 4,
                               .shuffle = 1 };
    EXPECT(zarr_meta_cache_layout_set(c, "a", &cl) == 0);
    EXPECT(atomic_load_explicit(&slot, memory_order_acquire) == 4u);
  }

  {
    struct chunk_layout cl = { .codec_id = (uint8_t)CODEC_BLOSC_ZSTD,
                               .typesize = 4,
                               .blocksize = 256,
                               .nbytes = 1024,
                               .nblocks = 2,
                               .shuffle = 1 };
    EXPECT(zarr_meta_cache_layout_set(c, "b", &cl) == 0);
    EXPECT(atomic_load_explicit(&slot, memory_order_acquire) == 4u);
  }

  {
    struct chunk_layout cl = { .codec_id = (uint8_t)CODEC_BLOSC_ZSTD,
                               .typesize = 4,
                               .blocksize = 256,
                               .nbytes = 1024,
                               .nblocks = DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK,
                               .shuffle = 1 };
    EXPECT(zarr_meta_cache_layout_set(c, "c", &cl) == 0);
    EXPECT(atomic_load_explicit(&slot, memory_order_acquire) ==
           DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK);
  }

  {
    // atomic_u16_observe_max short-circuits on value==0; the helper is
    // defensive even though a real probe rejects nblocks==0 upstream.
    struct chunk_layout cl = { .codec_id = (uint8_t)CODEC_BLOSC_ZSTD,
                               .typesize = 4,
                               .blocksize = 256,
                               .nbytes = 1024,
                               .nblocks = 0,
                               .shuffle = 1 };
    EXPECT(zarr_meta_cache_layout_set(c, "d", &cl) == 0);
    EXPECT(atomic_load_explicit(&slot, memory_order_acquire) ==
           DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK);
  }

  zarr_meta_cache_set_blosc_nblocks_observer(c, NULL);

  zarr_meta_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_meta_cache);
  RUN(test_meta_cache_unsharded);
  RUN(test_meta_cache_index_start);
  RUN(test_meta_cache_pointer_identity);
  RUN(test_meta_cache_blosc_nblocks_observer);
  log_info("all tests passed");
  return 0;
}
