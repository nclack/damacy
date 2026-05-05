// Integration test for zarr_meta_cache. Spins up a tmpdir-backed fs
// store, writes a minimal valid sharded-zstd zarr.json, and exercises
// hit/miss/eviction behaviour.

#include "fixture.h"
#include "store/store.h"
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

  struct zarr_meta_cache* c = zarr_meta_cache_create(store, 4);
  EXPECT(c);

  // First get: miss → load → return.
  const struct zarr_metadata* m1 = NULL;
  EXPECT(zarr_meta_cache_get(c, "foo", &m1) == DAMACY_OK);
  EXPECT(m1);
  EXPECT(m1->rank == 2);
  EXPECT(m1->shape[0] == 64 && m1->shape[1] == 1024);
  EXPECT(m1->inner_chunk_shape[0] == 32 && m1->inner_chunk_shape[1] == 128);
  EXPECT(m1->shard_shape[0] == 64 && m1->shard_shape[1] == 512);
  EXPECT(m1->sharded == 1);

  struct zarr_meta_cache_stats st;
  zarr_meta_cache_stats_get(c, &st);
  EXPECT(st.counters.hits == 0);
  EXPECT(st.counters.misses == 1);
  EXPECT(st.size == 1);

  // Second get same uri: hit. Returned pointer must be the same.
  const struct zarr_metadata* m1b = NULL;
  EXPECT(zarr_meta_cache_get(c, "foo", &m1b) == DAMACY_OK);
  EXPECT(m1b == m1);
  zarr_meta_cache_stats_get(c, &st);
  EXPECT(st.counters.hits == 1);
  EXPECT(st.counters.misses == 1);

  // Different uri: another miss; both entries cached.
  const struct zarr_metadata* m2 = NULL;
  EXPECT(zarr_meta_cache_get(c, "bar", &m2) == DAMACY_OK);
  EXPECT(m2 && m2 != m1);
  zarr_meta_cache_stats_get(c, &st);
  EXPECT(st.size == 2);

  // Missing uri: NOTFOUND.
  const struct zarr_metadata* mx = NULL;
  EXPECT(zarr_meta_cache_get(c, "doesnotexist", &mx) == DAMACY_NOTFOUND);
  EXPECT(mx == NULL);

  zarr_meta_cache_destroy(c);
  store_destroy(store);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_meta_cache);
  log_info("all tests passed");
  return 0;
}
