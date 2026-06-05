#include "damacy.h"
#include "fixture.h"
#include "prefetch/array_meta.h"
#include "prefetch/prefetch_cache.h"
#include "prefetch/shard_index.h"
#include "store/store.h"
#include "store/store_fs.h"
#include "util/hash.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_index.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// shape=[64,1024], inner=[32,128], shard=[64,512] => n_inner_per_shard = 8.
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
sync_post(struct prefetch_executor* self,
          void (*fn)(void*),
          void* ctx,
          void (*ctx_free)(void*))
{
  (void)self;
  fn(ctx);
  if (ctx_free)
    ctx_free(ctx);
  return 0;
}

static struct prefetch_executor SYNC_EXECUTOR = { .post = sync_post };

struct fixture
{
  char* root;
  struct store* store;
  struct array_meta_fetcher array_meta_fetcher;
  struct prefetch_cache* array_meta_cache;
  struct shard_index_fetcher shard_index_fetcher;
  struct prefetch_cache* shard_index_cache;
};

static int
write_array(const char* root, const char* uri, const char* contents)
{
  char path[512];
  snprintf(path, sizeof path, "%s/%s", root, uri);
  if (mkdir(path, 0755) != 0)
    return 1;
  snprintf(path, sizeof path, "%s/%s/zarr.json", root, uri);
  return fixture_write_file(path, contents);
}

static int
write_shard(const char* root, const char* uri, uint64_t a, uint64_t b)
{
  char path[512];
  snprintf(path, sizeof path, "%s/%s/c", root, uri);
  mkdir(path, 0755);
  snprintf(path, sizeof path, "%s/%s/c/%llu", root, uri, (unsigned long long)a);
  mkdir(path, 0755);
  snprintf(path,
           sizeof path,
           "%s/%s/c/%llu/%llu",
           root,
           uri,
           (unsigned long long)a,
           (unsigned long long)b);
  const uint64_t offsets[8] = { 0, 100, 200, 300, 400, 500, 600, 700 };
  const uint64_t nbytes[8] = { 50, 50, 50, 50, 50, 50, 50, 50 };
  return fixture_write_synthetic_shard(path, 4096, offsets, nbytes, 8);
}

static int
fixture_setup(struct fixture* fx)
{
  char tmpl[] = "/tmp/damacy_shard_index_XXXXXX";
  EXPECT(mkdtemp(tmpl));
  fx->root = strdup(tmpl);
  EXPECT(fx->root);

  struct store_fs_config sc = { .root = fx->root, .nthreads = 1 };
  fx->store = store_fs_create(&sc);
  EXPECT(fx->store);

  array_meta_fetcher_init(&fx->array_meta_fetcher, fx->store);
  struct prefetch_cache_config amc_cfg = {
    .capacity = 4,
    .max_probe = 16,
    .ops = &array_meta_ops,
    .fetcher = &fx->array_meta_fetcher.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->array_meta_cache = prefetch_cache_create(&amc_cfg);
  EXPECT(fx->array_meta_cache);

  shard_index_fetcher_init(
    &fx->shard_index_fetcher, fx->store, fx->array_meta_cache);
  struct prefetch_cache_config sic_cfg = {
    .capacity = 8,
    .max_probe = 16,
    .ops = &shard_index_ops,
    .fetcher = &fx->shard_index_fetcher.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->shard_index_cache = prefetch_cache_create(&sic_cfg);
  EXPECT(fx->shard_index_cache);
  return 0;
}

static void
fixture_teardown(struct fixture* fx)
{
  if (fx->shard_index_cache)
    prefetch_cache_destroy(fx->shard_index_cache);
  if (fx->array_meta_cache)
    prefetch_cache_destroy(fx->array_meta_cache);
  if (fx->store)
    store_destroy(fx->store);
  if (fx->root) {
    fixture_rm_tree(fx->root);
    free(fx->root);
  }
}

static int
warm_meta(struct fixture* fx, const char* uri)
{
  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h = prefetch_cache_request(
    fx->array_meta_cache, hash_fnv1a_str(uri), uri, 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));
  return 0;
}

static int
test_shard_index_parses_footer(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx) == 0);
  EXPECT(write_array(fx.root, "foo", MINIMAL_ZARR_JSON) == 0);
  EXPECT(write_shard(fx.root, "foo", 0, 0) == 0);

  EXPECT(warm_meta(&fx, "foo") == 0);

  struct shard_index_key probe = {
    .uri = "foo",
    .shard_coord = { 0, 0 },
    .rank = 2,
  };
  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h = prefetch_cache_request(
    fx.shard_index_cache, shard_index_key_hash(&probe), &probe, 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  const struct shard_index_value* v =
    (const struct shard_index_value*)prefetch_cache_try_get(
      fx.shard_index_cache, h);
  EXPECT(v);
  EXPECT(v->n_entries == 8);
  EXPECT(v->entries[0].offset == 0 && v->entries[0].nbytes == 50);
  EXPECT(v->entries[7].offset == 700 && v->entries[7].nbytes == 50);

  fixture_teardown(&fx);
  return 0;
}

static int
test_missing_shard_returns_notfound(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx) == 0);
  EXPECT(write_array(fx.root, "foo", MINIMAL_ZARR_JSON) == 0);
  EXPECT(warm_meta(&fx, "foo") == 0);

  struct shard_index_key probe = {
    .uri = "foo",
    .shard_coord = { 9, 9 },
    .rank = 2,
  };
  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h = prefetch_cache_request(
    fx.shard_index_cache, shard_index_key_hash(&probe), &probe, 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_has_error(&gate));

  int err = 0;
  EXPECT(prefetch_cache_query(fx.shard_index_cache, h, NULL, &err) ==
         PREFETCH_STATE_ERROR);
  EXPECT(err == DAMACY_NOTFOUND);

  fixture_teardown(&fx);
  return 0;
}

// Guards against aliasing EACCES to NOTFOUND (the prefetcher tolerates the
// latter). Root bypasses chmod 0, so skip.
static int
test_unreadable_shard_returns_io(void)
{
  if (geteuid() == 0) {
    log_info("skip test_unreadable_shard_returns_io: running as root");
    return 0;
  }

  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx) == 0);
  EXPECT(write_array(fx.root, "foo", MINIMAL_ZARR_JSON) == 0);
  EXPECT(write_shard(fx.root, "foo", 0, 0) == 0);
  EXPECT(warm_meta(&fx, "foo") == 0);

  char dir[512];
  snprintf(dir, sizeof dir, "%s/foo/c/0", fx.root);
  EXPECT(chmod(dir, 0) == 0);

  struct shard_index_key probe = {
    .uri = "foo",
    .shard_coord = { 0, 0 },
    .rank = 2,
  };
  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h = prefetch_cache_request(
    fx.shard_index_cache, shard_index_key_hash(&probe), &probe, 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_has_error(&gate));

  int err = 0;
  EXPECT(prefetch_cache_query(fx.shard_index_cache, h, NULL, &err) ==
         PREFETCH_STATE_ERROR);
  EXPECT(err == DAMACY_IO);

  // Restore mode so fixture_rm_tree can recurse.
  chmod(dir, 0755);
  fixture_teardown(&fx);
  return 0;
}

static int
test_different_coords_are_separate_entries(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx) == 0);
  EXPECT(write_array(fx.root, "foo", MINIMAL_ZARR_JSON) == 0);
  EXPECT(write_shard(fx.root, "foo", 0, 0) == 0);
  EXPECT(write_shard(fx.root, "foo", 0, 1) == 0);
  EXPECT(warm_meta(&fx, "foo") == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct shard_index_key probe00 = { .uri = "foo",
                                     .shard_coord = { 0, 0 },
                                     .rank = 2 };
  struct shard_index_key probe01 = { .uri = "foo",
                                     .shard_coord = { 0, 1 },
                                     .rank = 2 };

  struct prefetch_handle h00 = prefetch_cache_request(
    fx.shard_index_cache, shard_index_key_hash(&probe00), &probe00, 0, &gate);
  struct prefetch_handle h01 = prefetch_cache_request(
    fx.shard_index_cache, shard_index_key_hash(&probe01), &probe01, 0, &gate);
  EXPECT(prefetch_handle_valid(h00));
  EXPECT(prefetch_handle_valid(h01));
  EXPECT(h00.slot != h01.slot);

  struct prefetch_cache_stats st;
  prefetch_cache_stats_get(fx.shard_index_cache, &st);
  EXPECT(st.size == 2);
  EXPECT(st.counters.misses == 2);

  fixture_teardown(&fx);
  return 0;
}

static int
test_dedup_same_uri_same_coord(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx) == 0);
  EXPECT(write_array(fx.root, "foo", MINIMAL_ZARR_JSON) == 0);
  EXPECT(write_shard(fx.root, "foo", 0, 0) == 0);
  EXPECT(warm_meta(&fx, "foo") == 0);

  struct shard_index_key probe = { .uri = "foo",
                                   .shard_coord = { 0, 0 },
                                   .rank = 2 };
  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct prefetch_handle h1 = prefetch_cache_request(
    fx.shard_index_cache, shard_index_key_hash(&probe), &probe, 0, &gate);
  struct prefetch_handle h2 = prefetch_cache_request(
    fx.shard_index_cache, shard_index_key_hash(&probe), &probe, 0, &gate);
  EXPECT(h1.slot == h2.slot);
  EXPECT(h1.generation == h2.generation);

  struct prefetch_cache_stats st;
  prefetch_cache_stats_get(fx.shard_index_cache, &st);
  EXPECT(st.counters.hits == 1);
  EXPECT(st.counters.misses == 1);

  fixture_teardown(&fx);
  return 0;
}

int
main(void)
{
  RUN(test_shard_index_parses_footer);
  RUN(test_missing_shard_returns_notfound);
  RUN(test_unreadable_shard_returns_io);
  RUN(test_different_coords_are_separate_entries);
  RUN(test_dedup_same_uri_same_coord);
  log_info("all tests passed");
  return 0;
}
