#include "damacy.h"
#include "fixture.h"
#include "prefetch/array_meta.h"
#include "prefetch/prefetch_cache.h"
#include "store/store.h"
#include "store/store_fs.h"
#include "util/hash.h"
#include "zarr/zarr_metadata.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

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
  struct array_meta_fetcher fetcher;
  struct prefetch_cache* cache;
};

static int
fixture_setup(struct fixture* fx)
{
  char tmpl[] = "/tmp/damacy_array_meta_XXXXXX";
  EXPECT(mkdtemp(tmpl));
  fx->root = strdup(tmpl);
  EXPECT(fx->root);

  struct store_fs_config sc = { .root = fx->root, .nthreads = 1 };
  fx->store = store_fs_create(&sc);
  EXPECT(fx->store);

  array_meta_fetcher_init(&fx->fetcher, fx->store);
  struct prefetch_cache_config cfg = {
    .capacity = 4,
    .max_probe = 16,
    .ops = &array_meta_ops,
    .fetcher = &fx->fetcher.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->cache = prefetch_cache_create(&cfg);
  EXPECT(fx->cache);
  return 0;
}

static void
fixture_teardown(struct fixture* fx)
{
  if (fx->cache)
    prefetch_cache_destroy(fx->cache);
  if (fx->store)
    store_destroy(fx->store);
  if (fx->root) {
    fixture_rm_tree(fx->root);
    free(fx->root);
  }
}

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
test_fetch_parses_zarr_json(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx) == 0);
  EXPECT(write_array(fx.root, "foo", MINIMAL_ZARR_JSON) == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct prefetch_handle h =
    prefetch_cache_request(fx.cache, hash_fnv1a_str("foo"), "foo", 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  const struct zarr_metadata* meta =
    (const struct zarr_metadata*)prefetch_cache_try_get(fx.cache, h);
  EXPECT(meta);
  EXPECT(meta->rank == 2);
  EXPECT(meta->shape[0] == 64 && meta->shape[1] == 1024);
  EXPECT(meta->inner_chunk_shape[0] == 32 && meta->inner_chunk_shape[1] == 128);
  EXPECT(meta->shard_shape[0] == 64 && meta->shard_shape[1] == 512);
  EXPECT(meta->sharded == 1);

  fixture_teardown(&fx);
  return 0;
}

static int
test_missing_uri_returns_notfound(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx) == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct prefetch_handle h = prefetch_cache_request(
    fx.cache, hash_fnv1a_str("absent"), "absent", 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_has_error(&gate));

  int err = 0;
  EXPECT(prefetch_cache_query(fx.cache, h, NULL, &err) == PREFETCH_STATE_ERROR);
  EXPECT(err == DAMACY_NOTFOUND);

  fixture_teardown(&fx);
  return 0;
}

static int
test_malformed_json_returns_decode(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx) == 0);
  EXPECT(write_array(fx.root, "bad", "{ not valid json") == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct prefetch_handle h =
    prefetch_cache_request(fx.cache, hash_fnv1a_str("bad"), "bad", 0, &gate);
  EXPECT(prefetch_handle_valid(h));

  int err = 0;
  EXPECT(prefetch_cache_query(fx.cache, h, NULL, &err) == PREFETCH_STATE_ERROR);
  EXPECT(err == DAMACY_DECODE);

  fixture_teardown(&fx);
  return 0;
}

static int
test_dedup_returns_same_parsed_metadata(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx) == 0);
  EXPECT(write_array(fx.root, "foo", MINIMAL_ZARR_JSON) == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct prefetch_handle h1 =
    prefetch_cache_request(fx.cache, hash_fnv1a_str("foo"), "foo", 0, &gate);
  struct prefetch_handle h2 =
    prefetch_cache_request(fx.cache, hash_fnv1a_str("foo"), "foo", 0, &gate);
  EXPECT(h1.slot == h2.slot);
  EXPECT(h1.generation == h2.generation);

  const void* v1 = prefetch_cache_try_get(fx.cache, h1);
  const void* v2 = prefetch_cache_try_get(fx.cache, h2);
  EXPECT(v1 == v2);

  struct prefetch_cache_stats st;
  prefetch_cache_stats_get(fx.cache, &st);
  EXPECT(st.counters.hits == 1);
  EXPECT(st.counters.misses == 1);

  fixture_teardown(&fx);
  return 0;
}

int
main(void)
{
  RUN(test_fetch_parses_zarr_json);
  RUN(test_missing_uri_returns_notfound);
  RUN(test_malformed_json_returns_decode);
  RUN(test_dedup_returns_same_parsed_metadata);
  log_info("all tests passed");
  return 0;
}
