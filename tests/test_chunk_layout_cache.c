#include "damacy.h"
#include "damacy_limits.h"
#include "fixture.h"
#include "prefetch/array_meta.h"
#include "prefetch/chunk_layout.h"
#include "prefetch/prefetch_cache.h"
#include "prefetch/shard_index.h"
#include "store/store.h"
#include "store/store_fs.h"
#include "util/hash.h"
#include "zarr/zarr_chunk_layout.h"
#include "zarr/zarr_metadata.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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
  struct chunk_layout_fetcher chunk_layout_fetcher;
  struct prefetch_cache* chunk_layout_cache;
};

static int
fixture_setup_caches(struct fixture* fx)
{
  char tmpl[] = "/tmp/damacy_clayout_XXXXXX";
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
    .capacity = 4,
    .max_probe = 16,
    .ops = &shard_index_ops,
    .fetcher = &fx->shard_index_fetcher.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->shard_index_cache = prefetch_cache_create(&sic_cfg);
  EXPECT(fx->shard_index_cache);

  chunk_layout_fetcher_init(&fx->chunk_layout_fetcher,
                            fx->store,
                            fx->array_meta_cache,
                            fx->shard_index_cache,
                            DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK);
  struct prefetch_cache_config clc_cfg = {
    .capacity = 4,
    .max_probe = 16,
    .ops = &chunk_layout_ops,
    .fetcher = &fx->chunk_layout_fetcher.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->chunk_layout_cache = prefetch_cache_create(&clc_cfg);
  EXPECT(fx->chunk_layout_cache);
  return 0;
}

static int
fixture_setup_layout(struct fixture* fx,
                     const char* codec,
                     const int64_t shard[2])
{
  EXPECT(fixture_setup_caches(fx) == 0);
  char p[256];
  snprintf(p, sizeof p, "%s/foo", fx->root);
  int64_t shape[2] = { 16, 32 };
  int64_t inner[2] = { 8, 16 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, codec) == 0);
  return 0;
}

static int
fixture_setup(struct fixture* fx, const char* codec)
{
  int64_t shard[2] = { 16, 32 };
  return fixture_setup_layout(fx, codec, shard);
}

static void
fixture_teardown(struct fixture* fx)
{
  if (fx->chunk_layout_cache)
    prefetch_cache_destroy(fx->chunk_layout_cache);
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
warm_upstream(struct fixture* fx,
              const char* uri,
              uint64_t y,
              uint64_t x,
              struct prefetch_handle* out_h_shard,
              uint64_t out_shard_coord[DAMACY_MAX_RANK])
{
  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h_meta = prefetch_cache_request(
    fx->array_meta_cache, hash_fnv1a_str(uri), uri, 0, &gate);
  EXPECT(prefetch_handle_valid(h_meta));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  struct shard_index_key probe = { .uri = uri, .rank = 2 };
  probe.shard_coord[0] = y;
  probe.shard_coord[1] = x;
  out_shard_coord[0] = y;
  out_shard_coord[1] = x;
  prefetch_gate_init(&gate);
  struct prefetch_handle h_si = prefetch_cache_request(
    fx->shard_index_cache, shard_index_key_hash(&probe), &probe, 0, &gate);
  EXPECT(prefetch_handle_valid(h_si));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));
  *out_h_shard = h_si;
  return 0;
}

static struct prefetch_handle
request_layout(struct fixture* fx,
               const char* uri,
               const struct prefetch_handle* h_shards,
               const uint64_t* shard_coords,
               uint32_t n_shards,
               struct prefetch_gate* gate)
{
  struct chunk_layout_key key = {
    .uri = uri,
    .rank = 2,
    .n_shards = n_shards,
    .shard_coords = shard_coords,
    .h_shards = h_shards,
  };
  return prefetch_cache_request(
    fx->chunk_layout_cache, chunk_layout_key_hash(&key), &key, 0, gate);
}

static int
test_probes_blosc_zstd_layout(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);
  struct prefetch_handle h_shard = { 0 };
  uint64_t shard_coord[DAMACY_MAX_RANK] = { 0 };
  EXPECT(warm_upstream(&fx, "foo", 0, 0, &h_shard, shard_coord) == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h =
    request_layout(&fx, "foo", &h_shard, shard_coord, 1, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  const struct chunk_layout* layout =
    (const struct chunk_layout*)prefetch_cache_try_get(fx.chunk_layout_cache,
                                                       h);
  EXPECT(layout);
  EXPECT(layout->codec_id == CODEC_BLOSC_ZSTD);
  EXPECT(layout->typesize == 2);
  EXPECT(layout->nbytes > 0);
  EXPECT(layout->blocksize > 0);
  EXPECT(layout->nblocks > 0);

  fixture_teardown(&fx);
  return 0;
}

static int
test_non_blosc_codec_yields_no_layout(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "zstd") == 0);
  struct prefetch_handle h_shard = { 0 };
  uint64_t shard_coord[DAMACY_MAX_RANK] = { 0 };
  EXPECT(warm_upstream(&fx, "foo", 0, 0, &h_shard, shard_coord) == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h =
    request_layout(&fx, "foo", &h_shard, shard_coord, 1, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  // Non-blosc samples carry no blosc1-specific layout; decoder uses caps.
  EXPECT(prefetch_cache_try_get(fx.chunk_layout_cache, h) == NULL);

  fixture_teardown(&fx);
  return 0;
}

static int
test_probes_non_origin_shard_when_origin_missing(void)
{
  struct fixture fx = { 0 };
  int64_t shard[2] = { 8, 16 };
  EXPECT(fixture_setup_layout(&fx, "blosc-zstd", shard) == 0);

  char origin_path[512];
  snprintf(origin_path, sizeof origin_path, "%s/foo/c/0/0", fx.root);
  EXPECT(unlink(origin_path) == 0);

  struct prefetch_handle h_shard = { 0 };
  uint64_t shard_coord[DAMACY_MAX_RANK] = { 0 };
  EXPECT(warm_upstream(&fx, "foo", 1, 0, &h_shard, shard_coord) == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h =
    request_layout(&fx, "foo", &h_shard, shard_coord, 1, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));
  EXPECT(prefetch_cache_try_get(fx.chunk_layout_cache, h) != NULL);

  fixture_teardown(&fx);
  return 0;
}

// Guards against silently returning NULL for unsupported codecs (would
// reach READY then fail late at plan time).
static const char* BLOSC_LZ4_ZARR_JSON =
  "{"
  "\"zarr_format\":3,"
  "\"node_type\":\"array\","
  "\"shape\":[16,32],"
  "\"data_type\":\"uint16\","
  "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
  "\"chunk_shape\":[16,32]}},"
  "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
  "\"separator\":\"/\"}},"
  "\"fill_value\":0,"
  "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{"
  "\"chunk_shape\":[8,16],"
  "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
  "{\"name\":\"blosc\",\"configuration\":{\"cname\":\"lz4\"}}],"
  "\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{"
  "\"endian\":\"little\"}},{\"name\":\"crc32c\"}],"
  "\"index_location\":\"end\"}}]"
  "}";

static int
test_unsupported_codec_errors(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup_caches(&fx) == 0);

  char path[512];
  snprintf(path, sizeof path, "%s/foo", fx.root);
  EXPECT(mkdir(path, 0755) == 0);
  snprintf(path, sizeof path, "%s/foo/zarr.json", fx.root);
  EXPECT(fixture_write_file(path, BLOSC_LZ4_ZARR_JSON) == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h_meta = prefetch_cache_request(
    fx.array_meta_cache, hash_fnv1a_str("foo"), "foo", 0, &gate);
  EXPECT(prefetch_handle_valid(h_meta));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  prefetch_gate_init(&gate);
  struct prefetch_handle h = request_layout(&fx, "foo", NULL, NULL, 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_has_error(&gate));

  int err = 0;
  EXPECT(prefetch_cache_query(fx.chunk_layout_cache, h, NULL, &err) ==
         PREFETCH_STATE_ERROR);
  EXPECT(err == DAMACY_DECODE);

  fixture_teardown(&fx);
  return 0;
}

static int
test_dedup_returns_same_layout(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);
  struct prefetch_handle h_shard = { 0 };
  uint64_t shard_coord[DAMACY_MAX_RANK] = { 0 };
  EXPECT(warm_upstream(&fx, "foo", 0, 0, &h_shard, shard_coord) == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h1 =
    request_layout(&fx, "foo", &h_shard, shard_coord, 1, &gate);
  struct prefetch_handle h2 =
    request_layout(&fx, "foo", &h_shard, shard_coord, 1, &gate);
  EXPECT(h1.slot == h2.slot);
  EXPECT(h1.generation == h2.generation);

  const void* v1 = prefetch_cache_try_get(fx.chunk_layout_cache, h1);
  const void* v2 = prefetch_cache_try_get(fx.chunk_layout_cache, h2);
  EXPECT(v1 && v1 == v2);

  fixture_teardown(&fx);
  return 0;
}

int
main(void)
{
  RUN(test_probes_blosc_zstd_layout);
  RUN(test_non_blosc_codec_yields_no_layout);
  RUN(test_probes_non_origin_shard_when_origin_missing);
  RUN(test_unsupported_codec_errors);
  RUN(test_dedup_returns_same_layout);
  log_info("all tests passed");
  return 0;
}
