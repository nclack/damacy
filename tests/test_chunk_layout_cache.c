#include "damacy.h"
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
  struct array_meta_fetcher amf;
  struct prefetch_cache* amc;
  struct shard_index_fetcher sif;
  struct prefetch_cache* sic;
  struct chunk_layout_fetcher clf;
  struct prefetch_cache* clc;
};

static int
fixture_setup(struct fixture* fx, const char* codec)
{
  char tmpl[] = "/tmp/damacy_clayout_XXXXXX";
  EXPECT(mkdtemp(tmpl));
  fx->root = strdup(tmpl);
  EXPECT(fx->root);

  char p[256];
  snprintf(p, sizeof p, "%s/foo", fx->root);
  int64_t shape[2] = { 16, 32 };
  int64_t inner[2] = { 8, 16 };
  int64_t shard[2] = { 16, 32 };
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, codec) == 0);

  struct store_fs_config sc = { .root = fx->root, .nthreads = 1 };
  fx->store = store_fs_create(&sc);
  EXPECT(fx->store);

  array_meta_fetcher_init(&fx->amf, fx->store);
  struct prefetch_cache_config amc_cfg = {
    .capacity = 4,
    .max_probe = 16,
    .ops = &array_meta_ops,
    .fetcher = &fx->amf.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->amc = prefetch_cache_create(&amc_cfg);
  EXPECT(fx->amc);

  shard_index_fetcher_init(&fx->sif, fx->store, fx->amc);
  struct prefetch_cache_config sic_cfg = {
    .capacity = 4,
    .max_probe = 16,
    .ops = &shard_index_ops,
    .fetcher = &fx->sif.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->sic = prefetch_cache_create(&sic_cfg);
  EXPECT(fx->sic);

  chunk_layout_fetcher_init(&fx->clf, fx->store, fx->amc, fx->sic);
  struct prefetch_cache_config clc_cfg = {
    .capacity = 4,
    .max_probe = 16,
    .ops = &chunk_layout_ops,
    .fetcher = &fx->clf.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->clc = prefetch_cache_create(&clc_cfg);
  EXPECT(fx->clc);
  return 0;
}

static void
fixture_teardown(struct fixture* fx)
{
  if (fx->clc)
    prefetch_cache_destroy(fx->clc);
  if (fx->sic)
    prefetch_cache_destroy(fx->sic);
  if (fx->amc)
    prefetch_cache_destroy(fx->amc);
  if (fx->store)
    store_destroy(fx->store);
  if (fx->root) {
    fixture_rm_tree(fx->root);
    free(fx->root);
  }
}

static int
warm_upstream(struct fixture* fx, const char* uri)
{
  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h_meta =
    prefetch_cache_request(fx->amc, hash_fnv1a_str(uri), uri, 0, &gate);
  EXPECT(prefetch_handle_valid(h_meta));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  struct shard_index_key probe = { .uri = uri, .rank = 2 };
  probe.shard_coord[0] = 0;
  probe.shard_coord[1] = 0;
  prefetch_gate_init(&gate);
  struct prefetch_handle h_si = prefetch_cache_request(
    fx->sic, shard_index_key_hash(&probe), &probe, 0, &gate);
  EXPECT(prefetch_handle_valid(h_si));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));
  return 0;
}

static int
test_probes_blosc_zstd_layout(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);
  EXPECT(warm_upstream(&fx, "foo") == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h =
    prefetch_cache_request(fx.clc, hash_fnv1a_str("foo"), "foo", 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  const struct chunk_layout* layout =
    (const struct chunk_layout*)prefetch_cache_try_get(fx.clc, h);
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
test_non_blosc_codec_errors(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "zstd") == 0);
  EXPECT(warm_upstream(&fx, "foo") == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h =
    prefetch_cache_request(fx.clc, hash_fnv1a_str("foo"), "foo", 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_has_error(&gate));

  int err = 0;
  EXPECT(prefetch_cache_query(fx.clc, h, NULL, &err) == PREFETCH_STATE_ERROR);
  EXPECT(err == DAMACY_DECODE);

  fixture_teardown(&fx);
  return 0;
}

static int
test_dedup_returns_same_layout(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);
  EXPECT(warm_upstream(&fx, "foo") == 0);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h1 =
    prefetch_cache_request(fx.clc, hash_fnv1a_str("foo"), "foo", 0, &gate);
  struct prefetch_handle h2 =
    prefetch_cache_request(fx.clc, hash_fnv1a_str("foo"), "foo", 0, &gate);
  EXPECT(h1.slot == h2.slot);
  EXPECT(h1.generation == h2.generation);

  const void* v1 = prefetch_cache_try_get(fx.clc, h1);
  const void* v2 = prefetch_cache_try_get(fx.clc, h2);
  EXPECT(v1 && v1 == v2);

  fixture_teardown(&fx);
  return 0;
}

int
main(void)
{
  RUN(test_probes_blosc_zstd_layout);
  RUN(test_non_blosc_codec_errors);
  RUN(test_dedup_returns_same_layout);
  log_info("all tests passed");
  return 0;
}
