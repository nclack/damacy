#include "damacy.h"
#include "damacy_limits.h"
#include "fixture.h"
#include "lookahead/lookahead.h"
#include "prefetch/array_meta.h"
#include "prefetch/chunk_layout.h"
#include "prefetch/prefetch_cache.h"
#include "prefetch/prefetcher.h"
#include "prefetch/shard_index.h"
#include "store/store.h"
#include "store/store_fs.h"

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
  struct damacy_lookahead la;
  struct array_meta_fetcher amf;
  struct prefetch_cache* amc;
  struct shard_index_fetcher sif;
  struct prefetch_cache* sic;
  struct chunk_layout_fetcher clf;
  struct prefetch_cache* clc;
  struct prefetcher* p;
};

static int
fixture_setup_layout(struct fixture* fx,
                     const char* codec,
                     const int64_t shape[2],
                     const int64_t inner[2],
                     const int64_t shard[2])
{
  char tmpl[] = "/tmp/damacy_prefetcher_XXXXXX";
  EXPECT(mkdtemp(tmpl));
  fx->root = strdup(tmpl);
  EXPECT(fx->root);

  char p[256];
  snprintf(p, sizeof p, "%s/foo", fx->root);
  EXPECT(fixture_write_zarr_codec(
           p, shape, inner, shard, 2, "uint16", 0, codec) == 0);

  struct store_fs_config sc = { .root = fx->root, .nthreads = 1 };
  fx->store = store_fs_create(&sc);
  EXPECT(fx->store);

  array_meta_fetcher_init(&fx->amf, fx->store);
  struct prefetch_cache_config amc_cfg = {
    .capacity = 8,
    .max_probe = 16,
    .ops = &array_meta_ops,
    .fetcher = &fx->amf.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->amc = prefetch_cache_create(&amc_cfg);
  EXPECT(fx->amc);

  shard_index_fetcher_init(&fx->sif, fx->store, fx->amc);
  struct prefetch_cache_config sic_cfg = {
    .capacity = 8,
    .max_probe = 16,
    .ops = &shard_index_ops,
    .fetcher = &fx->sif.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->sic = prefetch_cache_create(&sic_cfg);
  EXPECT(fx->sic);

  chunk_layout_fetcher_init(&fx->clf,
                            fx->store,
                            fx->amc,
                            fx->sic,
                            DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK);
  struct prefetch_cache_config clc_cfg = {
    .capacity = 8,
    .max_probe = 16,
    .ops = &chunk_layout_ops,
    .fetcher = &fx->clf.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->clc = prefetch_cache_create(&clc_cfg);
  EXPECT(fx->clc);

  EXPECT(lookahead_init(&fx->la, 32) == 0);

  struct prefetcher_config pcfg = {
    .lookahead = &fx->la,
    .array_meta_cache = fx->amc,
    .shard_index_cache = fx->sic,
    .chunk_layout_cache = fx->clc,
    .capacity = 16,
    .batch_capacity = 8,
  };
  fx->p = prefetcher_create(&pcfg);
  EXPECT(fx->p);
  EXPECT(prefetcher_start(fx->p) == 0);
  return 0;
}

static int
fixture_setup(struct fixture* fx, const char* codec)
{
  int64_t shape[2] = { 16, 32 };
  int64_t inner[2] = { 8, 16 };
  int64_t shard[2] = { 16, 32 };
  return fixture_setup_layout(fx, codec, shape, inner, shard);
}

static void
fixture_teardown(struct fixture* fx)
{
  if (fx->p)
    prefetcher_destroy(fx->p);
  lookahead_destroy(&fx->la);
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
test_single_sample_walks_all_stages(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_batch(&fx.la, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_stats st;
  prefetcher_stats_get(fx.p, &st);
  EXPECT(st.submitted == 1);
  EXPECT(st.ready == 1);
  EXPECT(st.errored == 0);
  EXPECT(st.in_flight == 0);

  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(fx.amc, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.sic, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.clc, &cs);
  EXPECT(cs.size == 1);

  fixture_teardown(&fx);
  return 0;
}

static int
test_dedup_across_samples(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  for (int i = 0; i < 4; ++i)
    EXPECT(lookahead_push_with_batch(&fx.la, &s, (uint64_t)i) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_stats st;
  prefetcher_stats_get(fx.p, &st);
  EXPECT(st.submitted == 4);
  EXPECT(st.ready == 4);

  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(fx.amc, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.sic, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.clc, &cs);
  EXPECT(cs.size == 1);

  fixture_teardown(&fx);
  return 0;
}

static int
test_error_propagates(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "missing", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_batch(&fx.la, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_stats st;
  prefetcher_stats_get(fx.p, &st);
  EXPECT(st.submitted == 1);
  EXPECT(st.ready == 0);
  EXPECT(st.errored == 1);

  fixture_teardown(&fx);
  return 0;
}

static int
test_stop_restart_idempotent(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  prefetcher_stop(fx.p);
  prefetcher_stop(fx.p);

  fixture_teardown(&fx);
  return 0;
}

static int
test_multi_shard_sample_warms_all_shards(void)
{
  struct fixture fx = { 0 };
  int64_t shape[2] = { 32, 64 };
  int64_t inner[2] = { 8, 16 };
  int64_t shard[2] = { 16, 32 };
  EXPECT(fixture_setup_layout(&fx, "blosc-zstd", shape, inner, shard) == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 32 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 64 };

  EXPECT(lookahead_push_with_batch(&fx.la, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_stats st;
  prefetcher_stats_get(fx.p, &st);
  EXPECT(st.submitted == 1);
  EXPECT(st.ready == 1);
  EXPECT(st.errored == 0);

  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(fx.amc, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.sic, &cs);
  EXPECT(cs.size == 4);
  prefetch_cache_stats_get(fx.clc, &cs);
  EXPECT(cs.size == 1);

  fixture_teardown(&fx);
  return 0;
}

static int
test_pop_ready_returns_completed_sample(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_batch(&fx.la, &s, 7) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  EXPECT(r.state == PREFETCHER_READY);
  EXPECT(strcmp(r.uri, "foo") == 0);
  EXPECT(r.batch_id == 7);
  EXPECT(r.n_shards == 1);
  EXPECT(prefetch_handle_valid(r.h_meta));
  EXPECT(prefetch_handle_valid(r.h_shards[0]));
  EXPECT(prefetch_handle_valid(r.h_layout));
  prefetcher_ready_free(&r);

  EXPECT(prefetcher_pop_ready(fx.p, &r) == 0);

  fixture_teardown(&fx);
  return 0;
}

static int
test_pop_ready_surfaces_error_state(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "missing", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_batch(&fx.la, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  EXPECT(r.state == PREFETCHER_ERROR);
  EXPECT(r.err_code != 0);
  EXPECT(strcmp(r.uri, "missing") == 0);
  prefetcher_ready_free(&r);

  fixture_teardown(&fx);
  return 0;
}

static int
test_pop_ready_recycles_slot(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  for (int round = 0; round < 3; ++round) {
    EXPECT(lookahead_push_with_batch(&fx.la, &s, (uint64_t)round) == 0);
    EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);
    struct prefetcher_ready r = { 0 };
    EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
    EXPECT(r.batch_id == (uint64_t)round);
    prefetcher_ready_free(&r);
  }

  struct prefetcher_stats st;
  prefetcher_stats_get(fx.p, &st);
  EXPECT(st.submitted == 3);
  EXPECT(st.ready == 3);
  EXPECT(st.in_flight == 0);

  fixture_teardown(&fx);
  return 0;
}

static int
test_pop_ready_empty_returns_zero(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);
  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 0);
  fixture_teardown(&fx);
  return 0;
}

static int
test_batch_gate_ready_after_drain(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 42) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  const struct prefetch_gate* g = prefetcher_batch_gate(fx.p, 42);
  EXPECT(g);
  EXPECT(prefetch_gate_is_ready(g));
  EXPECT(!prefetch_gate_has_error(g));
  EXPECT(prefetch_gate_pending(g) == 0);

  fixture_teardown(&fx);
  return 0;
}

static int
test_batch_gate_error_on_failed_sample(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "missing", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 7) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  const struct prefetch_gate* g = prefetcher_batch_gate(fx.p, 7);
  EXPECT(g);
  EXPECT(prefetch_gate_has_error(g));
  EXPECT(prefetch_gate_pending(g) == 0);

  fixture_teardown(&fx);
  return 0;
}

static int
test_distinct_batches_get_separate_gates(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 1) == 0);
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 2) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  const struct prefetch_gate* g1 = prefetcher_batch_gate(fx.p, 1);
  const struct prefetch_gate* g2 = prefetcher_batch_gate(fx.p, 2);
  EXPECT(g1);
  EXPECT(g2);
  EXPECT(g1 != g2);

  fixture_teardown(&fx);
  return 0;
}

static int
test_release_batch_drops_gate(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 99) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  EXPECT(prefetcher_batch_gate(fx.p, 99));
  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  prefetcher_ready_free(&r);

  prefetcher_release_batch(fx.p, 99);
  EXPECT(prefetcher_batch_gate(fx.p, 99) == NULL);

  fixture_teardown(&fx);
  return 0;
}

static int
test_release_before_pop_defers(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 99) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  const struct prefetch_gate* g_before = prefetcher_batch_gate(fx.p, 99);
  EXPECT(g_before);

  prefetcher_release_batch(fx.p, 99);
  EXPECT(prefetcher_batch_gate(fx.p, 99) == g_before);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  prefetcher_ready_free(&r);

  EXPECT(prefetcher_batch_gate(fx.p, 99) == NULL);

  fixture_teardown(&fx);
  return 0;
}

static int
test_admit_fail_releases_batch_entry(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  for (int i = 0; i < 8; ++i) {
    char uri[16];
    snprintf(uri, sizeof uri, "missing%d", i);
    s.uri = uri;
    EXPECT(lookahead_push_with_batch(&fx.la, &s, 0) == 0);
  }
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  for (int i = 0; i < 8; ++i) {
    struct prefetcher_ready r = { 0 };
    EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
    prefetcher_ready_free(&r);
  }

  s.uri = "missing_overflow";
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 99) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  EXPECT(r.state == PREFETCHER_ERROR);
  EXPECT(r.batch_id == 99);
  EXPECT(r.err_code == DAMACY_OOM);
  prefetcher_ready_free(&r);

  prefetcher_release_batch(fx.p, 99);
  EXPECT(prefetcher_batch_gate(fx.p, 99) == NULL);

  fixture_teardown(&fx);
  return 0;
}

static int
test_unknown_batch_gate_is_null(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);
  EXPECT(prefetcher_batch_gate(fx.p, 12345) == NULL);
  fixture_teardown(&fx);
  return 0;
}

static int
test_pop_ready_for_batch_filters_by_batch_id(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_batch(&fx.la, &s, 5) == 0);
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 7) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready_for_batch(fx.p, 7, &r) == 1);
  EXPECT(r.batch_id == 7);
  prefetcher_ready_free(&r);

  EXPECT(prefetcher_pop_ready_for_batch(fx.p, 7, &r) == 0);

  EXPECT(prefetcher_pop_ready_for_batch(fx.p, 5, &r) == 1);
  EXPECT(r.batch_id == 5);
  prefetcher_ready_free(&r);

  fixture_teardown(&fx);
  return 0;
}

static int
test_batch_readiness_helpers(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  EXPECT(prefetcher_has_ready(fx.p) == 0);
  EXPECT(prefetcher_in_flight(fx.p) == 0);
  EXPECT(prefetcher_ready_count_for_batch(fx.p, 0) == 0);
  EXPECT(prefetcher_batch_full_ready(fx.p, 0, 1) == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_batch(&fx.la, &s, 3) == 0);
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 3) == 0);
  EXPECT(lookahead_push_with_batch(&fx.la, &s, 4) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  EXPECT(prefetcher_has_ready(fx.p) == 1);
  EXPECT(prefetcher_ready_count_for_batch(fx.p, 3) == 2);
  EXPECT(prefetcher_ready_count_for_batch(fx.p, 4) == 1);
  EXPECT(prefetcher_ready_count_for_batch(fx.p, 9) == 0);
  EXPECT(prefetcher_batch_full_ready(fx.p, 3, 2) == 1);
  EXPECT(prefetcher_batch_full_ready(fx.p, 3, 3) == 0);
  EXPECT(prefetcher_batch_full_ready(fx.p, 4, 1) == 1);

  fixture_teardown(&fx);
  return 0;
}

static int
test_advance_watermark_broadcasts(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(fx.amc, &cs);
  EXPECT(cs.watermark == 0);
  prefetch_cache_stats_get(fx.sic, &cs);
  EXPECT(cs.watermark == 0);
  prefetch_cache_stats_get(fx.clc, &cs);
  EXPECT(cs.watermark == 0);

  prefetcher_advance_watermark(fx.p, 7);

  prefetch_cache_stats_get(fx.amc, &cs);
  EXPECT(cs.watermark == 7);
  prefetch_cache_stats_get(fx.sic, &cs);
  EXPECT(cs.watermark == 7);
  prefetch_cache_stats_get(fx.clc, &cs);
  EXPECT(cs.watermark == 7);

  fixture_teardown(&fx);
  return 0;
}

static int
test_batch_capacity_saturation_surfaces_error(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  for (uint64_t i = 0; i < 8; ++i)
    EXPECT(lookahead_push_with_batch(&fx.la, &s, i) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  for (int i = 0; i < 8; ++i) {
    struct prefetcher_ready r = { 0 };
    EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
    EXPECT(r.state == PREFETCHER_READY);
    prefetcher_ready_free(&r);
  }

  EXPECT(lookahead_push_with_batch(&fx.la, &s, 99) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  EXPECT(r.state == PREFETCHER_ERROR);
  EXPECT(r.batch_id == 99);
  EXPECT(r.err_code == DAMACY_OOM);
  prefetcher_ready_free(&r);

  prefetcher_release_batch(fx.p, 99);
  EXPECT(prefetcher_batch_gate(fx.p, 99) == NULL);

  fixture_teardown(&fx);
  return 0;
}

int
main(void)
{
  RUN(test_single_sample_walks_all_stages);
  RUN(test_dedup_across_samples);
  RUN(test_error_propagates);
  RUN(test_stop_restart_idempotent);
  RUN(test_multi_shard_sample_warms_all_shards);
  RUN(test_pop_ready_returns_completed_sample);
  RUN(test_pop_ready_surfaces_error_state);
  RUN(test_pop_ready_recycles_slot);
  RUN(test_pop_ready_empty_returns_zero);
  RUN(test_batch_gate_ready_after_drain);
  RUN(test_batch_gate_error_on_failed_sample);
  RUN(test_distinct_batches_get_separate_gates);
  RUN(test_release_batch_drops_gate);
  RUN(test_release_before_pop_defers);
  RUN(test_admit_fail_releases_batch_entry);
  RUN(test_unknown_batch_gate_is_null);
  RUN(test_pop_ready_for_batch_filters_by_batch_id);
  RUN(test_batch_readiness_helpers);
  RUN(test_advance_watermark_broadcasts);
  RUN(test_batch_capacity_saturation_surfaces_error);
  log_info("all tests passed");
  return 0;
}
