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
  struct damacy_lookahead lookahead;
  struct array_meta_fetcher array_meta_fetcher;
  struct prefetch_cache* array_meta_cache;
  struct shard_index_fetcher shard_index_fetcher;
  struct prefetch_cache* shard_index_cache;
  struct chunk_layout_fetcher chunk_layout_fetcher;
  struct prefetch_cache* chunk_layout_cache;
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

  array_meta_fetcher_init(&fx->array_meta_fetcher, fx->store);
  struct prefetch_cache_config amc_cfg = {
    .capacity = 8,
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

  chunk_layout_fetcher_init(&fx->chunk_layout_fetcher,
                            fx->store,
                            fx->array_meta_cache,
                            fx->shard_index_cache,
                            DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK);
  struct prefetch_cache_config clc_cfg = {
    .capacity = 8,
    .max_probe = 16,
    .ops = &chunk_layout_ops,
    .fetcher = &fx->chunk_layout_fetcher.base,
    .executor = &SYNC_EXECUTOR,
  };
  fx->chunk_layout_cache = prefetch_cache_create(&clc_cfg);
  EXPECT(fx->chunk_layout_cache);

  EXPECT(lookahead_init(&fx->lookahead, 32) == 0);

  struct prefetcher_config pcfg = {
    .lookahead = &fx->lookahead,
    .array_meta_cache = fx->array_meta_cache,
    .shard_index_cache = fx->shard_index_cache,
    .chunk_layout_cache = fx->chunk_layout_cache,
    .capacity = 16,
    .owner_capacity = 8,
    .max_shards_per_sample = 64,
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
  lookahead_destroy(&fx->lookahead);
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
test_single_sample_walks_all_stages(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_stats st;
  prefetcher_stats_get(fx.p, &st);
  EXPECT(st.submitted == 1);
  EXPECT(st.ready == 1);
  EXPECT(st.errored == 0);
  EXPECT(st.in_flight == 0);

  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(fx.array_meta_cache, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.shard_index_cache, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.chunk_layout_cache, &cs);
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
    EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, (uint64_t)i) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_stats st;
  prefetcher_stats_get(fx.p, &st);
  EXPECT(st.submitted == 4);
  EXPECT(st.ready == 4);

  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(fx.array_meta_cache, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.shard_index_cache, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.chunk_layout_cache, &cs);
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

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
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

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_stats st;
  prefetcher_stats_get(fx.p, &st);
  EXPECT(st.submitted == 1);
  EXPECT(st.ready == 1);
  EXPECT(st.errored == 0);

  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(fx.array_meta_cache, &cs);
  EXPECT(cs.size == 1);
  prefetch_cache_stats_get(fx.shard_index_cache, &cs);
  EXPECT(cs.size == 4);
  prefetch_cache_stats_get(fx.chunk_layout_cache, &cs);
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

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  EXPECT(r.result == PREFETCHER_RESULT_READY);
  EXPECT(strcmp(r.uri, "foo") == 0);
  EXPECT(r.sample_seq == 0);
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

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  EXPECT(r.result == PREFETCHER_RESULT_ERROR);
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
    EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, (uint64_t)round) ==
           0);
    EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);
    struct prefetcher_ready r = { 0 };
    EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
    EXPECT(r.sample_seq == (uint64_t)round);
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
test_sample_gate_ready_after_drain(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  const struct prefetch_gate* g = prefetcher_sample_gate(fx.p, 0);
  EXPECT(g);
  EXPECT(prefetch_gate_is_ready(g));
  EXPECT(!prefetch_gate_has_error(g));
  EXPECT(prefetch_gate_pending(g) == 0);

  fixture_teardown(&fx);
  return 0;
}

static int
test_sample_gate_error_on_failed_sample(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "missing", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  const struct prefetch_gate* g = prefetcher_sample_gate(fx.p, 0);
  EXPECT(g);
  EXPECT(prefetch_gate_has_error(g));
  EXPECT(prefetch_gate_pending(g) == 0);

  fixture_teardown(&fx);
  return 0;
}

static int
test_distinct_samples_get_separate_gates(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 1) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  const struct prefetch_gate* g1 = prefetcher_sample_gate(fx.p, 0);
  const struct prefetch_gate* g2 = prefetcher_sample_gate(fx.p, 1);
  EXPECT(g1);
  EXPECT(g2);
  EXPECT(g1 != g2);

  fixture_teardown(&fx);
  return 0;
}

static int
test_pop_drops_sample_gate(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  EXPECT(prefetcher_sample_gate(fx.p, 0));
  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  prefetcher_ready_free(&r);

  EXPECT(prefetcher_sample_gate(fx.p, 0) == NULL);

  fixture_teardown(&fx);
  return 0;
}

static int
test_sample_gate_lives_until_pop(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  const struct prefetch_gate* g_before = prefetcher_sample_gate(fx.p, 0);
  EXPECT(g_before);
  EXPECT(prefetcher_sample_gate(fx.p, 0) == g_before);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  prefetcher_ready_free(&r);

  EXPECT(prefetcher_sample_gate(fx.p, 0) == NULL);

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
    EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, (uint64_t)i) == 0);
  }
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  s.uri = "missing_overflow";
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 8) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  for (int i = 0; i < 8; ++i) {
    struct prefetcher_ready r = { 0 };
    EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
    prefetcher_ready_free(&r);
  }

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  EXPECT(r.result == PREFETCHER_RESULT_ERROR);
  EXPECT(r.sample_seq == 8);
  EXPECT(r.err_code == DAMACY_BUDGET);
  prefetcher_ready_free(&r);

  EXPECT(prefetcher_sample_gate(fx.p, 8) == NULL);

  fixture_teardown(&fx);
  return 0;
}

static int
test_unknown_sample_gate_is_null(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);
  EXPECT(prefetcher_sample_gate(fx.p, 12345) == NULL);
  fixture_teardown(&fx);
  return 0;
}

static int
test_take_wave_yields_ticket(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  EXPECT(prefetcher_has_ready(fx.p) == 0);
  EXPECT(prefetcher_in_flight(fx.p) == 0);
  EXPECT(prefetcher_ready_prefix_count(fx.p) == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 1) == 0);
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 2) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  EXPECT(prefetcher_has_ready(fx.p) == 1);
  EXPECT(prefetcher_ready_prefix_count(fx.p) == 3);
  EXPECT(prefetcher_unconsumed_count(fx.p, 3) == 3);

  struct prefetcher_ready ready[3] = { 0 };
  struct prefetcher_wave_ticket ticket = { 0 };
  EXPECT(prefetcher_take_ready_wave(fx.p, 2, &ticket, ready) == 1);
  EXPECT(ticket.sample_seq_begin == 0);
  EXPECT(ticket.n_samples == 2);
  EXPECT(ready[0].sample_seq == 0);
  EXPECT(ready[1].sample_seq == 1);
  prefetcher_ready_free(&ready[0]);
  prefetcher_ready_free(&ready[1]);
  EXPECT(prefetcher_ready_prefix_count(fx.p) == 1);
  EXPECT(prefetcher_unconsumed_count(fx.p, 3) == 1);

  EXPECT(prefetcher_take_ready_wave(fx.p, 1, &ticket, ready) == 1);
  EXPECT(ticket.sample_seq_begin == 2);
  EXPECT(ticket.n_samples == 1);
  EXPECT(ready[0].sample_seq == 2);
  prefetcher_ready_free(&ready[0]);
  EXPECT(prefetcher_unconsumed_count(fx.p, 3) == 0);

  fixture_teardown(&fx);
  return 0;
}

static int
test_take_wave_caps_to_ready_prefix(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);
  EXPECT(prefetcher_ready_prefix_count(fx.p) == 1);

  struct prefetcher_ready ready[2] = { 0 };
  struct prefetcher_wave_ticket ticket = { 0 };
  EXPECT(prefetcher_take_ready_wave(fx.p, 2, &ticket, ready) == 1);
  EXPECT(ticket.sample_seq_begin == 0);
  EXPECT(ticket.n_samples == 1);
  EXPECT(ready[0].sample_seq == 0);
  prefetcher_ready_free(&ready[0]);
  EXPECT(prefetcher_ready_prefix_count(fx.p) == 0);

  fixture_teardown(&fx);
  return 0;
}

static int
test_ready_wave_waits_for_prefix(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 1) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);
  EXPECT(prefetcher_has_ready(fx.p) == 0);
  EXPECT(prefetcher_ready_prefix_count(fx.p) == 0);

  struct prefetcher_ready ready[2] = { 0 };
  struct prefetcher_wave_ticket ticket = { 0 };
  EXPECT(prefetcher_take_ready_wave(fx.p, 2, &ticket, ready) == 0);
  EXPECT(ticket.n_samples == 0);

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);
  EXPECT(prefetcher_take_ready_wave(fx.p, 2, &ticket, ready) == 1);
  EXPECT(ticket.sample_seq_begin == 0);
  EXPECT(ticket.n_samples == 2);
  EXPECT(ready[0].sample_seq == 0);
  EXPECT(ready[1].sample_seq == 1);
  prefetcher_ready_free(&ready[0]);
  prefetcher_ready_free(&ready[1]);

  fixture_teardown(&fx);
  return 0;
}

static int
test_advance_watermark_broadcasts(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct prefetch_cache_stats cs;
  prefetch_cache_stats_get(fx.array_meta_cache, &cs);
  EXPECT(cs.watermark == 0);
  prefetch_cache_stats_get(fx.shard_index_cache, &cs);
  EXPECT(cs.watermark == 0);
  prefetch_cache_stats_get(fx.chunk_layout_cache, &cs);
  EXPECT(cs.watermark == 0);

  prefetcher_advance_watermark(fx.p, 7);

  prefetch_cache_stats_get(fx.array_meta_cache, &cs);
  EXPECT(cs.watermark == 7);
  prefetch_cache_stats_get(fx.shard_index_cache, &cs);
  EXPECT(cs.watermark == 7);
  prefetch_cache_stats_get(fx.chunk_layout_cache, &cs);
  EXPECT(cs.watermark == 7);

  fixture_teardown(&fx);
  return 0;
}

static int
test_owner_capacity_saturation_surfaces_error(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  for (uint64_t i = 0; i < 8; ++i)
    EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, i) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 8) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  for (int i = 0; i < 8; ++i) {
    struct prefetcher_ready r = { 0 };
    EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
    EXPECT(r.result == PREFETCHER_RESULT_READY);
    prefetcher_ready_free(&r);
  }

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  EXPECT(r.result == PREFETCHER_RESULT_ERROR);
  EXPECT(r.sample_seq == 8);
  EXPECT(r.err_code == DAMACY_BUDGET);
  prefetcher_ready_free(&r);

  EXPECT(prefetcher_sample_gate(fx.p, 8) == NULL);

  fixture_teardown(&fx);
  return 0;
}

static int
test_owner_table_recycles_after_pop(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };

  for (uint64_t i = 0; i < 8; ++i)
    EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, i) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  for (uint64_t i = 0; i < 8; ++i) {
    struct prefetcher_ready r = { 0 };
    EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
    EXPECT(r.result == PREFETCHER_RESULT_READY);
    prefetcher_ready_free(&r);
    EXPECT(prefetcher_sample_gate(fx.p, i) == NULL);
  }

  for (uint64_t i = 8; i < 16; ++i)
    EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, i) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  for (uint64_t i = 8; i < 16; ++i) {
    struct prefetcher_ready r = { 0 };
    EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
    EXPECT(r.result == PREFETCHER_RESULT_READY);
    EXPECT(r.err_code == 0);
    prefetcher_ready_free(&r);
  }

  fixture_teardown(&fx);
  return 0;
}

// Missing shards must not fail the whole sample — the planner needs to
// see ERROR/NOTFOUND per-handle so it can emit fill for those chunks.
static int
test_missing_shard_reaches_ready(void)
{
  struct fixture fx = { 0 };
  EXPECT(fixture_setup(&fx, "blosc-zstd") == 0);

  char shard_path[512];
  snprintf(shard_path, sizeof shard_path, "%s/foo/c/0/0", fx.root);
  EXPECT(unlink(shard_path) == 0);

  struct damacy_sample s = { .uri = "foo", .aabb = { .rank = 2 } };
  s.aabb.dims[0] = (struct damacy_interval){ .beg = 0, .end = 16 };
  s.aabb.dims[1] = (struct damacy_interval){ .beg = 0, .end = 32 };
  EXPECT(lookahead_push_with_sample_seq(&fx.lookahead, &s, 0) == 0);
  EXPECT(prefetcher_drain(fx.p) == DAMACY_OK);

  struct prefetcher_ready r = { 0 };
  EXPECT(prefetcher_pop_ready(fx.p, &r) == 1);
  EXPECT(r.result == PREFETCHER_RESULT_READY);
  EXPECT(r.n_shards == 1);
  EXPECT(prefetch_handle_valid(r.h_layout));
  int err = 0;
  enum prefetch_state st =
    prefetch_cache_query(fx.shard_index_cache, r.h_shards[0], NULL, &err);
  EXPECT(st == PREFETCH_STATE_ERROR);
  EXPECT(err == DAMACY_NOTFOUND);
  prefetcher_ready_free(&r);

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
  RUN(test_sample_gate_ready_after_drain);
  RUN(test_sample_gate_error_on_failed_sample);
  RUN(test_distinct_samples_get_separate_gates);
  RUN(test_pop_drops_sample_gate);
  RUN(test_sample_gate_lives_until_pop);
  RUN(test_admit_fail_releases_batch_entry);
  RUN(test_unknown_sample_gate_is_null);
  RUN(test_take_wave_yields_ticket);
  RUN(test_take_wave_caps_to_ready_prefix);
  RUN(test_ready_wave_waits_for_prefix);
  RUN(test_advance_watermark_broadcasts);
  RUN(test_owner_capacity_saturation_surfaces_error);
  RUN(test_owner_table_recycles_after_pop);
  RUN(test_missing_shard_reaches_ready);
  log_info("all tests passed");
  return 0;
}
