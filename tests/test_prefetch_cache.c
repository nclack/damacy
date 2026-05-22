#include "damacy.h"
#include "expect.h"
#include "prefetch/prefetch_cache.h"
#include "util/prelude.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

static int
mock_eq(const struct prefetch_ops* self,
        const void* stored_key,
        const void* probe_key)
{
  (void)self;
  return strcmp((const char*)stored_key, (const char*)probe_key) == 0;
}

static void*
mock_key_clone(const struct prefetch_ops* self, const void* probe_key)
{
  (void)self;
  return strdup((const char*)probe_key);
}

static void
mock_key_destroy(const struct prefetch_ops* self, void* stored_key)
{
  (void)self;
  free(stored_key);
}

static void
mock_destroy(const struct prefetch_ops* self, void* value)
{
  (void)self;
  free(value);
}

static const struct prefetch_ops MOCK_OPS = {
  .key_eq = mock_eq,
  .key_clone = mock_key_clone,
  .key_destroy = mock_key_destroy,
  .value_destroy = mock_destroy,
};

static int
mock_fetch(struct prefetch_fetcher* self,
           const void* key,
           void** out_value,
           int* out_err)
{
  (void)self;
  (void)out_err;
  *out_value = strdup((const char*)key);
  return 0;
}

static struct prefetch_fetcher MOCK_FETCHER = { .fetch = mock_fetch };

struct error_fetcher
{
  struct prefetch_fetcher base;
  int err_code;
};

static int
error_fetch(struct prefetch_fetcher* self_,
            const void* key,
            void** out_value,
            int* out_err)
{
  struct error_fetcher* self = container_of(self_, struct error_fetcher, base);
  (void)key;
  *out_value = NULL;
  *out_err = self->err_code;
  return 1;
}

static int
mock_post(struct prefetch_executor* self,
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

static struct prefetch_executor MOCK_EXECUTOR = { .post = mock_post };

struct delayed_executor
{
  struct prefetch_executor base;
  void (*pending_fn)(void*);
  void* pending_ctx;
  void (*pending_ctx_free)(void*);
};

static int
delayed_post(struct prefetch_executor* self_,
             void (*fn)(void*),
             void* ctx,
             void (*ctx_free)(void*))
{
  struct delayed_executor* self =
    container_of(self_, struct delayed_executor, base);
  self->pending_fn = fn;
  self->pending_ctx = ctx;
  self->pending_ctx_free = ctx_free;
  return 0;
}

static void
delayed_drain(struct delayed_executor* self)
{
  while (self->pending_fn) {
    void (*fn)(void*) = self->pending_fn;
    void* ctx = self->pending_ctx;
    void (*free_)(void*) = self->pending_ctx_free;
    self->pending_fn = NULL;
    self->pending_ctx = NULL;
    self->pending_ctx_free = NULL;
    fn(ctx);
    if (free_)
      free_(ctx);
  }
}

static struct prefetch_cache_config
make_config(void)
{
  return (struct prefetch_cache_config){
    .capacity = 8,
    .max_probe = 16,
    .ops = &MOCK_OPS,
    .fetcher = &MOCK_FETCHER,
    .executor = &MOCK_EXECUTOR,
  };
}

static int
test_create_destroy_basic(void)
{
  struct prefetch_cache_config cfg = make_config();
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);
  prefetch_cache_destroy(c);
  return 0;
}

static int
test_create_rejects_invalid_args(void)
{
  EXPECT(prefetch_cache_create(NULL) == NULL);
  {
    struct prefetch_cache_config cfg = make_config();
    cfg.capacity = 0;
    EXPECT(prefetch_cache_create(&cfg) == NULL);
  }
  {
    struct prefetch_cache_config cfg = make_config();
    cfg.ops = NULL;
    EXPECT(prefetch_cache_create(&cfg) == NULL);
  }
  {
    struct prefetch_cache_config cfg = make_config();
    cfg.fetcher = NULL;
    EXPECT(prefetch_cache_create(&cfg) == NULL);
  }
  {
    struct prefetch_cache_config cfg = make_config();
    cfg.executor = NULL;
    EXPECT(prefetch_cache_create(&cfg) == NULL);
  }
  return 0;
}

static int
test_destroy_null_safe(void)
{
  prefetch_cache_destroy(NULL);
  return 0;
}

static int
test_handle_valid(void)
{
  EXPECT(!prefetch_handle_valid(PREFETCH_HANDLE_NONE));
  EXPECT(prefetch_handle_valid((struct prefetch_handle){ .slot = 1 }));
  EXPECT(prefetch_handle_valid((struct prefetch_handle){ .generation = 1 }));
  return 0;
}

static int
test_gate_init_ready(void)
{
  struct prefetch_gate g;
  prefetch_gate_init(&g);
  EXPECT(prefetch_gate_is_ready(&g));
  EXPECT(!prefetch_gate_has_error(&g));
  EXPECT(prefetch_gate_pending(&g) == 0);
  return 0;
}

static int
test_gate_pending_blocks_ready(void)
{
  struct prefetch_gate g;
  atomic_store_explicit(&g.state, 5, memory_order_release);
  EXPECT(prefetch_gate_pending(&g) == 5);
  EXPECT(!prefetch_gate_is_ready(&g));
  EXPECT(!prefetch_gate_has_error(&g));
  return 0;
}

static int
test_gate_error_independent_of_ready(void)
{
  struct prefetch_gate g;
  atomic_store_explicit(&g.state, UINT64_C(1) << 63, memory_order_release);
  EXPECT(prefetch_gate_has_error(&g));
  EXPECT(prefetch_gate_pending(&g) == 0);
  EXPECT(prefetch_gate_is_ready(&g));
  return 0;
}

static int
test_gate_helpers_null_safe(void)
{
  prefetch_gate_init(NULL);
  EXPECT(prefetch_gate_is_ready(NULL));
  EXPECT(!prefetch_gate_has_error(NULL));
  EXPECT(prefetch_gate_pending(NULL) == 0);
  return 0;
}

static int
test_advance_watermark_monotonic(void)
{
  struct prefetch_cache_config cfg = make_config();
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_cache_stats st;
  prefetch_cache_stats_get(c, &st);
  EXPECT(st.watermark == 0);

  prefetch_cache_advance_watermark(c, 10);
  prefetch_cache_stats_get(c, &st);
  EXPECT(st.watermark == 10);

  prefetch_cache_advance_watermark(c, 5);
  prefetch_cache_stats_get(c, &st);
  EXPECT(st.watermark == 10);

  prefetch_cache_advance_watermark(c, 10);
  prefetch_cache_stats_get(c, &st);
  EXPECT(st.watermark == 10);

  prefetch_cache_advance_watermark(c, 11);
  prefetch_cache_stats_get(c, &st);
  EXPECT(st.watermark == 11);

  prefetch_cache_destroy(c);
  return 0;
}

static uint64_t
khash(const char* s)
{
  uint64_t h = 1469598103934665603ULL;
  while (*s) {
    h ^= (uint8_t)*s++;
    h *= 1099511628211ULL;
  }
  return h;
}

static int
test_request_resolves_via_sync_executor(void)
{
  struct prefetch_cache_config cfg = make_config();
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct prefetch_handle h =
    prefetch_cache_request(c, khash("alpha"), "alpha", 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  const void* v = prefetch_cache_try_get(c, h);
  EXPECT(v);
  EXPECT(strcmp((const char*)v, "alpha") == 0);

  enum prefetch_state st = prefetch_cache_query(c, h, NULL, NULL);
  EXPECT(st == PREFETCH_STATE_READY);

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_request_dedup_returns_same_handle(void)
{
  struct prefetch_cache_config cfg = make_config();
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct prefetch_handle h1 =
    prefetch_cache_request(c, khash("k"), "k", 0, &gate);
  struct prefetch_handle h2 =
    prefetch_cache_request(c, khash("k"), "k", 0, &gate);
  EXPECT(h1.slot == h2.slot);
  EXPECT(h1.generation == h2.generation);

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_pending_observable_with_delayed_executor(void)
{
  struct delayed_executor exec = { .base = { .post = delayed_post } };
  struct prefetch_cache_config cfg = make_config();
  cfg.executor = &exec.base;
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct prefetch_handle h =
    prefetch_cache_request(c, khash("k"), "k", 0, &gate);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_cache_query(c, h, NULL, NULL) == PREFETCH_STATE_PENDING);
  EXPECT(prefetch_gate_pending(&gate) == 1);
  EXPECT(!prefetch_gate_is_ready(&gate));

  delayed_drain(&exec);

  EXPECT(prefetch_cache_query(c, h, NULL, NULL) == PREFETCH_STATE_READY);
  EXPECT(prefetch_gate_is_ready(&gate));
  EXPECT(!prefetch_gate_has_error(&gate));

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_two_gates_decremented_on_completion(void)
{
  struct delayed_executor exec = { .base = { .post = delayed_post } };
  struct prefetch_cache_config cfg = make_config();
  cfg.executor = &exec.base;
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate gA, gB;
  prefetch_gate_init(&gA);
  prefetch_gate_init(&gB);

  struct prefetch_handle hA =
    prefetch_cache_request(c, khash("k"), "k", 0, &gA);
  EXPECT(prefetch_gate_pending(&gA) == 1);

  struct prefetch_handle hB =
    prefetch_cache_request(c, khash("k"), "k", 1, &gB);
  EXPECT(prefetch_gate_pending(&gB) == 1);
  EXPECT(hA.slot == hB.slot);

  delayed_drain(&exec);

  EXPECT(prefetch_gate_pending(&gA) == 0);
  EXPECT(prefetch_gate_pending(&gB) == 0);
  EXPECT(!prefetch_gate_has_error(&gA));
  EXPECT(!prefetch_gate_has_error(&gB));

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_error_propagates_to_gate(void)
{
  struct error_fetcher ef = { .base = { .fetch = error_fetch },
                              .err_code = DAMACY_NOTFOUND };
  struct prefetch_cache_config cfg = make_config();
  cfg.fetcher = &ef.base;
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);
  struct prefetch_handle h =
    prefetch_cache_request(c, khash("bad"), "bad", 0, &gate);
  EXPECT(prefetch_handle_valid(h));

  int err = 0;
  EXPECT(prefetch_cache_query(c, h, NULL, &err) == PREFETCH_STATE_ERROR);
  EXPECT(err == DAMACY_NOTFOUND);
  EXPECT(prefetch_gate_has_error(&gate));
  EXPECT(prefetch_gate_pending(&gate) == 0);

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_sticky_error_hits_propagate(void)
{
  struct error_fetcher ef = { .base = { .fetch = error_fetch },
                              .err_code = DAMACY_DECODE };
  struct prefetch_cache_config cfg = make_config();
  cfg.fetcher = &ef.base;
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate g1;
  prefetch_gate_init(&g1);
  (void)prefetch_cache_request(c, khash("k"), "k", 0, &g1);
  EXPECT(prefetch_gate_has_error(&g1));

  struct prefetch_gate g2;
  prefetch_gate_init(&g2);
  struct prefetch_handle h = prefetch_cache_request(c, khash("k"), "k", 1, &g2);
  EXPECT(prefetch_handle_valid(h));
  EXPECT(prefetch_gate_pending(&g2) == 0);
  EXPECT(prefetch_gate_has_error(&g2));

  int err = 0;
  EXPECT(prefetch_cache_query(c, h, NULL, &err) == PREFETCH_STATE_ERROR);
  EXPECT(err == DAMACY_DECODE);

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_admission_rejection_when_all_pinned(void)
{
  struct prefetch_cache_config cfg = make_config();
  cfg.capacity = 2;
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  struct prefetch_handle h1 =
    prefetch_cache_request(c, khash("a"), "a", 0, &gate);
  struct prefetch_handle h2 =
    prefetch_cache_request(c, khash("b"), "b", 0, &gate);
  EXPECT(prefetch_handle_valid(h1));
  EXPECT(prefetch_handle_valid(h2));

  struct prefetch_handle h3 =
    prefetch_cache_request(c, khash("c"), "c", 0, &gate);
  EXPECT(!prefetch_handle_valid(h3));

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_eviction_after_watermark_advances(void)
{
  struct prefetch_cache_config cfg = make_config();
  cfg.capacity = 2;
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate g0, g1;
  prefetch_gate_init(&g0);
  prefetch_gate_init(&g1);

  (void)prefetch_cache_request(c, khash("a"), "a", 0, &g0);
  (void)prefetch_cache_request(c, khash("b"), "b", 0, &g0);

  struct prefetch_handle h_fail =
    prefetch_cache_request(c, khash("c"), "c", 0, &g0);
  EXPECT(!prefetch_handle_valid(h_fail));

  prefetch_cache_advance_watermark(c, 1);

  struct prefetch_handle h_ok =
    prefetch_cache_request(c, khash("c"), "c", 2, &g1);
  EXPECT(prefetch_handle_valid(h_ok));
  EXPECT(prefetch_gate_is_ready(&g1));

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_max_batch_id_widens_under_hit(void)
{
  struct prefetch_cache_config cfg = make_config();
  cfg.capacity = 2;
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  (void)prefetch_cache_request(c, khash("k"), "k", 5, &gate);
  (void)prefetch_cache_request(c, khash("k"), "k", 7, &gate);
  (void)prefetch_cache_request(c, khash("z"), "z", 5, &gate);

  prefetch_cache_advance_watermark(c, 7);

  struct prefetch_handle h_new =
    prefetch_cache_request(c, khash("new"), "new", 7, &gate);
  EXPECT(prefetch_handle_valid(h_new));

  struct prefetch_handle h_k =
    prefetch_cache_request(c, khash("k"), "k", 8, &gate);
  EXPECT(prefetch_handle_valid(h_k));
  EXPECT(prefetch_cache_query(c, h_k, NULL, NULL) == PREFETCH_STATE_READY);

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_handle_survives_sibling_eviction(void)
{
  struct prefetch_cache_config cfg = make_config();
  cfg.capacity = 2;
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_gate g0, g1;
  prefetch_gate_init(&g0);
  prefetch_gate_init(&g1);

  struct prefetch_handle ha =
    prefetch_cache_request(c, khash("a"), "a", 0, &g0);
  struct prefetch_handle hb =
    prefetch_cache_request(c, khash("b"), "b", 1, &g0);
  EXPECT(prefetch_handle_valid(ha));
  EXPECT(prefetch_handle_valid(hb));

  prefetch_cache_advance_watermark(c, 1);

  struct prefetch_handle hc =
    prefetch_cache_request(c, khash("c"), "c", 1, &g1);
  EXPECT(prefetch_handle_valid(hc));

  const void* vb = prefetch_cache_try_get(c, hb);
  EXPECT(vb);
  EXPECT(strcmp((const char*)vb, "b") == 0);
  EXPECT(prefetch_cache_query(c, hb, NULL, NULL) == PREFETCH_STATE_READY);

  prefetch_cache_destroy(c);
  return 0;
}

static int
test_stats_track_size_and_pending(void)
{
  struct delayed_executor exec = { .base = { .post = delayed_post } };
  struct prefetch_cache_config cfg = make_config();
  cfg.executor = &exec.base;
  struct prefetch_cache* c = prefetch_cache_create(&cfg);
  EXPECT(c);

  struct prefetch_cache_stats st;
  struct prefetch_gate gate;
  prefetch_gate_init(&gate);

  (void)prefetch_cache_request(c, khash("k"), "k", 0, &gate);
  prefetch_cache_stats_get(c, &st);
  EXPECT(st.size == 1);
  EXPECT(st.pending == 1);
  EXPECT(st.errored == 0);

  delayed_drain(&exec);
  prefetch_cache_stats_get(c, &st);
  EXPECT(st.size == 1);
  EXPECT(st.pending == 0);
  EXPECT(st.errored == 0);

  prefetch_cache_destroy(c);
  return 0;
}

int
main(void)
{
  RUN(test_create_destroy_basic);
  RUN(test_create_rejects_invalid_args);
  RUN(test_destroy_null_safe);
  RUN(test_handle_valid);
  RUN(test_gate_init_ready);
  RUN(test_gate_pending_blocks_ready);
  RUN(test_gate_error_independent_of_ready);
  RUN(test_gate_helpers_null_safe);
  RUN(test_advance_watermark_monotonic);
  RUN(test_request_resolves_via_sync_executor);
  RUN(test_request_dedup_returns_same_handle);
  RUN(test_pending_observable_with_delayed_executor);
  RUN(test_two_gates_decremented_on_completion);
  RUN(test_error_propagates_to_gate);
  RUN(test_sticky_error_hits_propagate);
  RUN(test_admission_rejection_when_all_pinned);
  RUN(test_eviction_after_watermark_advances);
  RUN(test_max_batch_id_widens_under_hit);
  RUN(test_handle_survives_sibling_eviction);
  RUN(test_stats_track_size_and_pending);
  log_info("all tests passed");
  return 0;
}
