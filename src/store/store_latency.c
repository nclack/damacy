#include "store/store_latency.h"

#include "platform/platform.h"
#include "store/store.h"
#include "store/store_internal.h"

#include <math.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdlib.h>

struct store_latency
{
  struct store base;
  struct store* inner;
  struct damacy_latency_model latency;
  uint64_t rng_state;
  struct platform_mutex* rng_lock;
  _Atomic uint64_t ops;
  _Atomic uint64_t map_ops;
  _Atomic uint64_t stat_ops;
  _Atomic uint64_t submit_ops;
  _Atomic uint64_t submit_dev_ops;
  _Atomic uint64_t active;
  _Atomic uint64_t max_active;
  _Atomic uint64_t total_sleep_ns;
  _Atomic uint64_t max_sleep_ns;
};

enum latency_op_kind
{
  LATENCY_OP_MAP,
  LATENCY_OP_STAT,
  LATENCY_OP_SUBMIT,
  LATENCY_OP_SUBMIT_DEV,
};

static int
latency_enabled(const struct damacy_latency_model* l)
{
  return l && (l->baseline_ns || l->lognormal_mu_ln_ns != 0.0 ||
               l->lognormal_sigma_ln_ns != 0.0);
}

static uint32_t
pcg32(uint64_t* state)
{
  uint64_t oldstate = *state;
  *state = oldstate * 6364136223846793005ULL + 1442695040888963407ULL;
  uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
  uint32_t rot = (uint32_t)(oldstate >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

static double
uniform01(uint64_t* state)
{
  uint32_t u = pcg32(state);
  return ((double)u + 1.0) / 4294967297.0;
}

static double
normal01(uint64_t* state)
{
  double u1 = uniform01(state);
  double u2 = uniform01(state);
  return sqrt(-2.0 * log(u1)) * cos(6.2831853071795864769 * u2);
}

static uint64_t
sample_delay_ns(struct store_latency* s)
{
  const struct damacy_latency_model* l = &s->latency;
  uint64_t ns = l->baseline_ns;
  if (l->lognormal_mu_ln_ns == 0.0 && l->lognormal_sigma_ln_ns == 0.0)
    return ns;

  platform_mutex_lock(s->rng_lock);
  double z = normal01(&s->rng_state);
  platform_mutex_unlock(s->rng_lock);

  double tail = exp(l->lognormal_mu_ln_ns + l->lognormal_sigma_ln_ns * z);
  uint64_t tail_ns = tail > 0.0 ? (uint64_t)tail : 0;
  if (l->cap_ns && tail_ns > l->cap_ns)
    tail_ns = l->cap_ns;
  if (UINT64_MAX - ns < tail_ns)
    return UINT64_MAX;
  return ns + tail_ns;
}

static void
atomic_max_u64(_Atomic uint64_t* dst, uint64_t val)
{
  uint64_t cur = atomic_load_explicit(dst, memory_order_relaxed);
  while (cur < val &&
         !atomic_compare_exchange_weak_explicit(dst,
                                                &cur,
                                                val,
                                                memory_order_relaxed,
                                                memory_order_relaxed)) {
  }
}

static void
sleep_for_sample(struct store_latency* s, enum latency_op_kind kind)
{
  uint64_t ns = sample_delay_ns(s);
  if (ns > INT64_MAX)
    ns = INT64_MAX;
  atomic_fetch_add_explicit(&s->ops, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&s->total_sleep_ns, ns, memory_order_relaxed);
  atomic_max_u64(&s->max_sleep_ns, ns);
  switch (kind) {
    case LATENCY_OP_MAP:
      atomic_fetch_add_explicit(&s->map_ops, 1, memory_order_relaxed);
      break;
    case LATENCY_OP_STAT:
      atomic_fetch_add_explicit(&s->stat_ops, 1, memory_order_relaxed);
      break;
    case LATENCY_OP_SUBMIT:
      atomic_fetch_add_explicit(&s->submit_ops, 1, memory_order_relaxed);
      break;
    case LATENCY_OP_SUBMIT_DEV:
      atomic_fetch_add_explicit(&s->submit_dev_ops, 1, memory_order_relaxed);
      break;
  }
  uint64_t active =
    atomic_fetch_add_explicit(&s->active, 1, memory_order_acq_rel) + 1;
  atomic_max_u64(&s->max_active, active);
  if (ns)
    platform_sleep_ns((int64_t)ns);
  atomic_fetch_sub_explicit(&s->active, 1, memory_order_acq_rel);
}

static void
latency_destroy(struct store* base)
{
  struct store_latency* s = (struct store_latency*)base;
  if (!s)
    return;
  platform_mutex_free(s->rng_lock);
  free(s);
}

static enum store_stat_result
latency_stat(struct store* base, const char* key, uint64_t* out)
{
  struct store_latency* s = (struct store_latency*)base;
  sleep_for_sample(s, LATENCY_OP_STAT);
  return store_stat(s->inner, key, out);
}

static struct store_submit_result
latency_submit(struct store* base, const struct store_read* reads, size_t n)
{
  struct store_latency* s = (struct store_latency*)base;
  sleep_for_sample(s, LATENCY_OP_SUBMIT);
  return store_read_submit(s->inner, reads, n);
}

static struct store_submit_result
latency_submit_dev(struct store* base, const struct store_read* reads, size_t n)
{
  struct store_latency* s = (struct store_latency*)base;
  sleep_for_sample(s, LATENCY_OP_SUBMIT_DEV);
  return store_read_submit_dev(s->inner, reads, n);
}

static enum damacy_status
latency_event_wait(struct store* base, struct store_event ev)
{
  struct store_latency* s = (struct store_latency*)base;
  return store_event_wait(s->inner, ev);
}

static struct store_event_poll
latency_event_query(struct store* base, struct store_event ev)
{
  struct store_latency* s = (struct store_latency*)base;
  return store_event_query(s->inner, ev);
}

static void
latency_event_discard(struct store* base, struct store_event ev)
{
  struct store_latency* s = (struct store_latency*)base;
  store_event_discard(s->inner, ev);
}

static int
latency_map(struct store* base, const char* key, struct store_view* out)
{
  struct store_latency* s = (struct store_latency*)base;
  sleep_for_sample(s, LATENCY_OP_MAP);
  return store_map(s->inner, key, out);
}

static void
latency_unmap(struct store* base, struct store_view* view)
{
  struct store_latency* s = (struct store_latency*)base;
  store_unmap(s->inner, view);
}

static const struct store_vtable latency_vtable = {
  .destroy = latency_destroy,
  .stat = latency_stat,
  .submit = latency_submit,
  .submit_dev = latency_submit_dev,
  .event_wait = latency_event_wait,
  .event_query = latency_event_query,
  .event_discard = latency_event_discard,
  .map = latency_map,
  .unmap = latency_unmap,
};

static const struct store_vtable latency_vtable_no_gds = {
  .destroy = latency_destroy,
  .stat = latency_stat,
  .submit = latency_submit,
  .submit_dev = NULL,
  .event_wait = latency_event_wait,
  .event_query = latency_event_query,
  .event_discard = latency_event_discard,
  .map = latency_map,
  .unmap = latency_unmap,
};

struct store*
store_latency_create(struct store* inner,
                     const struct damacy_latency_model* latency)
{
  if (!inner || !latency_enabled(latency))
    return NULL;
  if (!isfinite(latency->lognormal_mu_ln_ns) ||
      !isfinite(latency->lognormal_sigma_ln_ns) ||
      latency->lognormal_sigma_ln_ns < 0.0)
    return NULL;

  struct store_latency* s = (struct store_latency*)calloc(1, sizeof(*s));
  if (!s)
    return NULL;
  s->base.vt =
    store_supports_gds(inner) ? &latency_vtable : &latency_vtable_no_gds;
  s->inner = inner;
  s->latency = *latency;
  s->rng_state = latency->seed ? latency->seed : 0xc0ffee1234ULL;
  s->rng_lock = platform_mutex_new();
  if (!s->rng_lock) {
    free(s);
    return NULL;
  }
  return &s->base;
}

void
store_latency_stats_get(struct store* base, struct store_latency_stats* out)
{
  if (!out)
    return;
  *out = (struct store_latency_stats){ 0 };
  if (!base)
    return;
  struct store_latency* s = (struct store_latency*)base;
  *out = (struct store_latency_stats){
    .ops = atomic_load_explicit(&s->ops, memory_order_relaxed),
    .map_ops = atomic_load_explicit(&s->map_ops, memory_order_relaxed),
    .stat_ops = atomic_load_explicit(&s->stat_ops, memory_order_relaxed),
    .submit_ops = atomic_load_explicit(&s->submit_ops, memory_order_relaxed),
    .submit_dev_ops =
      atomic_load_explicit(&s->submit_dev_ops, memory_order_relaxed),
    .active = atomic_load_explicit(&s->active, memory_order_relaxed),
    .max_active = atomic_load_explicit(&s->max_active, memory_order_relaxed),
    .total_sleep_ns =
      atomic_load_explicit(&s->total_sleep_ns, memory_order_relaxed),
    .max_sleep_ns =
      atomic_load_explicit(&s->max_sleep_ns, memory_order_relaxed),
  };
}

void
store_latency_stats_reset(struct store* base)
{
  if (!base)
    return;
  struct store_latency* s = (struct store_latency*)base;
  uint64_t active = atomic_load_explicit(&s->active, memory_order_relaxed);
  atomic_store_explicit(&s->ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->map_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->stat_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->submit_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->submit_dev_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->max_active, active, memory_order_relaxed);
  atomic_store_explicit(&s->total_sleep_ns, 0, memory_order_relaxed);
  atomic_store_explicit(&s->max_sleep_ns, 0, memory_order_relaxed);
}
