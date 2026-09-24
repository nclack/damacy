#include "store/metadata_store_async_common.h"

#include "platform/platform.h"

#include <errno.h>
#include <math.h>
#include <time.h>

static void
atomic_max_u64(_Atomic uint64_t* dst, uint64_t val)
{
  uint64_t cur = atomic_load_explicit(dst, memory_order_relaxed);
  while (cur < val &&
         !atomic_compare_exchange_weak_explicit(
           dst, &cur, val, memory_order_relaxed, memory_order_relaxed)) {
  }
}

uint64_t
metadata_monotonic_ns(void)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static unsigned
latency_bucket(uint64_t ns)
{
  if (ns == 0)
    return 0;
  unsigned idx = 63u - (unsigned)__builtin_clzll(ns);
  if (idx >= METADATA_OP_LATENCY_NBUCKETS)
    idx = METADATA_OP_LATENCY_NBUCKETS - 1;
  return idx;
}

void
metadata_record_op_latency(struct metadata_store_common* c,
                           enum op_kind kind,
                           uint64_t submit_ts)
{
  if (!submit_ts)
    return;
  uint64_t now = metadata_monotonic_ns();
  uint64_t elapsed = now > submit_ts ? now - submit_ts : 0;
  atomic_fetch_add_explicit(
    &c->metrics.op_latency.count[kind], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(
    &c->metrics.op_latency.sum_ns[kind], elapsed, memory_order_relaxed);
  atomic_max_u64(&c->metrics.op_latency.max_ns[kind], elapsed);
  atomic_fetch_add_explicit(
    &c->metrics.op_latency.buckets[kind][latency_bucket(elapsed)],
    1,
    memory_order_relaxed);
}

static int
latency_enabled(const struct damacy_latency_model* l)
{
  return l && (l->baseline_ns || l->lognormal_mu_ln_ns != 0.0 ||
               l->lognormal_sigma_ln_ns != 0.0);
}

int
metadata_store_common_init(struct metadata_store_common* c,
                           const struct damacy_latency_model* latency)
{
  if (latency)
    c->latency = *latency;
  c->latency_enabled = latency_enabled(latency);
  c->rng_state = c->latency.seed ? c->latency.seed : 0xc0ffee1234ULL;
  return pthread_mutex_init(&c->rng_lock, NULL) != 0;
}

void
metadata_store_common_destroy(struct metadata_store_common* c)
{
  pthread_mutex_destroy(&c->rng_lock);
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
sample_delay_ns(struct metadata_store_common* c)
{
  const struct damacy_latency_model* l = &c->latency;
  uint64_t ns = l->baseline_ns;
  if (l->lognormal_mu_ln_ns == 0.0 && l->lognormal_sigma_ln_ns == 0.0)
    return ns;

  pthread_mutex_lock(&c->rng_lock);
  double z = normal01(&c->rng_state);
  pthread_mutex_unlock(&c->rng_lock);

  double tail = exp(l->lognormal_mu_ln_ns + l->lognormal_sigma_ln_ns * z);
  uint64_t tail_ns = tail > 0.0 ? (uint64_t)tail : 0;
  if (l->cap_ns && tail_ns > l->cap_ns)
    tail_ns = l->cap_ns;
  if (UINT64_MAX - ns < tail_ns)
    return UINT64_MAX;
  return ns + tail_ns;
}

void
metadata_inject_latency(struct metadata_store_common* c,
                        enum latency_op_kind kind)
{
  if (!c->latency_enabled)
    return;
  uint64_t ns = sample_delay_ns(c);
  if (ns > INT64_MAX)
    ns = INT64_MAX;
  atomic_fetch_add_explicit(&c->metrics.injector.ops, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(
    &c->metrics.injector.total_sleep_ns, ns, memory_order_relaxed);
  atomic_max_u64(&c->metrics.injector.max_sleep_ns, ns);
  switch (kind) {
    case LATENCY_OP_STAT:
      atomic_fetch_add_explicit(
        &c->metrics.injector.stat_ops, 1, memory_order_relaxed);
      break;
    case LATENCY_OP_SUBMIT:
      atomic_fetch_add_explicit(
        &c->metrics.injector.submit_ops, 1, memory_order_relaxed);
      break;
  }
  uint64_t active = atomic_fetch_add_explicit(
                      &c->metrics.injector.active, 1, memory_order_relaxed) +
                    1;
  atomic_max_u64(&c->metrics.injector.max_active, active);
  if (ns)
    platform_sleep_ns((int64_t)ns);
  atomic_fetch_sub_explicit(
    &c->metrics.injector.active, 1, memory_order_relaxed);
}

void
metadata_read_active_begin(struct metadata_store_common* c)
{
  atomic_fetch_add_explicit(&c->metrics.read.jobs, 1, memory_order_relaxed);
  uint64_t active = atomic_fetch_add_explicit(
                      &c->metrics.read.active, 1, memory_order_relaxed) +
                    1;
  atomic_max_u64(&c->metrics.read.max_active, active);
}

void
metadata_read_active_end(struct metadata_store_common* c)
{
  atomic_fetch_sub_explicit(&c->metrics.read.active, 1, memory_order_relaxed);
}

static int
status_not_found_errno(int err)
{
  return err == ENOENT || err == ENOTDIR;
}

enum damacy_status
metadata_status_from_errno(int err)
{
  return status_not_found_errno(err) ? DAMACY_NOTFOUND : DAMACY_IO;
}

enum store_stat_result
metadata_stat_status_from_errno(int err)
{
  return status_not_found_errno(err) ? STORE_STAT_NOT_FOUND : STORE_STAT_ERROR;
}

void
metadata_store_async_latency_stats_get(
  struct metadata_store_async* s,
  struct metadata_store_async_latency_stats* out)
{
  if (!out)
    return;
  *out = (struct metadata_store_async_latency_stats){ 0 };
  if (!s)
    return;
  struct metadata_metrics* m = &metadata_store_async_common(s)->metrics;
  *out = (struct metadata_store_async_latency_stats){
    .ops = atomic_load_explicit(&m->injector.ops, memory_order_relaxed),
    .stat_ops =
      atomic_load_explicit(&m->injector.stat_ops, memory_order_relaxed),
    .submit_ops =
      atomic_load_explicit(&m->injector.submit_ops, memory_order_relaxed),
    .active = atomic_load_explicit(&m->injector.active, memory_order_relaxed),
    .max_active =
      atomic_load_explicit(&m->injector.max_active, memory_order_relaxed),
    .total_sleep_ns =
      atomic_load_explicit(&m->injector.total_sleep_ns, memory_order_relaxed),
    .max_sleep_ns =
      atomic_load_explicit(&m->injector.max_sleep_ns, memory_order_relaxed),
  };
}

void
metadata_store_async_latency_stats_reset(struct metadata_store_async* s)
{
  if (!s)
    return;
  struct metadata_metrics* m = &metadata_store_async_common(s)->metrics;
  uint64_t active =
    atomic_load_explicit(&m->injector.active, memory_order_relaxed);
  atomic_store_explicit(&m->injector.ops, 0, memory_order_relaxed);
  atomic_store_explicit(&m->injector.stat_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&m->injector.submit_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&m->injector.max_active, active, memory_order_relaxed);
  atomic_store_explicit(&m->injector.total_sleep_ns, 0, memory_order_relaxed);
  atomic_store_explicit(&m->injector.max_sleep_ns, 0, memory_order_relaxed);
}

void
metadata_store_async_backend_stats_get(
  struct metadata_store_async* s,
  struct metadata_store_async_backend_stats* out)
{
  if (!out)
    return;
  *out = (struct metadata_store_async_backend_stats){ 0 };
  if (!s)
    return;
  struct metadata_metrics* m = &metadata_store_async_common(s)->metrics;
  *out = (struct metadata_store_async_backend_stats){
    .read_jobs = atomic_load_explicit(&m->read.jobs, memory_order_relaxed),
    .read_active = atomic_load_explicit(&m->read.active, memory_order_relaxed),
    .read_max_active =
      atomic_load_explicit(&m->read.max_active, memory_order_relaxed),
  };
}

void
metadata_store_async_backend_stats_reset(struct metadata_store_async* s)
{
  if (!s)
    return;
  struct metadata_metrics* m = &metadata_store_async_common(s)->metrics;
  uint64_t active = atomic_load_explicit(&m->read.active, memory_order_relaxed);
  atomic_store_explicit(&m->read.jobs, 0, memory_order_relaxed);
  atomic_store_explicit(&m->read.max_active, active, memory_order_relaxed);
}

void
metadata_store_async_op_latency_stats_get(
  struct metadata_store_async* s,
  struct metadata_store_async_op_latency_stats* out)
{
  if (!out)
    return;
  *out = (struct metadata_store_async_op_latency_stats){ 0 };
  if (!s)
    return;
  struct metadata_metrics* m = &metadata_store_async_common(s)->metrics;
  for (unsigned k = 0; k < METADATA_OP_LATENCY_NKINDS; ++k) {
    out->kinds[k].count =
      atomic_load_explicit(&m->op_latency.count[k], memory_order_relaxed);
    out->kinds[k].sum_ns =
      atomic_load_explicit(&m->op_latency.sum_ns[k], memory_order_relaxed);
    out->kinds[k].max_ns =
      atomic_load_explicit(&m->op_latency.max_ns[k], memory_order_relaxed);
    for (unsigned b = 0; b < METADATA_OP_LATENCY_NBUCKETS; ++b)
      out->kinds[k].buckets[b] = atomic_load_explicit(
        &m->op_latency.buckets[k][b], memory_order_relaxed);
  }
}

void
metadata_store_async_op_latency_stats_reset(struct metadata_store_async* s)
{
  if (!s)
    return;
  struct metadata_metrics* m = &metadata_store_async_common(s)->metrics;
  for (unsigned k = 0; k < METADATA_OP_LATENCY_NKINDS; ++k) {
    atomic_store_explicit(&m->op_latency.count[k], 0, memory_order_relaxed);
    atomic_store_explicit(&m->op_latency.sum_ns[k], 0, memory_order_relaxed);
    atomic_store_explicit(&m->op_latency.max_ns[k], 0, memory_order_relaxed);
    for (unsigned b = 0; b < METADATA_OP_LATENCY_NBUCKETS; ++b)
      atomic_store_explicit(
        &m->op_latency.buckets[k][b], 0, memory_order_relaxed);
  }
}
