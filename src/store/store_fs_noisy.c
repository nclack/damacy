#include "store/store_fs_noisy.h"

#include "io_queue/io_queue.h"
#include "log/log.h"
#include "store/store.h"
#include "store/store_fs.h"
#include "util/prelude.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct store_fs_noisy
{
  struct store base;
  struct store* inner; // owned
  struct io_queue* q;

  struct store_fs_noisy_params params;
  double mu;
  double sigma;

  pthread_mutex_t rng_mu;
  uint64_t rng_state;
};

static uint64_t
rng_next_locked(struct store_fs_noisy* n)
{
  uint64_t x = n->rng_state;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  n->rng_state = x;
  return x * 2685821657736338717ull;
}

static double
rng_uniform_locked(struct store_fs_noisy* n)
{
  uint64_t r = rng_next_locked(n) >> 11;
  return (double)r * (1.0 / 9007199254740992.0);
}

static double
rng_normal_locked(struct store_fs_noisy* n)
{
  double u1 = rng_uniform_locked(n);
  double u2 = rng_uniform_locked(n);
  if (u1 < 1e-300)
    u1 = 1e-300;
  return sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2);
}

// Closed form: mean = exp(mu + sigma^2/2) and p99 = exp(mu + z99*sigma).
// Degenerate inputs (p99 <= mean) collapse to zero variance.
static void
fit_lognormal(double mean_ms, double p99_ms, double* out_mu, double* out_sigma)
{
  const double z99 = 2.32634787404084;
  if (mean_ms <= 0.0 || p99_ms <= mean_ms) {
    *out_mu = 0.0;
    *out_sigma = 0.0;
    return;
  }
  double log_ratio = log(p99_ms / mean_ms);
  double disc = z99 * z99 - 2.0 * log_ratio;
  if (disc <= 0.0) {
    *out_mu = log(mean_ms);
    *out_sigma = 0.0;
    return;
  }
  double sigma = z99 - sqrt(disc);
  if (sigma < 0.0)
    sigma = -sigma;
  *out_mu = log(mean_ms) - 0.5 * sigma * sigma;
  *out_sigma = sigma;
}

static double
draw_latency_ms(struct store_fs_noisy* n)
{
  pthread_mutex_lock(&n->rng_mu);
  double z = (n->sigma > 0.0) ? rng_normal_locked(n) : 0.0;
  double jitter = (n->params.jitter_ms > 0.0)
                    ? (rng_uniform_locked(n) * 2.0 - 1.0) * n->params.jitter_ms
                    : 0.0;
  pthread_mutex_unlock(&n->rng_mu);
  double base = (n->sigma > 0.0) ? exp(n->mu + n->sigma * z)
                                 : (n->params.mean_ms > 0.0 ? n->params.mean_ms
                                                            : 0.0);
  double t = base + jitter;
  return t > 0.0 ? t : 0.0;
}

static void
sleep_ms_block(double ms)
{
  if (ms <= 0.0)
    return;
  long sec = (long)(ms / 1e3);
  long nsec = (long)((ms - (double)sec * 1e3) * 1e6);
  struct timespec req = { .tv_sec = sec, .tv_nsec = nsec };
  struct timespec rem;
  while (nanosleep(&req, &rem) == -1 && errno == EINTR)
    req = rem;
}

struct noisy_job
{
  struct store_fs_noisy* n;
  char* key;
  void* dst;
  uint64_t offset;
  size_t len;
};

static void
noisy_job_fn(void* vctx)
{
  struct noisy_job* j = (struct noisy_job*)vctx;
  sleep_ms_block(draw_latency_ms(j->n));
  struct store_read r = { .key = j->key,
                          .dst = j->dst,
                          .offset = j->offset,
                          .len = j->len };
  (void)store_read_many(j->n->inner, &r, 1);
}

static void
noisy_job_free(void* vctx)
{
  struct noisy_job* j = (struct noisy_job*)vctx;
  free(j->key);
  free(j);
}

static struct store_event
noisy_submit(struct store* s, const struct store_read* reads, size_t n)
{
  struct store_fs_noisy* nz = (struct store_fs_noisy*)s;
  struct store_event ev = { 0 };
  if (n == 0) {
    struct io_event ioev = io_queue_record(nz->q);
    ev.seq = ioev.seq;
    return ev;
  }
  for (size_t i = 0; i < n; ++i) {
    struct noisy_job* j = (struct noisy_job*)calloc(1, sizeof(*j));
    CHECK_SILENT(Drain, j);
    j->n = nz;
    j->key = strdup(reads[i].key);
    j->dst = reads[i].dst;
    j->offset = reads[i].offset;
    j->len = reads[i].len;
    if (!j->key) {
      free(j);
      goto Drain;
    }
    if (io_queue_post(nz->q, noisy_job_fn, j, noisy_job_free))
      goto Drain;
  }
  {
    struct io_event ioev = io_queue_record(nz->q);
    ev.seq = ioev.seq;
  }
  return ev;
Drain:
  io_event_wait(nz->q, io_queue_record(nz->q));
  return ev;
}

static void
noisy_event_wait(struct store* s, struct store_event ev)
{
  struct store_fs_noisy* nz = (struct store_fs_noisy*)s;
  io_event_wait(nz->q, (struct io_event){ .seq = ev.seq });
}

static int
noisy_event_query(struct store* s, struct store_event ev)
{
  struct store_fs_noisy* nz = (struct store_fs_noisy*)s;
  return io_event_query(nz->q, (struct io_event){ .seq = ev.seq });
}

static int
noisy_stat(struct store* s, const char* key, uint64_t* out)
{
  struct store_fs_noisy* nz = (struct store_fs_noisy*)s;
  return store_stat(nz->inner, key, out);
}

static int
noisy_map(struct store* s, const char* key, struct store_view* out)
{
  struct store_fs_noisy* nz = (struct store_fs_noisy*)s;
  return store_map(nz->inner, key, out);
}

static void
noisy_unmap(struct store* s, struct store_view* view)
{
  struct store_fs_noisy* nz = (struct store_fs_noisy*)s;
  store_unmap(nz->inner, view);
}

static void
noisy_destroy(struct store* s)
{
  struct store_fs_noisy* nz = (struct store_fs_noisy*)s;
  if (!nz)
    return;
  // Drain noisy jobs before tearing the inner store down, otherwise
  // worker threads may call store_read_many on a freed inner.
  if (nz->q) {
    io_event_wait(nz->q, io_queue_record(nz->q));
    io_queue_destroy(nz->q);
  }
  store_destroy(nz->inner);
  pthread_mutex_destroy(&nz->rng_mu);
  free(nz);
}

static const struct store_vtable noisy_vtable = {
  .destroy = noisy_destroy,
  .stat = noisy_stat,
  .submit = noisy_submit,
  .submit_dev = NULL,
  .event_wait = noisy_event_wait,
  .event_query = noisy_event_query,
  .map = noisy_map,
  .unmap = noisy_unmap,
};

struct store*
store_fs_noisy_create(const struct store_fs_config* cfg,
                      const struct store_fs_noisy_params* np)
{
  struct store_fs_noisy* nz = NULL;
  CHECK_SILENT(Fail, cfg);
  CHECK_SILENT(Fail, np);
  nz = (struct store_fs_noisy*)calloc(1, sizeof(*nz));
  CHECK_SILENT(Fail, nz);
  nz->base.vt = &noisy_vtable;
  pthread_mutex_init(&nz->rng_mu, NULL);
  nz->params = *np;
  fit_lognormal(np->mean_ms, np->p99_ms, &nz->mu, &nz->sigma);
  uint64_t seed = np->seed;
  if (seed == 0) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    seed = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
  }
  nz->rng_state = seed ? seed : 0x9e3779b97f4a7c15ull;
  nz->inner = store_fs_create(cfg);
  CHECK_SILENT(Fail, nz->inner);
  nz->q = io_queue_create(cfg->nthreads, cfg->affinity);
  CHECK_SILENT(Fail, nz->q);

  log_info("store_fs_noisy: mean=%.3fms p99=%.3fms jitter=%.3fms "
           "(mu=%.3f sigma=%.3f) threads=%d",
           np->mean_ms,
           np->p99_ms,
           np->jitter_ms,
           nz->mu,
           nz->sigma,
           cfg->nthreads);
  return &nz->base;
Fail:
  if (nz) {
    if (nz->q)
      io_queue_destroy(nz->q);
    if (nz->inner)
      store_destroy(nz->inner);
    pthread_mutex_destroy(&nz->rng_mu);
    free(nz);
  }
  return NULL;
}

static int
read_env_double(const char* name, double* out)
{
  const char* e = getenv(name);
  if (!e || !*e)
    return 0;
  char* endp = NULL;
  double v = strtod(e, &endp);
  if (endp == e)
    return 0;
  *out = v;
  return 1;
}

int
store_fs_noisy_params_from_env(struct store_fs_noisy_params* out)
{
  if (!out)
    return 0;
  memset(out, 0, sizeof(*out));
  int any = 0;
  any |= read_env_double("DAMACY_NOISY_MEAN_MS", &out->mean_ms);
  any |= read_env_double("DAMACY_NOISY_P99_MS", &out->p99_ms);
  any |= read_env_double("DAMACY_NOISY_JITTER_MS", &out->jitter_ms);
  const char* sd = getenv("DAMACY_NOISY_SEED");
  if (sd && *sd) {
    out->seed = strtoull(sd, NULL, 10);
  }
  return any;
}
