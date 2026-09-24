// macOS metadata backend: bounded POSIX workers behind the existing async API.
// Bulk data still uses the unchanged store_fs/io_queue path.
#include "store/metadata_store_async.h"
#include "log/log.h"
#include "platform/platform.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum request_kind
{
  REQ_READ_FILE,
  REQ_READ,
  REQ_STAT,
};

enum op_kind
{
  OP_STATX,
  OP_OPEN,
  OP_READ,
  OP_CLOSE,
};

enum latency_op_kind
{
  LATENCY_OP_STAT,
  LATENCY_OP_SUBMIT,
};

struct metadata_metrics
{
  struct
  {
    _Atomic uint64_t ops;
    _Atomic uint64_t stat_ops;
    _Atomic uint64_t submit_ops;
    _Atomic uint64_t active;
    _Atomic uint64_t max_active;
    _Atomic uint64_t total_sleep_ns;
    _Atomic uint64_t max_sleep_ns;
  } injector;

  struct
  {
    _Atomic uint64_t jobs;
    _Atomic uint64_t active;
    _Atomic uint64_t max_active;
  } read;

  struct
  {
    _Atomic uint64_t count[METADATA_OP_LATENCY_NKINDS];
    _Atomic uint64_t sum_ns[METADATA_OP_LATENCY_NKINDS];
    _Atomic uint64_t max_ns[METADATA_OP_LATENCY_NKINDS];
    _Atomic uint64_t buckets[METADATA_OP_LATENCY_NKINDS]
                            [METADATA_OP_LATENCY_NBUCKETS];
  } op_latency;
};

struct metadata_job
{
  struct metadata_job* next;
  enum request_kind kind;
  char* key;
  uint64_t offset;
  size_t requested_len;
  metadata_store_read_cb read_cb;
  metadata_store_stat_cb stat_cb;
  void* user;
};
struct metadata_store_async
{
  pthread_t* workers;
  int started;
  pthread_mutex_t lock;
  pthread_cond_t wake;
  struct metadata_job *pending_head, *pending_tail;
  int stopping;
  struct damacy_latency_model latency;
  int latency_enabled;
  uint64_t rng_state;
  pthread_mutex_t rng_lock;
  struct metadata_metrics metrics;
};

static void
atomic_max_u64(_Atomic uint64_t* dst, uint64_t val)
{
  uint64_t cur = atomic_load_explicit(dst, memory_order_relaxed);
  while (cur < val &&
         !atomic_compare_exchange_weak_explicit(
           dst, &cur, val, memory_order_relaxed, memory_order_relaxed)) {
  }
}

static uint64_t
monotonic_ns(void)
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

static void
record_op_latency(struct metadata_store_async* s,
                  enum op_kind kind,
                  uint64_t submit_ts)
{
  if (!submit_ts)
    return;
  uint64_t now = monotonic_ns();
  uint64_t elapsed = now > submit_ts ? now - submit_ts : 0;
  atomic_fetch_add_explicit(
    &s->metrics.op_latency.count[kind], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(
    &s->metrics.op_latency.sum_ns[kind], elapsed, memory_order_relaxed);
  atomic_max_u64(&s->metrics.op_latency.max_ns[kind], elapsed);
  atomic_fetch_add_explicit(
    &s->metrics.op_latency.buckets[kind][latency_bucket(elapsed)],
    1,
    memory_order_relaxed);
}

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
sample_delay_ns(struct metadata_store_async* s)
{
  const struct damacy_latency_model* l = &s->latency;
  uint64_t ns = l->baseline_ns;
  if (l->lognormal_mu_ln_ns == 0.0 && l->lognormal_sigma_ln_ns == 0.0)
    return ns;

  pthread_mutex_lock(&s->rng_lock);
  double z = normal01(&s->rng_state);
  pthread_mutex_unlock(&s->rng_lock);

  double tail = exp(l->lognormal_mu_ln_ns + l->lognormal_sigma_ln_ns * z);
  uint64_t tail_ns = tail > 0.0 ? (uint64_t)tail : 0;
  if (l->cap_ns && tail_ns > l->cap_ns)
    tail_ns = l->cap_ns;
  if (UINT64_MAX - ns < tail_ns)
    return UINT64_MAX;
  return ns + tail_ns;
}

static void
sleep_for_sample(struct metadata_store_async* s, enum latency_op_kind kind)
{
  if (!s->latency_enabled)
    return;
  uint64_t ns = sample_delay_ns(s);
  if (ns > INT64_MAX)
    ns = INT64_MAX;
  atomic_fetch_add_explicit(&s->metrics.injector.ops, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(
    &s->metrics.injector.total_sleep_ns, ns, memory_order_relaxed);
  atomic_max_u64(&s->metrics.injector.max_sleep_ns, ns);
  switch (kind) {
    case LATENCY_OP_STAT:
      atomic_fetch_add_explicit(
        &s->metrics.injector.stat_ops, 1, memory_order_relaxed);
      break;
    case LATENCY_OP_SUBMIT:
      atomic_fetch_add_explicit(
        &s->metrics.injector.submit_ops, 1, memory_order_relaxed);
      break;
  }
  uint64_t active = atomic_fetch_add_explicit(
                      &s->metrics.injector.active, 1, memory_order_relaxed) +
                    1;
  atomic_max_u64(&s->metrics.injector.max_active, active);
  if (ns)
    platform_sleep_ns((int64_t)ns);
  atomic_fetch_sub_explicit(
    &s->metrics.injector.active, 1, memory_order_relaxed);
}

static void
read_active_begin(struct metadata_store_async* s)
{
  atomic_fetch_add_explicit(&s->metrics.read.jobs, 1, memory_order_relaxed);
  uint64_t active = atomic_fetch_add_explicit(
                      &s->metrics.read.active, 1, memory_order_relaxed) +
                    1;
  atomic_max_u64(&s->metrics.read.max_active, active);
}

static void
read_active_end(struct metadata_store_async* s)
{
  atomic_fetch_sub_explicit(&s->metrics.read.active, 1, memory_order_relaxed);
}

static int
status_not_found_errno(int err)
{
  return err == ENOENT || err == ENOTDIR;
}

static enum damacy_status
damacy_status_from_errno(int err)
{
  return status_not_found_errno(err) ? DAMACY_NOTFOUND : DAMACY_IO;
}

static enum store_stat_result
stat_status_from_errno(int err)
{
  return status_not_found_errno(err) ? STORE_STAT_NOT_FOUND : STORE_STAT_ERROR;
}

static void
job_free(struct metadata_job* job)
{
  free(job->key);
  free(job);
}

static struct metadata_job*
job_new(const char* key, enum request_kind kind)
{
  struct metadata_job* job = calloc(1, sizeof(*job));
  if (!job)
    return NULL;
  job->key = strdup(key);
  if (!job->key) {
    free(job);
    return NULL;
  }
  job->kind = kind;
  return job;
}

static void
process_job(struct metadata_store_async* s, struct metadata_job* job)
{
  sleep_for_sample(s,
                   job->kind == REQ_STAT ? LATENCY_OP_STAT : LATENCY_OP_SUBMIT);
  if (job->kind == REQ_STAT) {
    struct stat st;
    uint64_t started = monotonic_ns();
    int rc = stat(job->key, &st), error = errno;
    record_op_latency(s, OP_STATX, started);
    job->stat_cb(job->user,
                 rc ? stat_status_from_errno(error) : STORE_STAT_OK,
                 rc ? 0 : (uint64_t)st.st_size);
    return;
  }
  enum damacy_status status = DAMACY_OK;
  void* data = NULL;
  size_t len = job->requested_len;
  uint64_t started = monotonic_ns();
  int fd = open(job->key, O_RDONLY | O_CLOEXEC), error = errno;
  record_op_latency(s, OP_OPEN, started);
  if (fd < 0) {
    status = damacy_status_from_errno(error);
    goto Complete;
  }
  if (job->kind == REQ_READ_FILE) {
    struct stat st;
    started = monotonic_ns();
    int rc = fstat(fd, &st);
    record_op_latency(s, OP_STATX, started);
    if (rc || st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX) {
      status = DAMACY_IO;
      goto Close;
    }
    len = (size_t)st.st_size;
  }
  if (!len)
    goto Close;
  if (job->offset > INT64_MAX || len > (uint64_t)INT64_MAX - job->offset) {
    status = DAMACY_IO;
    goto Close;
  }
  data = malloc(len);
  if (!data) {
    status = DAMACY_OOM;
    goto Close;
  }
  read_active_begin(s);
  started = monotonic_ns();
  size_t done = 0;
  while (done < len) {
    ssize_t n =
      pread(fd, (char*)data + done, len - done, (off_t)(job->offset + done));
    if (n < 0 && errno == EINTR)
      continue;
    if (n <= 0) {
      status = DAMACY_IO;
      break;
    }
    done += (size_t)n;
  }
  record_op_latency(s, OP_READ, started);
  read_active_end(s);
Close:
  started = monotonic_ns();
  if (close(fd) && status == DAMACY_OK)
    log_warn("metadata_store_async: close failed for %s: %s",
             job->key,
             strerror(errno));
  record_op_latency(s, OP_CLOSE, started);
Complete:
  if (status != DAMACY_OK) {
    free(data);
    data = NULL;
    len = 0;
  }
  job->read_cb(job->user, status, data, len);
}

static void*
worker_main(void* arg)
{
  struct metadata_store_async* s = arg;
  for (;;) {
    pthread_mutex_lock(&s->lock);
    while (!s->pending_head && !s->stopping)
      pthread_cond_wait(&s->wake, &s->lock);
    struct metadata_job* job = s->pending_head;
    if (!job) {
      pthread_mutex_unlock(&s->lock);
      break;
    }
    s->pending_head = job->next;
    if (!s->pending_head)
      s->pending_tail = NULL;
    pthread_mutex_unlock(&s->lock);
    process_job(s, job);
    job_free(job);
  }
  return NULL;
}

void
metadata_store_async_destroy(struct metadata_store_async* s)
{
  if (!s)
    return;
  pthread_mutex_lock(&s->lock);
  s->stopping = 1;
  pthread_cond_broadcast(&s->wake);
  pthread_mutex_unlock(&s->lock);
  for (int i = 0; i < s->started; ++i)
    pthread_join(s->workers[i], NULL);
  pthread_mutex_destroy(&s->rng_lock);
  pthread_cond_destroy(&s->wake);
  pthread_mutex_destroy(&s->lock);
  free(s->workers);
  free(s);
}

struct metadata_store_async*
metadata_store_async_create(int concurrency,
                            const struct numa_resolved* affinity,
                            const struct damacy_latency_model* latency)
{
  (void)affinity;
  if (concurrency < 1 ||
      (unsigned)concurrency > DAMACY_MAX_METADATA_IO_CONCURRENCY ||
      (latency && (!isfinite(latency->lognormal_mu_ln_ns) ||
                   !isfinite(latency->lognormal_sigma_ln_ns) ||
                   latency->lognormal_sigma_ln_ns < 0)))
    return NULL;
  struct metadata_store_async* s = calloc(1, sizeof(*s));
  if (!s)
    return NULL;
  s->workers = calloc((size_t)concurrency, sizeof(*s->workers));
  if (!s->workers) {
    free(s);
    return NULL;
  }
  if (pthread_mutex_init(&s->lock, NULL))
    goto Fail;
  if (pthread_cond_init(&s->wake, NULL)) {
    pthread_mutex_destroy(&s->lock);
    goto Fail;
  }
  if (pthread_mutex_init(&s->rng_lock, NULL)) {
    pthread_cond_destroy(&s->wake);
    pthread_mutex_destroy(&s->lock);
    goto Fail;
  }
  if (latency)
    s->latency = *latency;
  s->latency_enabled = latency_enabled(latency);
  s->rng_state = s->latency.seed ? s->latency.seed : 0xc0ffee1234ULL;
  for (int i = 0; i < concurrency; ++i) {
    if (pthread_create(&s->workers[i], NULL, worker_main, s)) {
      metadata_store_async_destroy(s);
      return NULL;
    }
    ++s->started;
  }
  return s;
Fail:
  free(s->workers);
  free(s);
  return NULL;
}

static int
enqueue_job(struct metadata_store_async* s, struct metadata_job* job)
{
  pthread_mutex_lock(&s->lock);
  if (s->stopping) {
    pthread_mutex_unlock(&s->lock);
    return 1;
  }
  if (s->pending_tail)
    s->pending_tail->next = job;
  else
    s->pending_head = job;
  s->pending_tail = job;
  pthread_cond_signal(&s->wake);
  pthread_mutex_unlock(&s->lock);
  return 0;
}

static int
post_read(struct metadata_store_async* s,
          const char* key,
          uint64_t offset,
          size_t len,
          enum request_kind kind,
          metadata_store_read_cb cb,
          void* user)
{
  if (!s || !key || !cb)
    return 1;
  struct metadata_job* job = job_new(key, kind);
  if (!job)
    return 1;
  job->offset = offset;
  job->requested_len = len;
  job->read_cb = cb;
  job->user = user;
  if (enqueue_job(s, job)) {
    job_free(job);
    return 1;
  }
  return 0;
}

int
metadata_store_async_read_file(struct metadata_store_async* s,
                               const char* key,
                               metadata_store_read_cb cb,
                               void* user)
{
  return post_read(s, key, 0, 0, REQ_READ_FILE, cb, user);
}

int
metadata_store_async_read(struct metadata_store_async* s,
                          const char* key,
                          uint64_t offset,
                          size_t len,
                          metadata_store_read_cb cb,
                          void* user)
{
  return post_read(s, key, offset, len, REQ_READ, cb, user);
}

int
metadata_store_async_stat(struct metadata_store_async* s,
                          const char* key,
                          metadata_store_stat_cb cb,
                          void* user)
{
  if (!s || !key || !cb)
    return 1;
  struct metadata_job* job = job_new(key, REQ_STAT);
  if (!job)
    return 1;
  job->stat_cb = cb;
  job->user = user;
  if (enqueue_job(s, job)) {
    job_free(job);
    return 1;
  }
  return 0;
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
  *out = (struct metadata_store_async_latency_stats){
    .ops = atomic_load_explicit(&s->metrics.injector.ops, memory_order_relaxed),
    .stat_ops =
      atomic_load_explicit(&s->metrics.injector.stat_ops, memory_order_relaxed),
    .submit_ops = atomic_load_explicit(&s->metrics.injector.submit_ops,
                                       memory_order_relaxed),
    .active =
      atomic_load_explicit(&s->metrics.injector.active, memory_order_relaxed),
    .max_active = atomic_load_explicit(&s->metrics.injector.max_active,
                                       memory_order_relaxed),
    .total_sleep_ns = atomic_load_explicit(&s->metrics.injector.total_sleep_ns,
                                           memory_order_relaxed),
    .max_sleep_ns = atomic_load_explicit(&s->metrics.injector.max_sleep_ns,
                                         memory_order_relaxed),
  };
}

void
metadata_store_async_latency_stats_reset(struct metadata_store_async* s)
{
  if (!s)
    return;
  uint64_t active =
    atomic_load_explicit(&s->metrics.injector.active, memory_order_relaxed);
  atomic_store_explicit(&s->metrics.injector.ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->metrics.injector.stat_ops, 0, memory_order_relaxed);
  atomic_store_explicit(
    &s->metrics.injector.submit_ops, 0, memory_order_relaxed);
  atomic_store_explicit(
    &s->metrics.injector.max_active, active, memory_order_relaxed);
  atomic_store_explicit(
    &s->metrics.injector.total_sleep_ns, 0, memory_order_relaxed);
  atomic_store_explicit(
    &s->metrics.injector.max_sleep_ns, 0, memory_order_relaxed);
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
  *out = (struct metadata_store_async_backend_stats){
    .read_jobs =
      atomic_load_explicit(&s->metrics.read.jobs, memory_order_relaxed),
    .read_active =
      atomic_load_explicit(&s->metrics.read.active, memory_order_relaxed),
    .read_max_active =
      atomic_load_explicit(&s->metrics.read.max_active, memory_order_relaxed),
  };
}

void
metadata_store_async_backend_stats_reset(struct metadata_store_async* s)
{
  if (!s)
    return;
  uint64_t active =
    atomic_load_explicit(&s->metrics.read.active, memory_order_relaxed);
  atomic_store_explicit(&s->metrics.read.jobs, 0, memory_order_relaxed);
  atomic_store_explicit(
    &s->metrics.read.max_active, active, memory_order_relaxed);
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
  for (unsigned k = 0; k < METADATA_OP_LATENCY_NKINDS; ++k) {
    out->kinds[k].count = atomic_load_explicit(&s->metrics.op_latency.count[k],
                                               memory_order_relaxed);
    out->kinds[k].sum_ns = atomic_load_explicit(
      &s->metrics.op_latency.sum_ns[k], memory_order_relaxed);
    out->kinds[k].max_ns = atomic_load_explicit(
      &s->metrics.op_latency.max_ns[k], memory_order_relaxed);
    for (unsigned b = 0; b < METADATA_OP_LATENCY_NBUCKETS; ++b)
      out->kinds[k].buckets[b] = atomic_load_explicit(
        &s->metrics.op_latency.buckets[k][b], memory_order_relaxed);
  }
}

void
metadata_store_async_op_latency_stats_reset(struct metadata_store_async* s)
{
  if (!s)
    return;
  for (unsigned k = 0; k < METADATA_OP_LATENCY_NKINDS; ++k) {
    atomic_store_explicit(
      &s->metrics.op_latency.count[k], 0, memory_order_relaxed);
    atomic_store_explicit(
      &s->metrics.op_latency.sum_ns[k], 0, memory_order_relaxed);
    atomic_store_explicit(
      &s->metrics.op_latency.max_ns[k], 0, memory_order_relaxed);
    for (unsigned b = 0; b < METADATA_OP_LATENCY_NBUCKETS; ++b)
      atomic_store_explicit(
        &s->metrics.op_latency.buckets[k][b], 0, memory_order_relaxed);
  }
}
