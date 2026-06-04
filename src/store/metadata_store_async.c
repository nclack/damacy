#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "store/metadata_store_async.h"

#include "log/log.h"
#include "platform/platform.h"
#ifdef DAMACY_METADATA_ASYNC_NUMA
#include "numa/numa.h"
#endif

#include <errno.h>
#include <fcntl.h>
#include <liburing.h>
#include <linux/openat2.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/stat.h>
#include <unistd.h>

#define URING_MAX_EPOLL_EVENTS 2

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

#define OP_TAG_MASK 0x7ull

struct metadata_job;

struct metadata_job
{
  struct metadata_job* next;
  struct metadata_store_async* s;
  enum request_kind kind;
  char* key;
  uint64_t offset;
  size_t requested_len;
  metadata_store_read_cb read_cb;
  metadata_store_stat_cb stat_cb;
  void* user;

  int fd;
  int open_done;
  int stat_done;
  int read_done;
  int close_done;
  int closing;
  int open_res;
  int stat_res;
  int read_res;
  int close_res;
  struct statx stx;
  struct open_how open_how;

  void* data;
  size_t len;
  enum damacy_status status;
};

_Static_assert(_Alignof(struct metadata_job) >= 8,
               "metadata_job pointers must leave low bits for op tags");

struct metadata_store_async
{
  struct io_uring ring;
  pthread_t worker;
  int worker_started;
  int ring_initialized;
  int lock_initialized;
  int rng_lock_initialized;
  int event_fd;
  int epoll_fd;
  uint32_t concurrency;
  uint32_t active;

  pthread_mutex_t lock;
  struct metadata_job* pending_head;
  struct metadata_job* pending_tail;
  int stopping;

  struct damacy_latency_model latency;
  int latency_enabled;
  uint64_t rng_state;
  pthread_mutex_t rng_lock;
  _Atomic uint64_t latency_ops;
  _Atomic uint64_t latency_map_ops;
  _Atomic uint64_t latency_stat_ops;
  _Atomic uint64_t latency_submit_ops;
  _Atomic uint64_t latency_submit_dev_ops;
  _Atomic uint64_t latency_active;
  _Atomic uint64_t latency_max_active;
  _Atomic uint64_t latency_total_sleep_ns;
  _Atomic uint64_t latency_max_sleep_ns;

  _Atomic uint64_t read_jobs;
  _Atomic uint64_t read_active;
  _Atomic uint64_t read_max_active;

#ifdef DAMACY_METADATA_ASYNC_NUMA
  struct numa_resolved affinity;
#endif
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
  atomic_fetch_add_explicit(&s->latency_ops, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(
    &s->latency_total_sleep_ns, ns, memory_order_relaxed);
  atomic_max_u64(&s->latency_max_sleep_ns, ns);
  switch (kind) {
    case LATENCY_OP_STAT:
      atomic_fetch_add_explicit(&s->latency_stat_ops, 1, memory_order_relaxed);
      break;
    case LATENCY_OP_SUBMIT:
      atomic_fetch_add_explicit(
        &s->latency_submit_ops, 1, memory_order_relaxed);
      break;
  }
  uint64_t active =
    atomic_fetch_add_explicit(&s->latency_active, 1, memory_order_acq_rel) + 1;
  atomic_max_u64(&s->latency_max_active, active);
  if (ns)
    platform_sleep_ns((int64_t)ns);
  atomic_fetch_sub_explicit(&s->latency_active, 1, memory_order_acq_rel);
}

static void
read_active_begin(struct metadata_store_async* s)
{
  atomic_fetch_add_explicit(&s->read_jobs, 1, memory_order_relaxed);
  uint64_t active =
    atomic_fetch_add_explicit(&s->read_active, 1, memory_order_acq_rel) + 1;
  atomic_max_u64(&s->read_max_active, active);
}

static void
read_active_end(struct metadata_store_async* s)
{
  atomic_fetch_sub_explicit(&s->read_active, 1, memory_order_acq_rel);
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
  if (!job)
    return;
  free(job->key);
  free(job);
}

static struct io_uring_sqe*
get_sqe(struct metadata_store_async* s)
{
  struct io_uring_sqe* sqe = io_uring_get_sqe(&s->ring);
  if (sqe)
    return sqe;
  int rc = io_uring_submit(&s->ring);
  if (rc < 0)
    log_warn("metadata_store_async: io_uring_submit failed: %s", strerror(-rc));
  return io_uring_get_sqe(&s->ring);
}

static int
ensure_sq_space(struct metadata_store_async* s, unsigned n)
{
  if (io_uring_sq_space_left(&s->ring) >= n)
    return 0;
  int rc = io_uring_submit(&s->ring);
  if (rc < 0) {
    log_warn("metadata_store_async: io_uring_submit failed: %s", strerror(-rc));
    return 1;
  }
  return io_uring_sq_space_left(&s->ring) < n;
}

static void
set_sqe_data(struct io_uring_sqe* sqe,
             struct metadata_job* job,
             enum op_kind kind)
{
  uintptr_t data = (uintptr_t)job | (uintptr_t)kind;
  io_uring_sqe_set_data64(sqe, (unsigned long long)data);
}

static int
submit_op(struct metadata_store_async* s,
          struct metadata_job* job,
          enum op_kind kind,
          struct io_uring_sqe** sqe_out)
{
  struct io_uring_sqe* sqe = get_sqe(s);
  if (!sqe)
    return 1;
  set_sqe_data(sqe, job, kind);
  *sqe_out = sqe;
  return 0;
}

static int
submit_close(struct metadata_store_async* s, struct metadata_job* job);

static void
complete_job(struct metadata_store_async* s, struct metadata_job* job)
{
  if (job->kind == REQ_STAT) {
    enum store_stat_result stat_status = STORE_STAT_ERROR;
    uint64_t size = 0;
    if (job->stat_res >= 0) {
      stat_status = STORE_STAT_OK;
      size = (uint64_t)job->stx.stx_size;
    } else {
      stat_status = stat_status_from_errno(-job->stat_res);
    }
    job->stat_cb(job->user, stat_status, size);
  } else {
    job->read_cb(job->user, job->status, job->data, job->len);
    job->data = NULL;
  }
  if (job->close_res < 0 && job->status == DAMACY_OK) {
    log_warn("metadata_store_async: close failed for %s: %s",
             job->key,
             strerror(-job->close_res));
  }
  s->active--;
  job_free(job);
}

static int
submit_statx(struct metadata_store_async* s, struct metadata_job* job)
{
  struct io_uring_sqe* sqe = NULL;
  if (submit_op(s, job, OP_STATX, &sqe))
    return 1;
  io_uring_prep_statx(sqe, AT_FDCWD, job->key, 0, STATX_SIZE, &job->stx);
  return 0;
}

static int
submit_open(struct metadata_store_async* s, struct metadata_job* job)
{
  struct io_uring_sqe* sqe = NULL;
  if (submit_op(s, job, OP_OPEN, &sqe))
    return 1;
  job->open_how = (struct open_how){
    .flags = O_RDONLY | O_CLOEXEC,
  };
  io_uring_prep_openat2(sqe, AT_FDCWD, job->key, &job->open_how);
  return 0;
}

static int
submit_read(struct metadata_store_async* s, struct metadata_job* job)
{
  struct io_uring_sqe* sqe = NULL;
  if (submit_op(s, job, OP_READ, &sqe))
    return 1;
  read_active_begin(s);
  io_uring_prep_read(sqe, job->fd, job->data, job->len, job->offset);
  return 0;
}

static int
submit_close(struct metadata_store_async* s, struct metadata_job* job)
{
  if (job->fd < 0 || job->closing)
    return 0;
  struct io_uring_sqe* sqe = NULL;
  if (submit_op(s, job, OP_CLOSE, &sqe))
    return 1;
  job->closing = 1;
  io_uring_prep_close(sqe, job->fd);
  return 0;
}

static void
fail_request_before_submit(struct metadata_store_async* s,
                           struct metadata_job* job,
                           enum damacy_status status)
{
  if (job->kind == REQ_STAT)
    job->stat_cb(job->user, STORE_STAT_ERROR, 0);
  else
    job->read_cb(job->user, status, NULL, 0);
  s->active--;
  job_free(job);
}

static void
start_job(struct metadata_store_async* s, struct metadata_job* job)
{
  job->fd = -1;
  job->status = DAMACY_OK;

  if (job->kind == REQ_STAT) {
    sleep_for_sample(s, LATENCY_OP_STAT);
    if (submit_statx(s, job))
      fail_request_before_submit(s, job, DAMACY_OOM);
    return;
  }

  sleep_for_sample(s, LATENCY_OP_SUBMIT);
  if (job->kind == REQ_READ_FILE) {
    if (ensure_sq_space(s, 2)) {
      fail_request_before_submit(s, job, DAMACY_IO);
      return;
    }
    int failed = submit_statx(s, job);
    if (!failed)
      failed = submit_open(s, job);
    if (failed)
      fail_request_before_submit(s, job, DAMACY_OOM);
    return;
  }

  job->len = job->requested_len;
  if (job->len) {
    job->data = malloc(job->len);
    if (!job->data) {
      fail_request_before_submit(s, job, DAMACY_OOM);
      return;
    }
  }
  if (submit_open(s, job))
    fail_request_before_submit(s, job, DAMACY_OOM);
}

static int
drain_pending_locked(struct metadata_store_async* s,
                     struct metadata_job** head,
                     struct metadata_job** tail)
{
  *head = NULL;
  *tail = NULL;
  while (s->pending_head && s->active < s->concurrency) {
    struct metadata_job* job = s->pending_head;
    s->pending_head = job->next;
    if (!s->pending_head)
      s->pending_tail = NULL;
    job->next = NULL;
    if (*tail)
      (*tail)->next = job;
    else
      *head = job;
    *tail = job;
    s->active++;
  }
  return *head != NULL;
}

static int
start_pending_jobs(struct metadata_store_async* s)
{
  struct metadata_job* head = NULL;
  struct metadata_job* tail = NULL;
  (void)tail;
  pthread_mutex_lock(&s->lock);
  int have_jobs = drain_pending_locked(s, &head, &tail);
  pthread_mutex_unlock(&s->lock);

  for (struct metadata_job* job = head; job;) {
    struct metadata_job* next = job->next;
    job->next = NULL;
    start_job(s, job);
    job = next;
  }
  if (have_jobs) {
    int rc = io_uring_submit(&s->ring);
    if (rc < 0)
      log_warn("metadata_store_async: io_uring_submit failed: %s",
               strerror(-rc));
  }
  return have_jobs;
}

static void
handle_statx_complete(struct metadata_store_async* s, struct metadata_job* job)
{
  job->stat_done = 1;
  if (job->kind == REQ_STAT) {
    complete_job(s, job);
    return;
  }

  if (job->stat_res < 0) {
    job->status = damacy_status_from_errno(-job->stat_res);
    if (job->open_done) {
      if (job->fd >= 0) {
        if (submit_close(s, job)) {
          job->close_res = 0;
          complete_job(s, job);
        }
      } else {
        complete_job(s, job);
      }
    }
    return;
  }

  if ((uint64_t)job->stx.stx_size > SIZE_MAX) {
    job->status = DAMACY_BUDGET;
    if (job->open_done) {
      if (job->fd >= 0) {
        if (submit_close(s, job)) {
          job->close_res = 0;
          complete_job(s, job);
        }
      } else {
        complete_job(s, job);
      }
    }
    return;
  }
  job->len = (size_t)job->stx.stx_size;
  if (job->open_done && job->fd >= 0) {
    if (job->len) {
      job->data = malloc(job->len);
      if (!job->data) {
        job->status = DAMACY_OOM;
        if (submit_close(s, job)) {
          job->close_res = 0;
          complete_job(s, job);
        }
        return;
      }
      if (submit_read(s, job)) {
        free(job->data);
        job->data = NULL;
        job->status = DAMACY_OOM;
        if (submit_close(s, job)) {
          job->close_res = 0;
          complete_job(s, job);
        }
      }
    } else if (submit_close(s, job)) {
      job->close_res = 0;
      complete_job(s, job);
    }
  }
}

static void
handle_open_complete(struct metadata_store_async* s, struct metadata_job* job)
{
  job->open_done = 1;
  if (job->open_res < 0) {
    job->status = damacy_status_from_errno(-job->open_res);
    if (job->kind == REQ_READ || job->stat_done)
      complete_job(s, job);
    return;
  }
  job->fd = job->open_res;
  if (job->kind == REQ_READ) {
    if (job->len) {
      if (submit_read(s, job)) {
        job->status = DAMACY_OOM;
        if (submit_close(s, job)) {
          job->close_res = 0;
          complete_job(s, job);
        }
      }
    } else if (submit_close(s, job)) {
      job->close_res = 0;
      complete_job(s, job);
    }
    return;
  }

  if (!job->stat_done)
    return;
  if (job->status != DAMACY_OK) {
    if (submit_close(s, job)) {
      job->close_res = 0;
      complete_job(s, job);
    }
    return;
  }
  if (job->len) {
    job->data = malloc(job->len);
    if (!job->data) {
      job->status = DAMACY_OOM;
      if (submit_close(s, job)) {
        job->close_res = 0;
        complete_job(s, job);
      }
      return;
    }
    if (submit_read(s, job)) {
      free(job->data);
      job->data = NULL;
      job->status = DAMACY_OOM;
      if (submit_close(s, job)) {
        job->close_res = 0;
        complete_job(s, job);
      }
    }
  } else if (submit_close(s, job)) {
    job->close_res = 0;
    complete_job(s, job);
  }
}

static void
handle_read_complete(struct metadata_store_async* s, struct metadata_job* job)
{
  read_active_end(s);
  job->read_done = 1;
  if (job->read_res < 0) {
    free(job->data);
    job->data = NULL;
    job->len = 0;
    job->status = damacy_status_from_errno(-job->read_res);
  } else if ((size_t)job->read_res != job->len) {
    free(job->data);
    job->data = NULL;
    job->len = 0;
    job->status = DAMACY_IO;
  }
  if (submit_close(s, job)) {
    job->close_res = 0;
    complete_job(s, job);
  }
}

static void
handle_close_complete(struct metadata_store_async* s, struct metadata_job* job)
{
  job->close_done = 1;
  job->fd = -1;
  complete_job(s, job);
}

static void
handle_cqe(struct metadata_store_async* s, struct io_uring_cqe* cqe)
{
  uintptr_t data = (uintptr_t)io_uring_cqe_get_data64(cqe);
  if (!data)
    return;
  struct metadata_job* job = (struct metadata_job*)(data & ~OP_TAG_MASK);
  enum op_kind kind = (enum op_kind)(data & OP_TAG_MASK);
  switch (kind) {
    case OP_STATX:
      job->stat_res = cqe->res;
      handle_statx_complete(s, job);
      break;
    case OP_OPEN:
      job->open_res = cqe->res;
      handle_open_complete(s, job);
      break;
    case OP_READ:
      job->read_res = cqe->res;
      handle_read_complete(s, job);
      break;
    case OP_CLOSE:
      job->close_res = cqe->res;
      handle_close_complete(s, job);
      break;
  }
}

static void
drain_event_fd(int fd)
{
  uint64_t val = 0;
  while (read(fd, &val, sizeof(val)) == (ssize_t)sizeof(val)) {
  }
}

static void
wake_driver(struct metadata_store_async* s)
{
  uint64_t one = 1;
  ssize_t n = write(s->event_fd, &one, sizeof(one));
  if (n < 0 && errno != EAGAIN)
    log_warn("metadata_store_async: eventfd wake failed: %s", strerror(errno));
}

static int
should_stop(struct metadata_store_async* s)
{
  pthread_mutex_lock(&s->lock);
  int stop = s->stopping && !s->pending_head && s->active == 0;
  pthread_mutex_unlock(&s->lock);
  return stop;
}

static void*
worker_main(void* arg)
{
  struct metadata_store_async* s = (struct metadata_store_async*)arg;
#ifdef DAMACY_METADATA_ASYNC_NUMA
  numa_apply_thread_affinity(&s->affinity, "metadata_store_uring");
#endif

  for (;;) {
    start_pending_jobs(s);

    struct io_uring_cqe* cqe = NULL;
    unsigned head = 0;
    unsigned count = 0;
    io_uring_for_each_cqe(&s->ring, head, cqe)
    {
      handle_cqe(s, cqe);
      count++;
    }
    if (count) {
      io_uring_cq_advance(&s->ring, count);
      int rc = io_uring_submit(&s->ring);
      if (rc < 0)
        log_warn("metadata_store_async: io_uring_submit failed: %s",
                 strerror(-rc));
      continue;
    }

    if (should_stop(s))
      break;

    struct epoll_event events[URING_MAX_EPOLL_EVENTS];
    int n = epoll_wait(s->epoll_fd, events, URING_MAX_EPOLL_EVENTS, -1);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      log_error("metadata_store_async: epoll_wait failed: %s", strerror(errno));
      break;
    }
    for (int i = 0; i < n; ++i)
      if (events[i].data.fd == s->event_fd)
        drain_event_fd(s->event_fd);
  }
  return NULL;
}

static int
probe_required_ops(struct io_uring* ring)
{
  static const struct
  {
    int op;
    const char* name;
  } required[] = {
    { IORING_OP_STATX, "IORING_OP_STATX" },
    { IORING_OP_OPENAT2, "IORING_OP_OPENAT2" },
    { IORING_OP_READ, "IORING_OP_READ" },
    { IORING_OP_CLOSE, "IORING_OP_CLOSE" },
  };
  struct io_uring_probe* probe = io_uring_get_probe_ring(ring);
  if (!probe) {
    log_error("metadata_store_async: failed to probe io_uring operations");
    return 1;
  }
  int missing = 0;
  for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i) {
    if (!io_uring_opcode_supported(probe, required[i].op)) {
      log_error("metadata_store_async: required operation unsupported: %s",
                required[i].name);
      missing = 1;
    }
  }
  io_uring_free_probe(probe);
  return missing;
}

static uint32_t
ring_entries_for_concurrency(int concurrency)
{
  uint64_t entries = (uint64_t)concurrency * 4u + 16u;
  if (entries < 32u)
    entries = 32u;
  if (entries > 16384u)
    entries = 16384u;
  return (uint32_t)entries;
}

static void
close_fd_if_valid(int* fd)
{
  if (*fd >= 0) {
    close(*fd);
    *fd = -1;
  }
}

static void
metadata_store_async_free_partial(struct metadata_store_async* s)
{
  if (!s)
    return;
  if (s->worker_started)
    pthread_join(s->worker, NULL);
  if (s->event_fd >= 0)
    close(s->event_fd);
  close_fd_if_valid(&s->epoll_fd);
  if (s->ring_initialized)
    io_uring_queue_exit(&s->ring);
  if (s->lock_initialized)
    pthread_mutex_destroy(&s->lock);
  if (s->rng_lock_initialized)
    pthread_mutex_destroy(&s->rng_lock);
  while (s->pending_head) {
    struct metadata_job* job = s->pending_head;
    s->pending_head = job->next;
    job_free(job);
  }
  free(s);
}

struct metadata_store_async*
metadata_store_async_create(int concurrency,
                            const struct numa_resolved* affinity,
                            const struct damacy_latency_model* latency)
{
  if (concurrency <= 0)
    return NULL;
  if (latency && (!isfinite(latency->lognormal_mu_ln_ns) ||
                  !isfinite(latency->lognormal_sigma_ln_ns) ||
                  latency->lognormal_sigma_ln_ns < 0.0))
    return NULL;

  struct metadata_store_async* s =
    (struct metadata_store_async*)calloc(1, sizeof(*s));
  if (!s)
    return NULL;
  s->event_fd = -1;
  s->epoll_fd = -1;
  s->concurrency = (uint32_t)concurrency;
#ifdef DAMACY_METADATA_ASYNC_NUMA
  if (affinity)
    s->affinity = *affinity;
  else
    s->affinity.node = -1;
#else
  (void)affinity;
#endif
  if (latency) {
    s->latency = *latency;
    s->latency_enabled = latency_enabled(latency);
    s->rng_state = latency->seed ? latency->seed : 0xc0ffee1234ULL;
  }

  if (pthread_mutex_init(&s->lock, NULL) != 0)
    goto Fail;
  s->lock_initialized = 1;
  if (pthread_mutex_init(&s->rng_lock, NULL) != 0)
    goto Fail;
  s->rng_lock_initialized = 1;

  uint32_t entries = ring_entries_for_concurrency(concurrency);
  int rc = io_uring_queue_init(entries, &s->ring, 0);
  if (rc < 0) {
    log_error("metadata_store_async: io_uring_queue_init(%u) failed: %s",
              entries,
              strerror(-rc));
    goto Fail;
  }
  s->ring_initialized = 1;
  if (probe_required_ops(&s->ring))
    goto Fail;

  s->event_fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (s->event_fd < 0) {
    log_error("metadata_store_async: eventfd failed: %s", strerror(errno));
    goto Fail;
  }
  s->epoll_fd = epoll_create1(EPOLL_CLOEXEC);
  if (s->epoll_fd < 0) {
    log_error("metadata_store_async: epoll_create1 failed: %s",
              strerror(errno));
    goto Fail;
  }
  struct epoll_event ev = {
    .events = EPOLLIN,
    .data.fd = s->event_fd,
  };
  if (epoll_ctl(s->epoll_fd, EPOLL_CTL_ADD, s->event_fd, &ev) != 0) {
    log_error("metadata_store_async: epoll_ctl(eventfd) failed: %s",
              strerror(errno));
    goto Fail;
  }
  ev = (struct epoll_event){
    .events = EPOLLIN,
    .data.fd = s->ring.ring_fd,
  };
  if (epoll_ctl(s->epoll_fd, EPOLL_CTL_ADD, s->ring.ring_fd, &ev) != 0) {
    log_error("metadata_store_async: epoll_ctl(io_uring) failed: %s",
              strerror(errno));
    goto Fail;
  }

  if (pthread_create(&s->worker, NULL, worker_main, s) != 0) {
    log_error("metadata_store_async: pthread_create failed");
    goto Fail;
  }
  s->worker_started = 1;
  log_info("metadata_store_async: using io_uring metadata path "
           "(request_concurrency=%u ring_entries=%u)",
           s->concurrency,
           entries);
  return s;

Fail:
  metadata_store_async_free_partial(s);
  return NULL;
}

void
metadata_store_async_destroy(struct metadata_store_async* s)
{
  if (!s)
    return;
  pthread_mutex_lock(&s->lock);
  s->stopping = 1;
  pthread_mutex_unlock(&s->lock);
  wake_driver(s);
  if (s->worker_started)
    pthread_join(s->worker, NULL);
  s->worker_started = 0;
  metadata_store_async_free_partial(s);
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
  pthread_mutex_unlock(&s->lock);

  wake_driver(s);
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
  struct metadata_job* job = (struct metadata_job*)calloc(1, sizeof(*job));
  if (!job)
    return 1;
  job->key = strdup(key);
  if (!job->key) {
    job_free(job);
    return 1;
  }
  job->s = s;
  job->kind = kind;
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
  struct metadata_job* job = (struct metadata_job*)calloc(1, sizeof(*job));
  if (!job)
    return 1;
  job->key = strdup(key);
  if (!job->key) {
    job_free(job);
    return 1;
  }
  job->s = s;
  job->kind = REQ_STAT;
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
    .ops = atomic_load_explicit(&s->latency_ops, memory_order_relaxed),
    .map_ops = atomic_load_explicit(&s->latency_map_ops, memory_order_relaxed),
    .stat_ops =
      atomic_load_explicit(&s->latency_stat_ops, memory_order_relaxed),
    .submit_ops =
      atomic_load_explicit(&s->latency_submit_ops, memory_order_relaxed),
    .submit_dev_ops =
      atomic_load_explicit(&s->latency_submit_dev_ops, memory_order_relaxed),
    .active = atomic_load_explicit(&s->latency_active, memory_order_relaxed),
    .max_active =
      atomic_load_explicit(&s->latency_max_active, memory_order_relaxed),
    .total_sleep_ns =
      atomic_load_explicit(&s->latency_total_sleep_ns, memory_order_relaxed),
    .max_sleep_ns =
      atomic_load_explicit(&s->latency_max_sleep_ns, memory_order_relaxed),
  };
}

void
metadata_store_async_latency_stats_reset(struct metadata_store_async* s)
{
  if (!s)
    return;
  uint64_t active =
    atomic_load_explicit(&s->latency_active, memory_order_relaxed);
  atomic_store_explicit(&s->latency_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->latency_map_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->latency_stat_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->latency_submit_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->latency_submit_dev_ops, 0, memory_order_relaxed);
  atomic_store_explicit(&s->latency_max_active, active, memory_order_relaxed);
  atomic_store_explicit(&s->latency_total_sleep_ns, 0, memory_order_relaxed);
  atomic_store_explicit(&s->latency_max_sleep_ns, 0, memory_order_relaxed);
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
    .read_jobs = atomic_load_explicit(&s->read_jobs, memory_order_relaxed),
    .read_active = atomic_load_explicit(&s->read_active, memory_order_relaxed),
    .read_max_active =
      atomic_load_explicit(&s->read_max_active, memory_order_relaxed),
  };
}

void
metadata_store_async_backend_stats_reset(struct metadata_store_async* s)
{
  if (!s)
    return;
  uint64_t active = atomic_load_explicit(&s->read_active, memory_order_relaxed);
  atomic_store_explicit(&s->read_jobs, 0, memory_order_relaxed);
  atomic_store_explicit(&s->read_max_active, active, memory_order_relaxed);
}
