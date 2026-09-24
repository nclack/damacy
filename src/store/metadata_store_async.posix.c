// macOS metadata backend: bounded POSIX workers behind the existing async API.
// Bulk data still uses the unchanged store_fs/io_queue path.
#include "store/metadata_store_async.h"
#include "log/log.h"
#include "store/metadata_store_async_common.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum request_kind
{
  REQ_READ_FILE,
  REQ_READ,
  REQ_STAT,
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
  struct metadata_store_common common;
};

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
  metadata_inject_latency(
    &s->common, job->kind == REQ_STAT ? LATENCY_OP_STAT : LATENCY_OP_SUBMIT);
  if (job->kind == REQ_STAT) {
    struct stat st;
    uint64_t started = metadata_monotonic_ns();
    int rc = stat(job->key, &st), error = errno;
    metadata_record_op_latency(&s->common, OP_STATX, started);
    job->stat_cb(job->user,
                 rc ? metadata_stat_status_from_errno(error) : STORE_STAT_OK,
                 rc ? 0 : (uint64_t)st.st_size);
    return;
  }
  enum damacy_status status = DAMACY_OK;
  void* data = NULL;
  size_t len = job->requested_len;
  uint64_t started = metadata_monotonic_ns();
  int fd = open(job->key, O_RDONLY | O_CLOEXEC), error = errno;
  metadata_record_op_latency(&s->common, OP_OPEN, started);
  if (fd < 0) {
    status = metadata_status_from_errno(error);
    goto Complete;
  }
  if (job->kind == REQ_READ_FILE) {
    struct stat st;
    started = metadata_monotonic_ns();
    int rc = fstat(fd, &st);
    metadata_record_op_latency(&s->common, OP_STATX, started);
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
  metadata_read_active_begin(&s->common);
  started = metadata_monotonic_ns();
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
  metadata_record_op_latency(&s->common, OP_READ, started);
  metadata_read_active_end(&s->common);
Close:
  started = metadata_monotonic_ns();
  if (close(fd) && status == DAMACY_OK)
    log_warn("metadata_store_async: close failed for %s: %s",
             job->key,
             strerror(errno));
  metadata_record_op_latency(&s->common, OP_CLOSE, started);
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
  metadata_store_common_destroy(&s->common);
  pthread_cond_destroy(&s->wake);
  pthread_mutex_destroy(&s->lock);
  free(s->workers);
  free(s);
}

struct metadata_store_common*
metadata_store_async_common(struct metadata_store_async* s)
{
  return &s->common;
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
  if (metadata_store_common_init(&s->common, latency)) {
    pthread_cond_destroy(&s->wake);
    pthread_mutex_destroy(&s->lock);
    goto Fail;
  }
  for (int i = 0; i < concurrency; ++i) {
    int rc = pthread_create(&s->workers[i], NULL, worker_main, s);
    if (rc) {
      log_error("metadata_store_async: pthread_create failed for worker %d "
                "of %d: %s",
                i + 1,
                concurrency,
                strerror(rc));
      metadata_store_async_destroy(s);
      return NULL;
    }
    ++s->started;
  }
  log_info("metadata_store_async: using POSIX worker metadata path "
           "(workers=%d)",
           concurrency);
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
