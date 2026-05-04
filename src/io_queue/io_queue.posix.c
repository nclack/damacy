#include "io_queue/io_queue.h"

#include "util/prelude.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct io_job
{
  void (*fn)(void*);
  void* ctx;
  void (*ctx_free)(void*);
  uint64_t seq;
};

struct io_queue
{
  pthread_t* workers;
  int nworkers;

  pthread_mutex_t mutex;
  pthread_cond_t cond_not_empty;
  pthread_cond_t cond_retired;

  struct io_job* ring;
  uint64_t ring_cap; // power of 2; also the bitmap size
  uint64_t head;
  uint64_t tail;

  // Per-seq completion bitmap, indexed by `seq mod ring_cap`.
  // Set when the job for that seq has finished. Drained when retired_seq
  // advances through contiguous bits.
  uint8_t* completed;

  uint64_t next_seq;
  uint64_t retired_seq;

  int shutdown;
  int started;
};

// Caller holds q->mutex.
static void
drain_completed(struct io_queue* q)
{
  for (;;) {
    uint64_t want = q->retired_seq + 1;
    uint8_t* slot = &q->completed[want & (q->ring_cap - 1)];
    if (!*slot)
      break;
    *slot = 0;
    q->retired_seq = want;
  }
  pthread_cond_broadcast(&q->cond_retired);
}

static void*
worker_thread(void* arg)
{
  struct io_queue* q = (struct io_queue*)arg;
  for (;;) {
    struct io_job job;

    pthread_mutex_lock(&q->mutex);
    while (q->head == q->tail && !q->shutdown)
      pthread_cond_wait(&q->cond_not_empty, &q->mutex);

    if (q->head == q->tail && q->shutdown) {
      pthread_mutex_unlock(&q->mutex);
      break;
    }

    job = q->ring[q->tail & (q->ring_cap - 1)];
    q->tail++;
    pthread_mutex_unlock(&q->mutex);

    job.fn(job.ctx);
    if (job.ctx_free)
      job.ctx_free(job.ctx);

    pthread_mutex_lock(&q->mutex);
    q->completed[job.seq & (q->ring_cap - 1)] = 1;
    if (job.seq == q->retired_seq + 1)
      drain_completed(q);
    pthread_mutex_unlock(&q->mutex);
  }
  return NULL;
}

// Helper: free a partially-constructed io_queue from io_queue_create's
// Fail label. Each leaf-free is NULL-safe; this just guards the
// container deref.
static void
io_queue_free_partial(struct io_queue* q)
{
  if (!q)
    return;
  free(q->ring);
  free(q->completed);
  free(q->workers);
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->cond_not_empty);
  pthread_cond_destroy(&q->cond_retired);
  free(q);
}

struct io_queue*
io_queue_create(int nthreads)
{
  struct io_queue* q = NULL;

  CHECK_SILENT(Fail, nthreads >= 0);
  q = (struct io_queue*)calloc(1, sizeof(*q));
  CHECK_SILENT(Fail, q);

  // Init sync primitives first so io_queue_free_partial can always
  // unconditionally destroy them.
  pthread_mutex_init(&q->mutex, NULL);
  pthread_cond_init(&q->cond_not_empty, NULL);
  pthread_cond_init(&q->cond_retired, NULL);

  q->ring_cap = 64;
  q->ring = (struct io_job*)calloc(q->ring_cap, sizeof(struct io_job));
  q->completed = (uint8_t*)calloc(q->ring_cap, sizeof(uint8_t));
  CHECK_SILENT(Fail, q->ring);
  CHECK_SILENT(Fail, q->completed);

  q->nworkers = nthreads;
  if (nthreads > 0) {
    q->workers = (pthread_t*)calloc((size_t)nthreads, sizeof(pthread_t));
    CHECK_SILENT(Fail, q->workers);
    for (int i = 0; i < nthreads; ++i) {
      if (pthread_create(&q->workers[i], NULL, worker_thread, q) != 0) {
        // Tear down already-started workers.
        pthread_mutex_lock(&q->mutex);
        q->shutdown = 1;
        pthread_cond_broadcast(&q->cond_not_empty);
        pthread_mutex_unlock(&q->mutex);
        for (int j = 0; j < i; ++j)
          pthread_join(q->workers[j], NULL);
        goto Fail;
      }
    }
  }
  q->started = 1;
  return q;

Fail:
  io_queue_free_partial(q);
  return NULL;
}

void
io_queue_destroy(struct io_queue* q)
{
  if (!q)
    return;

  pthread_mutex_lock(&q->mutex);
  q->shutdown = 1;
  pthread_cond_broadcast(&q->cond_not_empty);
  pthread_mutex_unlock(&q->mutex);

  if (q->started) {
    for (int i = 0; i < q->nworkers; ++i)
      pthread_join(q->workers[i], NULL);
  }
  free(q->workers);
  free(q->ring);
  free(q->completed);
  pthread_mutex_destroy(&q->mutex);
  pthread_cond_destroy(&q->cond_not_empty);
  pthread_cond_destroy(&q->cond_retired);
  free(q);
}

// Caller holds q->mutex; in-flight count must be 0 (head == tail) or all
// jobs in the ring must be unretired (ie. the bitmap holds no live bits we
// would lose). We grow only when the ring is full, which means head == tail
// (mod ring_cap) and head - tail == ring_cap; reissuing the ring relocates
// jobs but the completion bitmap must be remapped consistently.
static int
ring_grow(struct io_queue* q)
{
  uint64_t new_cap = q->ring_cap * 2;
  struct io_job* new_ring =
    (struct io_job*)calloc(new_cap, sizeof(struct io_job));
  uint8_t* new_done = (uint8_t*)calloc(new_cap, sizeof(uint8_t));
  if (!new_ring || !new_done) {
    free(new_ring);
    free(new_done);
    return 1;
  }

  uint64_t count = q->head - q->tail;
  for (uint64_t i = 0; i < count; ++i) {
    struct io_job job = q->ring[(q->tail + i) & (q->ring_cap - 1)];
    new_ring[i] = job;
    // Remap the completion bit for this seq from old slot to new.
    uint64_t old_slot = job.seq & (q->ring_cap - 1);
    uint64_t new_slot = job.seq & (new_cap - 1);
    new_done[new_slot] = q->completed[old_slot];
    q->completed[old_slot] = 0;
  }
  // Any remaining set bits in q->completed are for seqs already retired —
  // safe to drop. Replace ring + bitmap.
  free(q->ring);
  free(q->completed);
  q->ring = new_ring;
  q->completed = new_done;
  q->ring_cap = new_cap;
  q->head = count;
  q->tail = 0;
  return 0;
}

int
io_queue_post(struct io_queue* q,
              void (*fn)(void*),
              void* ctx,
              void (*ctx_free)(void*))
{
  if (!q || !fn)
    return 1;

  // nthreads == 0: synchronous fast path. Run on the caller, advance seq.
  if (q->nworkers == 0) {
    pthread_mutex_lock(&q->mutex);
    q->next_seq++;
    uint64_t my_seq = q->next_seq;
    pthread_mutex_unlock(&q->mutex);
    fn(ctx);
    if (ctx_free)
      ctx_free(ctx);
    pthread_mutex_lock(&q->mutex);
    q->retired_seq = my_seq;
    pthread_cond_broadcast(&q->cond_retired);
    pthread_mutex_unlock(&q->mutex);
    return 0;
  }

  pthread_mutex_lock(&q->mutex);
  if (q->head - q->tail == q->ring_cap) {
    if (ring_grow(q)) {
      pthread_mutex_unlock(&q->mutex);
      return 1;
    }
  }
  q->next_seq++;
  q->ring[q->head & (q->ring_cap - 1)] = (struct io_job){
    .fn = fn,
    .ctx = ctx,
    .ctx_free = ctx_free,
    .seq = q->next_seq,
  };
  q->head++;
  pthread_cond_signal(&q->cond_not_empty);
  pthread_mutex_unlock(&q->mutex);
  return 0;
}

struct io_event
io_queue_record(struct io_queue* q)
{
  pthread_mutex_lock(&q->mutex);
  struct io_event ev = { .seq = q->next_seq };
  pthread_mutex_unlock(&q->mutex);
  return ev;
}

void
io_event_wait(const struct io_queue* q, struct io_event ev)
{
  struct io_queue* mq = (struct io_queue*)q;
  pthread_mutex_lock(&mq->mutex);
  while (mq->retired_seq < ev.seq && !mq->shutdown)
    pthread_cond_wait(&mq->cond_retired, &mq->mutex);
  pthread_mutex_unlock(&mq->mutex);
}

int
io_event_query(const struct io_queue* q, struct io_event ev)
{
  struct io_queue* mq = (struct io_queue*)q;
  pthread_mutex_lock(&mq->mutex);
  int retired = (mq->retired_seq >= ev.seq);
  pthread_mutex_unlock(&mq->mutex);
  return retired;
}

int
io_queue_is_shutdown(const struct io_queue* q)
{
  struct io_queue* mq = (struct io_queue*)q;
  pthread_mutex_lock(&mq->mutex);
  int r = mq->shutdown;
  pthread_mutex_unlock(&mq->mutex);
  return r;
}
