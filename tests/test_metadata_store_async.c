#include "expect.h"
#include "fixture.h"
#include "store/metadata_store_async.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct read_wait
{
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int done;
  enum damacy_status status;
  void* data;
  size_t len;
};

struct stat_wait
{
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int done;
  enum store_stat_result status;
  uint64_t size;
};

static void
read_cb(void* user, enum damacy_status status, void* data, size_t len)
{
  struct read_wait* w = (struct read_wait*)user;
  pthread_mutex_lock(&w->mutex);
  w->status = status;
  w->data = data;
  w->len = len;
  w->done = 1;
  pthread_cond_signal(&w->cond);
  pthread_mutex_unlock(&w->mutex);
}

static void
stat_cb(void* user, enum store_stat_result status, uint64_t size)
{
  struct stat_wait* w = (struct stat_wait*)user;
  pthread_mutex_lock(&w->mutex);
  w->status = status;
  w->size = size;
  w->done = 1;
  pthread_cond_signal(&w->cond);
  pthread_mutex_unlock(&w->mutex);
}

static void
read_wait_init(struct read_wait* w)
{
  memset(w, 0, sizeof(*w));
  pthread_mutex_init(&w->mutex, NULL);
  pthread_cond_init(&w->cond, NULL);
}

static void
stat_wait_init(struct stat_wait* w)
{
  memset(w, 0, sizeof(*w));
  pthread_mutex_init(&w->mutex, NULL);
  pthread_cond_init(&w->cond, NULL);
}

static void
read_wait_done(struct read_wait* w)
{
  free(w->data);
  pthread_cond_destroy(&w->cond);
  pthread_mutex_destroy(&w->mutex);
}

static void
stat_wait_done(struct stat_wait* w)
{
  pthread_cond_destroy(&w->cond);
  pthread_mutex_destroy(&w->mutex);
}

static void
read_wait_block(struct read_wait* w)
{
  pthread_mutex_lock(&w->mutex);
  while (!w->done)
    pthread_cond_wait(&w->cond, &w->mutex);
  pthread_mutex_unlock(&w->mutex);
}

static int
read_wait_block_ms(struct read_wait* w, int timeout_ms)
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += timeout_ms / 1000;
  ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
  if (ts.tv_nsec >= 1000000000L) {
    ts.tv_sec++;
    ts.tv_nsec -= 1000000000L;
  }

  pthread_mutex_lock(&w->mutex);
  while (!w->done) {
    int rc = pthread_cond_timedwait(&w->cond, &w->mutex, &ts);
    if (rc) {
      pthread_mutex_unlock(&w->mutex);
      return 0;
    }
  }
  pthread_mutex_unlock(&w->mutex);
  return 1;
}

static void
stat_wait_block(struct stat_wait* w)
{
  pthread_mutex_lock(&w->mutex);
  while (!w->done)
    pthread_cond_wait(&w->cond, &w->mutex);
  pthread_mutex_unlock(&w->mutex);
}

static int
make_root(char* root, size_t root_cap)
{
  snprintf(root, root_cap, "/tmp/damacy_metadata_store_async_XXXXXX");
  return mkdtemp(root) ? 0 : 1;
}

static int
test_read_stat_and_stats(void)
{
  char root[128];
  EXPECT(make_root(root, sizeof root) == 0);

  char path[256];
  snprintf(path, sizeof path, "%s/payload", root);
  EXPECT(fixture_write_file(path, "abcdef") == 0);

  struct metadata_store_async* s = metadata_store_async_create(
    1, NULL, &(struct damacy_latency_model){ .baseline_ns = 1000 });
  EXPECT(s);

  struct read_wait whole;
  read_wait_init(&whole);
  EXPECT(metadata_store_async_read_file(s, path, read_cb, &whole) == 0);
  read_wait_block(&whole);
  EXPECT(whole.status == DAMACY_OK);
  EXPECT(whole.len == 6);
  EXPECT(memcmp(whole.data, "abcdef", 6) == 0);
  read_wait_done(&whole);

  struct read_wait slice;
  read_wait_init(&slice);
  EXPECT(metadata_store_async_read(s, path, 2, 3, read_cb, &slice) == 0);
  read_wait_block(&slice);
  EXPECT(slice.status == DAMACY_OK);
  EXPECT(slice.len == 3);
  EXPECT(memcmp(slice.data, "cde", 3) == 0);
  read_wait_done(&slice);

  struct stat_wait st;
  stat_wait_init(&st);
  EXPECT(metadata_store_async_stat(s, path, stat_cb, &st) == 0);
  stat_wait_block(&st);
  EXPECT(st.status == STORE_STAT_OK);
  EXPECT(st.size == 6);
  stat_wait_done(&st);

  char missing[256];
  snprintf(missing, sizeof missing, "%s/missing", root);
  struct read_wait miss;
  read_wait_init(&miss);
  EXPECT(metadata_store_async_read_file(s, missing, read_cb, &miss) == 0);
  read_wait_block(&miss);
  EXPECT(miss.status == DAMACY_NOTFOUND);
  EXPECT(miss.data == NULL);
  read_wait_done(&miss);

  struct metadata_store_async_backend_stats backend;
  metadata_store_async_backend_stats_get(s, &backend);
  EXPECT(backend.read_jobs >= 2);
  EXPECT(backend.read_active == 0);
  EXPECT(backend.read_max_active >= 1);

  struct metadata_store_async_latency_stats lat;
  metadata_store_async_latency_stats_get(s, &lat);
  EXPECT(lat.ops >= 4);
  EXPECT(lat.submit_ops >= 3);
  EXPECT(lat.stat_ops >= 1);
  EXPECT(lat.active == 0);
  EXPECT(lat.max_active >= 1);
  EXPECT(lat.total_sleep_ns >= lat.ops * 1000);

  metadata_store_async_latency_stats_reset(s);
  metadata_store_async_backend_stats_reset(s);
  metadata_store_async_latency_stats_get(s, &lat);
  metadata_store_async_backend_stats_get(s, &backend);
  EXPECT(lat.ops == 0);
  EXPECT(backend.read_jobs == 0);

  metadata_store_async_destroy(s);
  fixture_rm_tree(root);
  return 0;
}

static uint64_t
op_latency_bucket_total(const struct metadata_store_async_op_latency_kind* k)
{
  uint64_t total = 0;
  for (unsigned b = 0; b < METADATA_OP_LATENCY_NBUCKETS; ++b)
    total += k->buckets[b];
  return total;
}

static int
test_op_latency_measured(void)
{
  char root[128];
  EXPECT(make_root(root, sizeof root) == 0);

  char path[256];
  snprintf(path, sizeof path, "%s/payload", root);
  EXPECT(fixture_write_file(path, "abcdef") == 0);

  struct metadata_store_async* s =
    metadata_store_async_create(1, NULL, &(struct damacy_latency_model){ 0 });
  EXPECT(s);

  struct read_wait whole;
  read_wait_init(&whole);
  EXPECT(metadata_store_async_read_file(s, path, read_cb, &whole) == 0);
  read_wait_block(&whole);
  EXPECT(whole.status == DAMACY_OK);
  read_wait_done(&whole);

  struct stat_wait st;
  stat_wait_init(&st);
  EXPECT(metadata_store_async_stat(s, path, stat_cb, &st) == 0);
  stat_wait_block(&st);
  EXPECT(st.status == STORE_STAT_OK);
  stat_wait_done(&st);

  struct metadata_store_async_op_latency_stats ol;
  metadata_store_async_op_latency_stats_get(s, &ol);

  // REQ_READ_FILE issues statx+open+read+close; REQ_STAT issues one statx.
  EXPECT(ol.kinds[0].count >= 2); // statx
  EXPECT(ol.kinds[1].count >= 1); // open
  EXPECT(ol.kinds[2].count >= 1); // read
  EXPECT(ol.kinds[3].count >= 1); // close
  for (unsigned k = 0; k < METADATA_OP_LATENCY_NKINDS; ++k) {
    EXPECT(op_latency_bucket_total(&ol.kinds[k]) == ol.kinds[k].count);
    if (ol.kinds[k].count) {
      EXPECT(ol.kinds[k].max_ns > 0);
      EXPECT(ol.kinds[k].sum_ns >= ol.kinds[k].max_ns);
    }
  }

  metadata_store_async_op_latency_stats_reset(s);
  metadata_store_async_op_latency_stats_get(s, &ol);
  for (unsigned k = 0; k < METADATA_OP_LATENCY_NKINDS; ++k) {
    EXPECT(ol.kinds[k].count == 0);
    EXPECT(ol.kinds[k].sum_ns == 0);
    EXPECT(ol.kinds[k].max_ns == 0);
    EXPECT(op_latency_bucket_total(&ol.kinds[k]) == 0);
  }

  metadata_store_async_destroy(s);
  fixture_rm_tree(root);
  return 0;
}

static int
test_read_file_open_error_completes(void)
{
  char root[128];
  EXPECT(make_root(root, sizeof root) == 0);

  char path[256];
  snprintf(path, sizeof path, "%s/unreadable", root);
  EXPECT(fixture_write_file(path, "secret") == 0);
  EXPECT(chmod(path, 0000) == 0);

  struct metadata_store_async* s =
    metadata_store_async_create(1, NULL, &(struct damacy_latency_model){ 0 });
  EXPECT(s);

  struct read_wait w;
  read_wait_init(&w);
  EXPECT(metadata_store_async_read_file(s, path, read_cb, &w) == 0);
  EXPECT(read_wait_block_ms(&w, 2000) == 1);
  if (geteuid() == 0)
    EXPECT(w.status == DAMACY_OK || w.status == DAMACY_IO);
  else
    EXPECT(w.status == DAMACY_IO);
  read_wait_done(&w);

  metadata_store_async_destroy(s);
  chmod(path, 0600);
  fixture_rm_tree(root);
  return 0;
}

int
main(void)
{
  RUN(test_read_stat_and_stats);
  RUN(test_op_latency_measured);
  RUN(test_read_file_open_error_completes);
  return 0;
}
