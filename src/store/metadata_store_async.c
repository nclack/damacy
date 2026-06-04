#include "store/metadata_store_async.h"

#include "io_queue/io_queue.h"

#include <stdlib.h>
#include <string.h>

struct metadata_store_async
{
  struct store* store;
  struct io_queue* q;
};

struct read_job
{
  struct metadata_store_async* s;
  char* key;
  uint64_t offset;
  size_t len;
  int whole_file;
  metadata_store_read_cb cb;
  void* user;
};

struct stat_job
{
  struct metadata_store_async* s;
  char* key;
  metadata_store_stat_cb cb;
  void* user;
};

static void
read_job_free(void* ctx_)
{
  struct read_job* ctx = (struct read_job*)ctx_;
  if (!ctx)
    return;
  free(ctx->key);
  free(ctx);
}

static void
stat_job_free(void* ctx_)
{
  struct stat_job* ctx = (struct stat_job*)ctx_;
  if (!ctx)
    return;
  free(ctx->key);
  free(ctx);
}

static enum damacy_status
stat_to_status(enum store_stat_result s)
{
  switch (s) {
    case STORE_STAT_OK:
      return DAMACY_OK;
    case STORE_STAT_NOT_FOUND:
      return DAMACY_NOTFOUND;
    case STORE_STAT_ERROR:
      break;
  }
  return DAMACY_IO;
}

static void
read_job_run(void* ctx_)
{
  struct read_job* ctx = (struct read_job*)ctx_;
  enum damacy_status status = DAMACY_OK;
  void* data = NULL;
  size_t len = ctx->len;

  if (ctx->whole_file) {
    uint64_t size = 0;
    enum store_stat_result st = store_stat(ctx->s->store, ctx->key, &size);
    status = stat_to_status(st);
    if (status != DAMACY_OK)
      goto Done;
    if (size > SIZE_MAX) {
      status = DAMACY_BUDGET;
      goto Done;
    }
    len = (size_t)size;
  }

  if (len) {
    data = malloc(len);
    if (!data) {
      status = DAMACY_OOM;
      goto Done;
    }
    struct store_read read = {
      .key = ctx->key,
      .dst = data,
      .offset = ctx->offset,
      .len = len,
    };
    if (store_read_many(ctx->s->store, &read, 1)) {
      free(data);
      data = NULL;
      status = DAMACY_IO;
      goto Done;
    }
  }

Done:
  ctx->cb(ctx->user, status, data, len);
}

static void
stat_job_run(void* ctx_)
{
  struct stat_job* ctx = (struct stat_job*)ctx_;
  uint64_t size = 0;
  enum store_stat_result status = store_stat(ctx->s->store, ctx->key, &size);
  ctx->cb(ctx->user, status, size);
}

struct metadata_store_async*
metadata_store_async_create(struct store* store,
                            int concurrency,
                            const struct numa_resolved* affinity)
{
  if (!store || concurrency <= 0)
    return NULL;
  struct metadata_store_async* s =
    (struct metadata_store_async*)calloc(1, sizeof(*s));
  if (!s)
    return NULL;
  s->store = store;
  s->q = io_queue_create(concurrency, affinity);
  if (!s->q) {
    free(s);
    return NULL;
  }
  return s;
}

void
metadata_store_async_destroy(struct metadata_store_async* s)
{
  if (!s)
    return;
  io_queue_destroy(s->q);
  free(s);
}

static int
post_read(struct metadata_store_async* s,
          const char* key,
          uint64_t offset,
          size_t len,
          int whole_file,
          metadata_store_read_cb cb,
          void* user)
{
  if (!s || !key || !cb)
    return 1;
  struct read_job* ctx = (struct read_job*)calloc(1, sizeof(*ctx));
  if (!ctx)
    return 1;
  char* key_copy = strdup(key);
  if (!key_copy) {
    free(ctx);
    return 1;
  }
  *ctx = (struct read_job){
    .s = s,
    .key = key_copy,
    .offset = offset,
    .len = len,
    .whole_file = whole_file,
    .cb = cb,
    .user = user,
  };
  if (io_queue_post(s->q, read_job_run, ctx, read_job_free)) {
    read_job_free(ctx);
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
  return post_read(s, key, 0, 0, 1, cb, user);
}

int
metadata_store_async_read(struct metadata_store_async* s,
                          const char* key,
                          uint64_t offset,
                          size_t len,
                          metadata_store_read_cb cb,
                          void* user)
{
  return post_read(s, key, offset, len, 0, cb, user);
}

int
metadata_store_async_stat(struct metadata_store_async* s,
                          const char* key,
                          metadata_store_stat_cb cb,
                          void* user)
{
  if (!s || !key || !cb)
    return 1;
  struct stat_job* ctx = (struct stat_job*)calloc(1, sizeof(*ctx));
  if (!ctx)
    return 1;
  char* key_copy = strdup(key);
  if (!key_copy) {
    free(ctx);
    return 1;
  }
  *ctx = (struct stat_job){
    .s = s,
    .key = key_copy,
    .cb = cb,
    .user = user,
  };
  if (io_queue_post(s->q, stat_job_run, ctx, stat_job_free)) {
    stat_job_free(ctx);
    return 1;
  }
  return 0;
}
