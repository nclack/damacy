#include "store_fs.h"

#include "store.h"
#include "util/strbuf.h"

#include <stdlib.h>
#include <string.h>

static char*
str_dup(const char* s)
{
  if (!s)
    return NULL;
  size_t n = strlen(s);
  char* p = (char*)malloc(n + 1);
  if (!p)
    return NULL;
  memcpy(p, s, n + 1);
  return p;
}

static int
join_path(struct strbuf* sb, const char* root, const char* key)
{
  strbuf_reset(sb);
  if (root && root[0]) {
    if (strbuf_append_cstr(sb, root))
      return 1;
    size_t L = strbuf_len(sb);
    if (L > 0 && strbuf_cstr(sb)[L - 1] != '/') {
      if (strbuf_append(sb, "/", 1))
        return 1;
    }
  }
  return strbuf_append_cstr(sb, key);
}

// Locate or open a file for the given key. Returns NULL on failure.
// Caller does not own the returned handle; the store owns it for its
// lifetime.
static platform_file*
fs_get_file(struct store_fs* fs, const char* key)
{
  pthread_mutex_lock(&fs->cache_mu);
  for (size_t i = 0; i < fs->n_slots; ++i) {
    if (strcmp(fs->slots[i].key, key) == 0) {
      platform_file* f = fs->slots[i].file;
      pthread_mutex_unlock(&fs->cache_mu);
      return f;
    }
  }
  pthread_mutex_unlock(&fs->cache_mu);

  // Compose full path and open outside the lock.
  struct strbuf path = { 0 };
  if (join_path(&path, fs->root, key)) {
    strbuf_free(&path);
    return NULL;
  }
  platform_file* f = platform_file_open_read(strbuf_cstr(&path), 0);
  strbuf_free(&path);
  if (!f)
    return NULL;

  // Insert into cache; if a concurrent caller raced us, drop ours and reuse
  // theirs.
  pthread_mutex_lock(&fs->cache_mu);
  for (size_t i = 0; i < fs->n_slots; ++i) {
    if (strcmp(fs->slots[i].key, key) == 0) {
      platform_file* winner = fs->slots[i].file;
      pthread_mutex_unlock(&fs->cache_mu);
      platform_file_close(f);
      return winner;
    }
  }
  if (fs->n_slots == fs->cap_slots) {
    size_t new_cap = fs->cap_slots ? fs->cap_slots * 2 : 16;
    struct fs_cache_slot* p = (struct fs_cache_slot*)realloc(
      fs->slots, new_cap * sizeof(struct fs_cache_slot));
    if (!p) {
      pthread_mutex_unlock(&fs->cache_mu);
      platform_file_close(f);
      return NULL;
    }
    fs->slots = p;
    fs->cap_slots = new_cap;
  }
  char* dup = str_dup(key);
  if (!dup) {
    pthread_mutex_unlock(&fs->cache_mu);
    platform_file_close(f);
    return NULL;
  }
  fs->slots[fs->n_slots].key = dup;
  fs->slots[fs->n_slots].file = f;
  fs->n_slots++;
  pthread_mutex_unlock(&fs->cache_mu);
  return f;
}

// io_queue job context.
struct fs_read_job
{
  struct store_fs* fs;
  char* key; // heap-owned
  void* dst;
  uint64_t offset;
  size_t len;
};

static void
fs_read_job_fn(void* vctx)
{
  struct fs_read_job* j = (struct fs_read_job*)vctx;
  platform_file* f = fs_get_file(j->fs, j->key);
  if (!f)
    return; // silently fail; store has no error channel yet
  (void)platform_file_pread(f, j->dst, j->len, j->offset);
}

static void
fs_read_job_free(void* vctx)
{
  struct fs_read_job* j = (struct fs_read_job*)vctx;
  free(j->key);
  free(j);
}

static struct store_event
fs_submit(struct store* s, const struct store_read* reads, size_t n)
{
  struct store_fs* fs = (struct store_fs*)s;
  struct store_event ev = { 0 };
  if (n == 0) {
    struct io_event ioev = io_queue_record(fs->q);
    ev.seq = ioev.seq;
    return ev;
  }

  for (size_t i = 0; i < n; ++i) {
    struct fs_read_job* j =
      (struct fs_read_job*)calloc(1, sizeof(struct fs_read_job));
    if (!j)
      break;
    j->fs = fs;
    j->key = str_dup(reads[i].key);
    j->dst = reads[i].dst;
    j->offset = reads[i].offset;
    j->len = reads[i].len;
    if (!j->key) {
      free(j);
      break;
    }
    if (io_queue_post(fs->q, fs_read_job_fn, j, fs_read_job_free)) {
      fs_read_job_free(j);
      break;
    }
  }
  struct io_event ioev = io_queue_record(fs->q);
  ev.seq = ioev.seq;
  return ev;
}

static void
fs_event_wait(struct store* s, struct store_event ev)
{
  struct store_fs* fs = (struct store_fs*)s;
  io_event_wait(fs->q, (struct io_event){ .seq = ev.seq });
}

static int
fs_stat(struct store* s, const char* key, uint64_t* out)
{
  struct store_fs* fs = (struct store_fs*)s;
  struct strbuf path = { 0 };
  if (join_path(&path, fs->root, key)) {
    strbuf_free(&path);
    return 1;
  }
  int rc = platform_path_size(strbuf_cstr(&path), out);
  strbuf_free(&path);
  return rc;
}

static int
fs_map(struct store* s, const char* key, struct store_view* out)
{
  struct store_fs* fs = (struct store_fs*)s;
  struct strbuf path = { 0 };
  if (join_path(&path, fs->root, key)) {
    strbuf_free(&path);
    return 1;
  }
  struct platform_file_view* pv =
    (struct platform_file_view*)calloc(1, sizeof(*pv));
  if (!pv) {
    strbuf_free(&path);
    return 1;
  }
  int rc = platform_file_map_path(strbuf_cstr(&path), pv);
  strbuf_free(&path);
  if (rc) {
    free(pv);
    return rc;
  }
  out->data = pv->data;
  out->len = pv->len;
  out->backend = pv;
  return 0;
}

static void
fs_unmap(struct store* s, struct store_view* view)
{
  (void)s;
  if (!view || !view->backend)
    return;
  struct platform_file_view* pv = (struct platform_file_view*)view->backend;
  platform_file_unmap(pv);
  free(pv);
  view->data = NULL;
  view->len = 0;
  view->backend = NULL;
}

static void
fs_destroy(struct store* s)
{
  struct store_fs* fs = (struct store_fs*)s;
  if (!fs)
    return;
  // Drain pending I/O before tearing down file handles.
  if (fs->q) {
    io_event_wait(fs->q, io_queue_record(fs->q));
    io_queue_destroy(fs->q);
  }
  for (size_t i = 0; i < fs->n_slots; ++i) {
    free(fs->slots[i].key);
    platform_file_close(fs->slots[i].file);
  }
  free(fs->slots);
  pthread_mutex_destroy(&fs->cache_mu);
  free(fs->root);
  free(fs);
}

static const struct store_vtable fs_vtable = {
  .destroy = fs_destroy,
  .stat = fs_stat,
  .submit = fs_submit,
  .event_wait = fs_event_wait,
  .map = fs_map,
  .unmap = fs_unmap,
};

struct store*
store_fs_create(const struct store_fs_config* cfg)
{
  if (!cfg || !cfg->root)
    return NULL;
  struct store_fs* fs = (struct store_fs*)calloc(1, sizeof(*fs));
  if (!fs)
    return NULL;
  fs->base.vt = &fs_vtable;
  fs->root = str_dup(cfg->root);
  if (!fs->root)
    goto fail;
  pthread_mutex_init(&fs->cache_mu, NULL);
  fs->q = io_queue_create(cfg->nthreads);
  if (!fs->q)
    goto fail;
  return &fs->base;

fail:
  if (fs->q)
    io_queue_destroy(fs->q);
  pthread_mutex_destroy(&fs->cache_mu);
  free(fs->root);
  free(fs);
  return NULL;
}
