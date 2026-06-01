#include "store/store.h"

#include "log/log.h"
#include "platform/platform.h"
#include "store/store_internal.h"
#include "util/prelude.h"

void
store_destroy(struct store* s)
{
  CHECK_SILENT(Out, s);
  CHECK_SILENT(Out, s->vt);
  CHECK_SILENT(Out, s->vt->destroy);
  s->vt->destroy(s);
Out:
  return;
}

struct store_submit_result
store_read_submit(struct store* s, const struct store_read* reads, size_t n)
{
  CHECK_SILENT(Empty, s);
  CHECK_SILENT(Empty, s->vt);
  CHECK_SILENT(Empty, s->vt->submit);
  return s->vt->submit(s, reads, n);
Empty:
  return (struct store_submit_result){ .status = DAMACY_IO };
}

struct store_submit_result
store_read_submit_dev(struct store* s, const struct store_read* reads, size_t n)
{
  CHECK_SILENT(Empty, s);
  CHECK_SILENT(Empty, s->vt);
  CHECK_SILENT(Empty, s->vt->submit_dev);
  return s->vt->submit_dev(s, reads, n);
Empty:
  return (struct store_submit_result){ .status = DAMACY_IO };
}

int
store_supports_gds(struct store* s)
{
  if (!s || !s->vt)
    return 0;
  return s->vt->submit_dev != NULL;
}

enum damacy_status
store_event_wait(struct store* s, struct store_event ev)
{
  CHECK_SILENT(Fail, s);
  CHECK_SILENT(Fail, s->vt);
  CHECK_SILENT(Fail, s->vt->event_wait);
  return s->vt->event_wait(s, ev);
Fail:
  return DAMACY_IO;
}

struct store_event_poll
store_event_query(struct store* s, struct store_event ev)
{
  CHECK_SILENT(Fail, s);
  CHECK_SILENT(Fail, s->vt);
  CHECK_SILENT(Fail, s->vt->event_query);
  return s->vt->event_query(s, ev);
Fail:
  return (struct store_event_poll){ .status = DAMACY_IO, .ready = 1 };
}

void
store_event_discard(struct store* s, struct store_event ev)
{
  CHECK_SILENT(Out, s);
  CHECK_SILENT(Out, s->vt);
  if (s->vt->event_discard)
    s->vt->event_discard(s, ev);
Out:
  return;
}

int
store_read_many(struct store* s, const struct store_read* reads, size_t n)
{
  CHECK_SILENT(Fail, s);
  CHECK_SILENT(Fail, reads);
  if (n == 0)
    return 0;
  struct store_submit_result result = store_read_submit(s, reads, n);
  CHECK_SILENT(Fail, result.status == DAMACY_OK);
  return store_event_wait(s, result.event) == DAMACY_OK ? 0 : 1;
Fail:
  return 1;
}

enum store_stat_result
store_stat(struct store* s, const char* key, uint64_t* out)
{
  CHECK_SILENT(Fail, s);
  CHECK_SILENT(Fail, s->vt);
  CHECK_SILENT(Fail, s->vt->stat);
  CHECK_SILENT(Fail, key);
  CHECK_SILENT(Fail, out);
  return s->vt->stat(s, key, out);
Fail:
  return STORE_STAT_ERROR;
}

int
store_map(struct store* s, const char* key, struct store_view* out)
{
  CHECK_SILENT(Fail, s);
  CHECK_SILENT(Fail, s->vt);
  CHECK_SILENT(Fail, s->vt->map);
  CHECK_SILENT(Fail, key);
  CHECK_SILENT(Fail, out);
  return s->vt->map(s, key, out);
Fail:
  return 1;
}

void
store_unmap(struct store* s, struct store_view* view)
{
  CHECK_SILENT(Out, s);
  CHECK_SILENT(Out, s->vt);
  CHECK_SILENT(Out, s->vt->unmap);
  CHECK_SILENT(Out, view);
  s->vt->unmap(s, view);
Out:
  return;
}

uint32_t
store_default_fd_cache_capacity(void)
{
  enum
  {
    HEADROOM = 64,
    MIN_CAP = 16,
    MAX_CAP = 4096,
    FALLBACK = 256,
  };
  uint64_t limit = platform_max_open_files();
  if (limit == 0 || limit <= HEADROOM) {
    log_info("store: RLIMIT_NOFILE unknown or <= headroom; fd_cache_capacity "
             "fallback=%u (override via store_fs_config.fd_cache_capacity)",
             (unsigned)FALLBACK);
    return FALLBACK;
  }
  uint64_t cap = (limit - HEADROOM) / 2;
  if (cap < MIN_CAP)
    return MIN_CAP;
  if (cap > MAX_CAP) {
    log_info("store: fd_cache_capacity clamped to %u (RLIMIT_NOFILE=%llu); "
             "raise the cap explicitly via store_fs_config.fd_cache_capacity",
             (unsigned)MAX_CAP,
             (unsigned long long)limit);
    return MAX_CAP;
  }
  return (uint32_t)cap;
}
