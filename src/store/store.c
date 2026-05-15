#include "store/store.h"

#include "store/store_fs.h"
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

struct store_event
store_read_submit(struct store* s, const struct store_read* reads, size_t n)
{
  CHECK_SILENT(Empty, s);
  CHECK_SILENT(Empty, s->vt);
  CHECK_SILENT(Empty, s->vt->submit);
  return s->vt->submit(s, reads, n);
Empty:
  return (struct store_event){ 0 };
}

struct store_event
store_read_submit_dev(struct store* s, const struct store_read* reads, size_t n)
{
  CHECK_SILENT(Empty, s);
  CHECK_SILENT(Empty, s->vt);
  CHECK_SILENT(Empty, s->vt->submit_dev);
  return s->vt->submit_dev(s, reads, n);
Empty:
  return (struct store_event){ 0 };
}

int
store_supports_gds(struct store* s)
{
  if (!s || !s->vt)
    return 0;
  return s->vt->submit_dev != NULL;
}

void
store_event_wait(struct store* s, struct store_event ev)
{
  CHECK_SILENT(Out, s);
  CHECK_SILENT(Out, s->vt);
  CHECK_SILENT(Out, s->vt->event_wait);
  s->vt->event_wait(s, ev);
Out:
  return;
}

int
store_event_query(struct store* s, struct store_event ev)
{
  CHECK_SILENT(NotReady, s);
  CHECK_SILENT(NotReady, s->vt);
  CHECK_SILENT(NotReady, s->vt->event_query);
  return s->vt->event_query(s, ev);
NotReady:
  return 0;
}

int
store_read_many(struct store* s, const struct store_read* reads, size_t n)
{
  CHECK_SILENT(Fail, s);
  CHECK_SILENT(Fail, reads);
  if (n == 0)
    return 0;
  struct store_event ev = store_read_submit(s, reads, n);
  CHECK_SILENT(Fail, ev.seq != 0);
  store_event_wait(s, ev);
  return 0;
Fail:
  return 1;
}

int
store_stat(struct store* s, const char* key, uint64_t* out)
{
  CHECK_SILENT(Fail, s);
  CHECK_SILENT(Fail, s->vt);
  CHECK_SILENT(Fail, s->vt->stat);
  CHECK_SILENT(Fail, key);
  CHECK_SILENT(Fail, out);
  return s->vt->stat(s, key, out);
Fail:
  return 1;
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
