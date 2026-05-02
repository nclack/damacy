#include "store.h"

#include "store_fs.h"

void
store_destroy(struct store* s)
{
  if (!s || !s->vt || !s->vt->destroy)
    return;
  s->vt->destroy(s);
}

struct store_event
store_read_submit(struct store* s, const struct store_read* reads, size_t n)
{
  if (!s || !s->vt || !s->vt->submit)
    return (struct store_event){ 0 };
  return s->vt->submit(s, reads, n);
}

void
store_event_wait(struct store* s, struct store_event ev)
{
  if (!s || !s->vt || !s->vt->event_wait)
    return;
  s->vt->event_wait(s, ev);
}

int
store_read_many(struct store* s, const struct store_read* reads, size_t n)
{
  if (!s || !reads)
    return 1;
  if (n == 0)
    return 0;
  struct store_event ev = store_read_submit(s, reads, n);
  if (ev.seq == 0)
    return 1;
  store_event_wait(s, ev);
  return 0;
}

int
store_stat(struct store* s, const char* key, uint64_t* out)
{
  if (!s || !s->vt || !s->vt->stat || !key || !out)
    return 1;
  return s->vt->stat(s, key, out);
}

int
store_map(struct store* s, const char* key, struct store_view* out)
{
  if (!s || !s->vt || !s->vt->map || !key || !out)
    return 1;
  return s->vt->map(s, key, out);
}

void
store_unmap(struct store* s, struct store_view* view)
{
  if (!s || !s->vt || !s->vt->unmap || !view)
    return;
  s->vt->unmap(s, view);
}
