#include "lookahead.h"

#include "log/log.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

void
sample_slot_clear(struct damacy_sample_slot* slot)
{
  if (!slot)
    return;
  free(slot->uri);
  slot->uri = NULL;
  memset(&slot->aabb, 0, sizeof(slot->aabb));
  slot->sample_seq = 0;
}

int
lookahead_init(struct damacy_lookahead* la, uint32_t cap)
{
  CHECK(Error, la);
  CHECK(Error, cap > 0);
  memset(la, 0, sizeof(*la));
  la->slots =
    (struct damacy_sample_slot*)calloc(cap, sizeof(struct damacy_sample_slot));
  CHECK(Error, la->slots);
  la->cap = cap;
  la->lock = platform_mutex_new();
  CHECK(Error, la->lock);
  la->cond = platform_cond_new();
  CHECK(Error, la->cond);
  return 0;
Error:
  lookahead_destroy(la);
  return 1;
}

void
lookahead_destroy(struct damacy_lookahead* la)
{
  if (!la)
    return;
  if (la->slots) {
    for (uint32_t i = 0; i < la->cap; ++i)
      sample_slot_clear(&la->slots[i]);
    free(la->slots);
    la->slots = NULL;
  }
  if (la->cond) {
    platform_cond_free(la->cond);
    la->cond = NULL;
  }
  if (la->lock) {
    platform_mutex_free(la->lock);
    la->lock = NULL;
  }
}

int
lookahead_push_with_sample_seq(struct damacy_lookahead* la,
                               const struct damacy_sample* sample,
                               uint64_t sample_seq)
{
  CHECK(Bad, la);
  CHECK(Bad, sample);
  CHECK(Bad, sample->uri);

  int rc = 1;
  platform_mutex_lock(la->lock);
  if (la->size == la->cap)
    goto Unlock;
  struct damacy_sample_slot* slot = &la->slots[la->tail];
  slot->uri = strdup(sample->uri);
  if (!slot->uri)
    goto Unlock;
  slot->aabb = sample->aabb;
  slot->sample_seq = sample_seq;
  la->tail = (la->tail + 1) % la->cap;
  la->size++;
  platform_cond_broadcast(la->cond);
  rc = 0;
Unlock:
  platform_mutex_unlock(la->lock);
  return rc;
Bad:
  return 1;
}

int
lookahead_push(struct damacy_lookahead* la, const struct damacy_sample* sample)
{
  return lookahead_push_with_sample_seq(la, sample, 0);
}

static void
pop_one_locked(struct damacy_lookahead* la, struct damacy_sample_slot* out)
{
  *out = la->slots[la->head];
  la->slots[la->head] = (struct damacy_sample_slot){ 0 };
  la->head = (la->head + 1) % la->cap;
  la->size--;
}

void
lookahead_drain(struct damacy_lookahead* la,
                struct damacy_sample_slot* out,
                uint32_t n)
{
  CHECK(End, la);
  CHECK(End, out);
  platform_mutex_lock(la->lock);
  uint32_t take = n < la->size ? n : la->size;
  for (uint32_t i = 0; i < take; ++i)
    pop_one_locked(la, &out[i]);
  platform_mutex_unlock(la->lock);
End:
  return;
}

int
lookahead_pop_blocking(struct damacy_lookahead* la,
                       struct damacy_sample_slot* out)
{
  CHECK(Bad, la);
  CHECK(Bad, out);
  platform_mutex_lock(la->lock);
  while (la->size == 0 && !la->stop_signaled)
    platform_cond_wait(la->cond, la->lock);
  int popped = 0;
  if (la->size > 0) {
    pop_one_locked(la, out);
    popped = 1;
  }
  platform_mutex_unlock(la->lock);
  return popped;
Bad:
  return 0;
}

int
lookahead_pop_blocking_timeout(struct damacy_lookahead* la,
                               struct damacy_sample_slot* out,
                               int timeout_ms)
{
  CHECK(Bad, la);
  CHECK(Bad, out);
  platform_mutex_lock(la->lock);
  if (la->size == 0 && !la->stop_signaled)
    platform_cond_timedwait_ms(la->cond, la->lock, timeout_ms);
  int popped = 0;
  if (la->size > 0) {
    pop_one_locked(la, out);
    popped = 1;
  }
  platform_mutex_unlock(la->lock);
  return popped;
Bad:
  return 0;
}

int
lookahead_try_pop(struct damacy_lookahead* la, struct damacy_sample_slot* out)
{
  CHECK(Bad, la);
  CHECK(Bad, out);
  platform_mutex_lock(la->lock);
  int popped = 0;
  if (la->size > 0) {
    pop_one_locked(la, out);
    popped = 1;
  }
  platform_mutex_unlock(la->lock);
  return popped;
Bad:
  return 0;
}

void
lookahead_signal_stop(struct damacy_lookahead* la)
{
  if (!la)
    return;
  platform_mutex_lock(la->lock);
  la->stop_signaled = 1;
  platform_cond_broadcast(la->cond);
  platform_mutex_unlock(la->lock);
}

uint32_t
lookahead_size(struct damacy_lookahead* la)
{
  if (!la)
    return 0;
  platform_mutex_lock(la->lock);
  uint32_t n = la->size;
  platform_mutex_unlock(la->lock);
  return n;
}
