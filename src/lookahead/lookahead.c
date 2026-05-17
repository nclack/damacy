#include "lookahead.h"

#include "log/log.h"
#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

void
sample_slot_clear(struct damacy_sample_slot* slot, struct path_intern* uris)
{
  if (!slot)
    return;
  if (slot->uri)
    path_intern_release(uris, slot->uri);
  slot->uri = NULL;
  memset(&slot->aabb, 0, sizeof(slot->aabb));
}

int
lookahead_init(struct damacy_lookahead* la,
               uint32_t cap,
               struct path_intern* uris)
{
  la->slots =
    (struct damacy_sample_slot*)calloc(cap, sizeof(struct damacy_sample_slot));
  CHECK(Error, la->slots);
  la->uris = uris;
  la->cap = cap;
  la->head = 0;
  la->tail = 0;
  la->size = 0;
  return 0;
Error:
  return 1;
}

void
lookahead_destroy(struct damacy_lookahead* la)
{
  if (!la || !la->slots)
    return;
  for (uint32_t i = 0; i < la->cap; ++i)
    sample_slot_clear(&la->slots[i], la->uris);
  free(la->slots);
  la->slots = NULL;
}

int
lookahead_push(struct damacy_lookahead* la, const struct damacy_sample* sample)
{
  if (la->size == la->cap)
    return 1;
  struct damacy_sample_slot* slot = &la->slots[la->tail];
  slot->uri = path_intern_acquire(la->uris, sample->uri);
  if (!slot->uri)
    return 1;
  slot->aabb = sample->aabb;
  la->tail = (la->tail + 1) % la->cap;
  la->size++;
  return 0;
}

void
lookahead_drain(struct damacy_lookahead* la,
                struct damacy_sample_slot* out,
                uint32_t n)
{
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = la->slots[la->head];
    la->slots[la->head] = (struct damacy_sample_slot){ 0 };
    la->head = (la->head + 1) % la->cap;
    la->size--;
  }
}
