#pragma once

#include "damacy.h"

#include <stdint.h>

struct damacy_sample_slot
{
  char* uri;
  struct damacy_aabb aabb;
};

struct damacy_lookahead
{
  struct damacy_sample_slot* slots;
  uint32_t cap;
  uint32_t head;
  uint32_t tail;
  uint32_t size;
};

void
sample_slot_clear(struct damacy_sample_slot* slot);

int
lookahead_init(struct damacy_lookahead* la, uint32_t cap);

void
lookahead_destroy(struct damacy_lookahead* la);

int
lookahead_push(struct damacy_lookahead* la, const struct damacy_sample* sample);

void
lookahead_drain(struct damacy_lookahead* la,
                struct damacy_sample_slot* out,
                uint32_t n);
