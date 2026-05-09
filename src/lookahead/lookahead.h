// Push-side ring of pending samples. Owns the URI string copies; the
// drain step transfers ownership to a caller-provided sample-slot array.
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

// Frees slot->uri and zeroes the slot. NULL-safe.
void sample_slot_clear(struct damacy_sample_slot* slot);

// Returns 0 on success, non-zero on alloc failure.
int lookahead_init(struct damacy_lookahead* la, uint32_t cap);

// NULL-safe.
void lookahead_destroy(struct damacy_lookahead* la);

// Returns 0 on success, non-zero if full or strdup failed.
int lookahead_push(struct damacy_lookahead* la,
                   const struct damacy_sample* sample);

// Moves n slots from the ring into `out` (transferring uri ownership).
// Caller is responsible for n <= la->size.
void lookahead_drain(struct damacy_lookahead* la,
                     struct damacy_sample_slot* out,
                     uint32_t n);
