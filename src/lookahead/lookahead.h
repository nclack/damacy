// Push-side ring of pending samples. URI strings are interned in a
// caller-supplied path_intern table; each slot holds one acquired ref,
// transferred on drain and released by sample_slot_clear.
#pragma once

#include "damacy.h"
#include "util/path_intern.h"

#include <stdint.h>

struct damacy_sample_slot
{
  const char* uri;
  struct damacy_aabb aabb;
};

struct damacy_lookahead
{
  struct damacy_sample_slot* slots;
  struct path_intern* uris; // borrowed; must outlive the lookahead
  uint32_t cap;
  uint32_t head;
  uint32_t tail;
  uint32_t size;
};

// Release slot->uri through `uris` and zero the slot. NULL-safe.
void sample_slot_clear(struct damacy_sample_slot* slot,
                       struct path_intern* uris);

// `uris` is borrowed; the caller owns it and must outlive the lookahead.
// Returns 0 on success, non-zero on alloc failure.
int lookahead_init(struct damacy_lookahead* la,
                   uint32_t cap,
                   struct path_intern* uris);

// NULL-safe. Releases any URIs still held by un-drained slots.
void lookahead_destroy(struct damacy_lookahead* la);

// Acquires a URI ref via la->uris. Returns 0 on success, non-zero if
// full or the intern acquire failed.
int lookahead_push(struct damacy_lookahead* la,
                   const struct damacy_sample* sample);

// Moves n slots from the ring into `out`, transferring the URI ref.
// Caller is responsible for n <= la->size.
void lookahead_drain(struct damacy_lookahead* la,
                     struct damacy_sample_slot* out,
                     uint32_t n);
