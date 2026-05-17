#include "util/path_intern.h"

#include "log/log.h"
#include "util/hash.h"
#include "util/prelude.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct path_intern_slot
{
  uint64_t hash;
  char* str; // NULL = empty bucket
  uint32_t refs;
};

static int
slots_rehash(struct path_intern* pi, size_t new_cap)
{
  struct path_intern_slot* ns =
    (struct path_intern_slot*)calloc(new_cap, sizeof(struct path_intern_slot));
  if (!ns)
    return 1;
  size_t mask = new_cap - 1u;
  for (size_t i = 0; i < pi->cap; ++i) {
    struct path_intern_slot* s = &pi->slots[i];
    if (!s->str)
      continue;
    size_t j = (size_t)s->hash & mask;
    while (ns[j].str)
      j = (j + 1u) & mask;
    ns[j] = *s;
  }
  free(pi->slots);
  pi->slots = ns;
  pi->cap = new_cap;
  return 0;
}

// Backward-shift deletion. Walk forward until an empty slot; any entry
// whose ideal home is at-or-before the gap (in mod order) migrates into
// the gap. The "stop at first in-home entry" optimization is unsafe —
// it orphans entries whose ideal preceded the gap but landed past an
// in-home neighbor (e.g. ideals A=5, B=6, C=5 land at 5,6,7; deleting
// A must shift C, not stop at B).
static void
slot_evict(struct path_intern* pi, size_t idx)
{
  if (!pi->cap)
    return;
  size_t mask = pi->cap - 1u;
  free(pi->slots[idx].str);
  pi->slots[idx] = (struct path_intern_slot){ 0 };
  pi->n--;
  for (size_t scan = (idx + 1u) & mask; pi->slots[scan].str;
       scan = (scan + 1u) & mask) {
    size_t ideal = (size_t)pi->slots[scan].hash & mask;
    if (((idx - ideal) & mask) <= ((scan - ideal) & mask)) {
      pi->slots[idx] = pi->slots[scan];
      pi->slots[scan] = (struct path_intern_slot){ 0 };
      idx = scan;
    }
  }
}

void
path_intern_free(struct path_intern* pi)
{
  if (!pi || !pi->slots)
    return;
  for (size_t i = 0; i < pi->cap; ++i)
    free(pi->slots[i].str);
  free(pi->slots);
  pi->slots = NULL;
  pi->cap = 0;
  pi->n = 0;
}

void
path_intern_reset(struct path_intern* pi)
{
  if (!pi || !pi->slots)
    return;
  for (size_t i = 0; i < pi->cap; ++i) {
    free(pi->slots[i].str);
    pi->slots[i] = (struct path_intern_slot){ 0 };
  }
  pi->n = 0;
}

const char*
path_intern_acquire(struct path_intern* pi, const char* s)
{
  if (!pi || !s)
    return NULL;

  // Grow before insert so load factor stays under 0.75.
  if ((pi->n + 1u) * 4u > pi->cap * 3u) {
    size_t new_cap = pi->cap ? pi->cap * 2u : 16u;
    if (slots_rehash(pi, new_cap))
      return NULL;
  }

  uint64_t h = hash_fnv1a_str(s);
  size_t mask = pi->cap - 1u;
  size_t j = (size_t)h & mask;
  for (;;) {
    struct path_intern_slot* slot = &pi->slots[j];
    if (!slot->str) {
      char* copy = strdup(s);
      if (!copy)
        return NULL;
      slot->hash = h;
      slot->str = copy;
      slot->refs = 1u;
      pi->n++;
      return slot->str;
    }
    if (slot->hash == h && strcmp(slot->str, s) == 0) {
      slot->refs++;
      return slot->str;
    }
    j = (j + 1u) & mask;
  }
}

void
path_intern_release(struct path_intern* pi, const char* s)
{
  if (!pi || !s)
    return;
  size_t idx;
  for (idx = 0; idx < pi->cap; ++idx)
    if (pi->slots[idx].str == s)
      break;
  CHECK(End, idx < pi->cap);
  struct path_intern_slot* slot = &pi->slots[idx];
  CHECK(End, slot->refs > 0);
  if (--slot->refs == 0)
    slot_evict(pi, idx);
End:
  return;
}
