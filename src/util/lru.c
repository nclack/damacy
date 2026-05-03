#include "lru.h"

#include <stdlib.h>
#include <string.h>

#define LRU_NIL UINT32_MAX

struct lru_entry
{
  uint64_t hash;
  void* value;
  uint32_t refcount;
  uint32_t lru_prev; // toward MRU; LRU_NIL if this is the head
  uint32_t lru_next; // toward LRU; LRU_NIL if this is the tail
};

struct lru_idx
{
  uint32_t slot; // LRU_NIL = empty
};

struct lru
{
  struct lru_ops ops;
  uint32_t capacity;
  uint32_t max_probe;
  uint32_t idx_size; // power of two; >= 2 * capacity
  uint32_t mask;     // idx_size - 1

  struct lru_entry* slots; // capacity entries
  struct lru_idx* idx;     // idx_size entries

  uint32_t size;
  uint32_t lru_head;  // MRU; LRU_NIL when empty
  uint32_t lru_tail;  // LRU; LRU_NIL when empty
  uint32_t free_head; // LRU_NIL when no free slots
  // free list links live in slots[s].lru_next (reused)

  // stats
  uint64_t hits, misses, insertions, evictions, replacements, put_failures;
  uint32_t max_probe_observed;
};

static uint32_t
next_pow2_ge(uint32_t v)
{
  if (v < 2)
    return 2;
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  return v + 1;
}

struct lru*
lru_create(uint32_t capacity, uint32_t max_probe, const struct lru_ops* ops)
{
  if (capacity == 0 || max_probe == 0 || !ops || !ops->eq || !ops->destroy)
    return NULL;
  struct lru* l = (struct lru*)calloc(1, sizeof(*l));
  if (!l)
    return NULL;
  l->ops = *ops;
  l->capacity = capacity;
  l->max_probe = max_probe;
  l->idx_size = next_pow2_ge(2u * capacity);
  l->mask = l->idx_size - 1u;

  l->slots = (struct lru_entry*)calloc(capacity, sizeof(*l->slots));
  l->idx = (struct lru_idx*)malloc(l->idx_size * sizeof(*l->idx));
  if (!l->slots || !l->idx) {
    free(l->slots);
    free(l->idx);
    free(l);
    return NULL;
  }
  for (uint32_t i = 0; i < l->idx_size; ++i)
    l->idx[i].slot = LRU_NIL;

  // Build free list: slot[i] -> slot[i+1] via lru_next.
  for (uint32_t i = 0; i + 1 < capacity; ++i)
    l->slots[i].lru_next = i + 1;
  if (capacity > 0)
    l->slots[capacity - 1].lru_next = LRU_NIL;
  l->free_head = (capacity > 0) ? 0u : LRU_NIL;

  l->lru_head = LRU_NIL;
  l->lru_tail = LRU_NIL;
  return l;
}

void
lru_destroy(struct lru* l)
{
  if (!l)
    return;
  // Walk the LRU list and destroy values for live entries.
  for (uint32_t s = l->lru_head; s != LRU_NIL; s = l->slots[s].lru_next)
    l->ops.destroy(l->slots[s].value, l->ops.user);
  free(l->slots);
  free(l->idx);
  free(l);
}

// --- LRU list helpers (intrusive doubly-linked) ---------------------------

static void
lru_list_push_front(struct lru* l, uint32_t s)
{
  l->slots[s].lru_prev = LRU_NIL;
  l->slots[s].lru_next = l->lru_head;
  if (l->lru_head != LRU_NIL)
    l->slots[l->lru_head].lru_prev = s;
  l->lru_head = s;
  if (l->lru_tail == LRU_NIL)
    l->lru_tail = s;
}

static void
lru_list_unlink(struct lru* l, uint32_t s)
{
  uint32_t prev = l->slots[s].lru_prev;
  uint32_t next = l->slots[s].lru_next;
  if (prev != LRU_NIL)
    l->slots[prev].lru_next = next;
  else
    l->lru_head = next;
  if (next != LRU_NIL)
    l->slots[next].lru_prev = prev;
  else
    l->lru_tail = prev;
}

static void
lru_list_promote(struct lru* l, uint32_t s)
{
  if (l->lru_head == s)
    return;
  lru_list_unlink(l, s);
  lru_list_push_front(l, s);
}

// --- Free slot management -------------------------------------------------

static uint32_t
slot_alloc(struct lru* l)
{
  if (l->free_head == LRU_NIL)
    return LRU_NIL;
  uint32_t s = l->free_head;
  l->free_head = l->slots[s].lru_next;
  return s;
}

static void
slot_release(struct lru* l, uint32_t s)
{
  l->slots[s].lru_next = l->free_head;
  l->free_head = s;
}

// --- Robin-hood index helpers --------------------------------------------

static uint32_t
ideal_pos(const struct lru* l, uint64_t hash)
{
  return (uint32_t)(hash & l->mask);
}

static uint32_t
probe_dist(const struct lru* l, uint32_t pos, uint32_t ideal)
{
  return (pos - ideal) & l->mask;
}

// Backshift-delete the index entry at position `pos`. Walks forward,
// shifting entries with probe > 0 backward by one until we hit an empty
// slot or a probe == 0 entry.
static void
idx_erase(struct lru* l, uint32_t pos)
{
  for (;;) {
    uint32_t next = (pos + 1u) & l->mask;
    uint32_t s = l->idx[next].slot;
    if (s == LRU_NIL) {
      l->idx[pos].slot = LRU_NIL;
      return;
    }
    uint32_t ideal = ideal_pos(l, l->slots[s].hash);
    if (probe_dist(l, next, ideal) == 0) {
      l->idx[pos].slot = LRU_NIL;
      return;
    }
    l->idx[pos] = l->idx[next];
    pos = next;
  }
}

// --- Eviction -------------------------------------------------------------

// Evict the least-recently-used non-pinned entry. Returns 0 on success,
// -1 if no candidate exists.
static int
evict_one(struct lru* l)
{
  uint32_t s = l->lru_tail;
  while (s != LRU_NIL && l->slots[s].refcount > 0)
    s = l->slots[s].lru_prev;
  if (s == LRU_NIL)
    return -1;

  // Find this entry's index position by re-probing.
  uint32_t pos = ideal_pos(l, l->slots[s].hash);
  for (uint32_t p = 0; p <= l->max_probe; ++p) {
    if (l->idx[pos].slot == s) {
      idx_erase(l, pos);
      break;
    }
    pos = (pos + 1u) & l->mask;
  }

  void* val = l->slots[s].value;
  lru_list_unlink(l, s);
  // Reset the slot before releasing it (lru_next is reused as free link).
  l->slots[s].hash = 0;
  l->slots[s].value = NULL;
  l->slots[s].refcount = 0;
  slot_release(l, s);
  l->size--;
  l->evictions++;
  l->ops.destroy(val, l->ops.user);
  return 0;
}

// --- Public API -----------------------------------------------------------

struct lru_entry*
lru_get(struct lru* l, uint64_t hash, const void* probe_key)
{
  if (!l)
    return NULL;
  uint32_t pos = ideal_pos(l, hash);
  for (uint32_t p = 0; p <= l->max_probe; ++p) {
    uint32_t s = l->idx[pos].slot;
    if (s == LRU_NIL)
      break;
    if (l->slots[s].hash == hash &&
        l->ops.eq(l->slots[s].value, probe_key, l->ops.user)) {
      lru_list_promote(l, s);
      l->hits++;
      return &l->slots[s];
    }
    pos = (pos + 1u) & l->mask;
  }
  l->misses++;
  return NULL;
}

// Insert a slot index into the table via linear probing. Returns 0 on
// success, -1 if the chain is full (probe would exceed max_probe).
// Backshift deletion preserves the invariant that lookup terminates on
// the first empty slot, so robin-hood reordering during insertion is
// not required for correctness.
static int
idx_insert(struct lru* l, uint32_t s)
{
  uint32_t pos = ideal_pos(l, l->slots[s].hash);
  for (uint32_t p = 0; p <= l->max_probe; ++p) {
    if (l->idx[pos].slot == LRU_NIL) {
      l->idx[pos].slot = s;
      if (p > l->max_probe_observed)
        l->max_probe_observed = p;
      return 0;
    }
    pos = (pos + 1u) & l->mask;
  }
  return -1;
}

struct lru_entry*
lru_put(struct lru* l, uint64_t hash, const void* probe_key, void* value)
{
  if (!l) {
    return NULL;
  }
  if (!value) {
    // Treat NULL value as invalid; nothing to destroy.
    return NULL;
  }

  // 1. If a matching entry exists, replace its value.
  uint32_t pos = ideal_pos(l, hash);
  for (uint32_t p = 0; p <= l->max_probe; ++p) {
    uint32_t s = l->idx[pos].slot;
    if (s == LRU_NIL)
      break;
    if (l->slots[s].hash == hash &&
        l->ops.eq(l->slots[s].value, probe_key, l->ops.user)) {
      void* old = l->slots[s].value;
      l->slots[s].value = value;
      lru_list_promote(l, s);
      l->replacements++;
      l->ops.destroy(old, l->ops.user);
      return &l->slots[s];
    }
    pos = (pos + 1u) & l->mask;
  }

  // 2. Allocate a slot, evicting as needed.
  uint32_t s;
  for (;;) {
    s = slot_alloc(l);
    if (s != LRU_NIL)
      break;
    if (evict_one(l) != 0) {
      // All entries pinned; cannot make room.
      l->put_failures++;
      l->ops.destroy(value, l->ops.user);
      return NULL;
    }
  }

  l->slots[s].hash = hash;
  l->slots[s].value = value;
  l->slots[s].refcount = 0;

  // 3. Insert into the index, evicting on probe-too-deep.
  for (;;) {
    if (idx_insert(l, s) == 0)
      break;
    if (evict_one(l) != 0) {
      // Couldn't make room. Roll back: free our slot and destroy value.
      void* val = l->slots[s].value;
      l->slots[s].hash = 0;
      l->slots[s].value = NULL;
      slot_release(l, s);
      l->put_failures++;
      l->ops.destroy(val, l->ops.user);
      return NULL;
    }
  }

  // 4. Link into LRU list at MRU.
  lru_list_push_front(l, s);
  l->size++;
  l->insertions++;
  return &l->slots[s];
}

void*
lru_entry_value(const struct lru_entry* e)
{
  return e ? e->value : NULL;
}

void
lru_entry_acquire(struct lru_entry* e)
{
  if (!e)
    return;
  e->refcount++;
}

void
lru_entry_release(struct lru_entry* e)
{
  if (!e || e->refcount == 0)
    return;
  e->refcount--;
}

void
lru_stats_get(const struct lru* l, struct lru_stats* out)
{
  if (!out)
    return;
  if (!l) {
    memset(out, 0, sizeof(*out));
    return;
  }
  uint32_t pinned = 0;
  for (uint32_t s = l->lru_head; s != LRU_NIL; s = l->slots[s].lru_next) {
    if (l->slots[s].refcount > 0)
      ++pinned;
  }
  out->hits = l->hits;
  out->misses = l->misses;
  out->insertions = l->insertions;
  out->evictions = l->evictions;
  out->replacements = l->replacements;
  out->put_failures = l->put_failures;
  out->size = l->size;
  out->capacity = l->capacity;
  out->pinned = pinned;
  out->max_probe_observed = l->max_probe_observed;
}

void
lru_stats_reset(struct lru* l)
{
  if (!l)
    return;
  l->hits = 0;
  l->misses = 0;
  l->insertions = 0;
  l->evictions = 0;
  l->replacements = 0;
  l->put_failures = 0;
  l->max_probe_observed = 0;
}
