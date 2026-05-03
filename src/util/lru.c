#include "lru.h"

#include "util/prelude.h"

#include <stdlib.h>
#include <string.h>

#define LRU_NIL UINT32_MAX

// Per-slot data — the public lru_entry pointer is just &slots[i].
// Slots are stable (never moved); the index reorders references to
// them. lru_prev/lru_next double as the free-list link when the slot
// is unused (free_next is stored in lru_next).
struct lru_entry
{
  uint64_t hash;
  void* value;
  uint32_t refcount;
  uint32_t lru_prev; // toward MRU; LRU_NIL if head
  uint32_t lru_next; // toward LRU; LRU_NIL if tail (or free-list link)
};

// Index entry: a slot index (or LRU_NIL for empty).
struct lru_idx_cell
{
  uint32_t slot;
};

// Robin-hood hash index over slot indices.
struct lru_index
{
  struct lru_idx_cell* cells; // idx_size entries
  uint32_t idx_size;          // power of two
  uint32_t mask;              // idx_size - 1
  uint32_t max_probe;
  uint32_t max_probe_observed;
};

// Doubly-linked list of slot indices (intrusive in struct lru_entry).
struct lru_list
{
  uint32_t head; // MRU end; LRU_NIL when empty
  uint32_t tail; // LRU end; LRU_NIL when empty
};

// Singly-linked freelist of slot indices (intrusive in slot.lru_next).
struct lru_freelist
{
  uint32_t head; // LRU_NIL when empty
};

// Internal counters; copied into the public lru_stats on stats_get.
struct lru_counters
{
  uint64_t hits;
  uint64_t misses;
  uint64_t insertions;
  uint64_t evictions;
  uint64_t replacements;
  uint64_t put_failures;
};

struct lru
{
  struct lru_ops ops;
  struct lru_entry* slots;
  uint32_t capacity;
  uint32_t size;
  struct lru_index index;
  struct lru_list list;
  struct lru_freelist freelist;
  struct lru_counters counters;
};

// --- small helpers --------------------------------------------------------

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

static inline uint32_t
ideal_pos(const struct lru* l, uint64_t hash)
{
  return (uint32_t)(hash & l->index.mask);
}

static inline uint32_t
probe_dist(const struct lru* l, uint32_t pos, uint32_t ideal)
{
  return (pos - ideal) & l->index.mask;
}

// --- LRU list helpers (intrusive doubly-linked) ---------------------------

static void
list_push_front(struct lru* l, uint32_t s)
{
  l->slots[s].lru_prev = LRU_NIL;
  l->slots[s].lru_next = l->list.head;
  if (l->list.head != LRU_NIL)
    l->slots[l->list.head].lru_prev = s;
  l->list.head = s;
  if (l->list.tail == LRU_NIL)
    l->list.tail = s;
}

static void
list_unlink(struct lru* l, uint32_t s)
{
  uint32_t prev = l->slots[s].lru_prev;
  uint32_t next = l->slots[s].lru_next;
  if (prev != LRU_NIL)
    l->slots[prev].lru_next = next;
  else
    l->list.head = next;
  if (next != LRU_NIL)
    l->slots[next].lru_prev = prev;
  else
    l->list.tail = prev;
}

static void
list_promote(struct lru* l, uint32_t s)
{
  if (l->list.head == s)
    return;
  list_unlink(l, s);
  list_push_front(l, s);
}

// --- free-list helpers ---------------------------------------------------

static uint32_t
freelist_pop(struct lru* l)
{
  if (l->freelist.head == LRU_NIL)
    return LRU_NIL;
  uint32_t s = l->freelist.head;
  l->freelist.head = l->slots[s].lru_next;
  return s;
}

static void
freelist_push(struct lru* l, uint32_t s)
{
  l->slots[s].lru_next = l->freelist.head;
  l->freelist.head = s;
}

// --- index helpers -------------------------------------------------------

// Robin-hood insert. Pinned entries (refcount > 0) are immovable; we
// probe past them without considering them for swap. Returns LRU_NIL on
// success, or the index of the orphaned slot on cap-exceed. The orphan
// is guaranteed to have refcount == 0 (we never displace pinned).
static uint32_t
index_insert_robin(struct lru* l, uint32_t s_in)
{
  uint32_t s = s_in;
  uint32_t pos = ideal_pos(l, l->slots[s].hash);
  for (uint32_t p = 0; p <= l->index.max_probe; ++p) {
    uint32_t cur = l->index.cells[pos].slot;
    if (cur == LRU_NIL) {
      l->index.cells[pos].slot = s;
      if (p > l->index.max_probe_observed)
        l->index.max_probe_observed = p;
      return LRU_NIL;
    }
    // Pinned entries are immovable; never swap with them.
    if (l->slots[cur].refcount == 0) {
      uint32_t cur_ideal = ideal_pos(l, l->slots[cur].hash);
      uint32_t cur_probe = probe_dist(l, pos, cur_ideal);
      if (p > cur_probe) {
        l->index.cells[pos].slot = s;
        if (p > l->index.max_probe_observed)
          l->index.max_probe_observed = p;
        s = cur;
        p = cur_probe;
      }
    }
    pos = (pos + 1u) & l->index.mask;
  }
  return s;
}

// Backshift-delete the cell at `pos`. Walks forward shifting unpinned
// entries with probe > 0 backward by one until we hit an empty cell, a
// pinned entry, or an entry with probe == 0. Pinned entries cannot be
// shifted, so a pinned entry adjacent to the deleted cell leaves a gap.
static void
index_erase(struct lru* l, uint32_t pos)
{
  for (;;) {
    uint32_t next = (pos + 1u) & l->index.mask;
    uint32_t s = l->index.cells[next].slot;
    if (s == LRU_NIL) {
      l->index.cells[pos].slot = LRU_NIL;
      return;
    }
    if (l->slots[s].refcount > 0) {
      // Pinned entries don't move; leave a gap at `pos`.
      l->index.cells[pos].slot = LRU_NIL;
      return;
    }
    uint32_t ideal = ideal_pos(l, l->slots[s].hash);
    if (probe_dist(l, next, ideal) == 0) {
      l->index.cells[pos].slot = LRU_NIL;
      return;
    }
    l->index.cells[pos] = l->index.cells[next];
    pos = next;
  }
}

// Look up `s`'s current index position (re-probing from its ideal).
// Returns the position or LRU_NIL if not found (shouldn't happen for
// live entries).
static uint32_t
index_locate(const struct lru* l, uint32_t s)
{
  uint32_t pos = ideal_pos(l, l->slots[s].hash);
  for (uint32_t p = 0; p <= l->index.max_probe; ++p) {
    if (l->index.cells[pos].slot == s)
      return pos;
    pos = (pos + 1u) & l->index.mask;
  }
  return LRU_NIL;
}

// Look up an entry by hash + probe key; returns LRU_NIL on miss. Walks
// the full probe range — pin-aware insertion can leave gaps in chains,
// so empty cells do not terminate the search.
static uint32_t
index_lookup(const struct lru* l, uint64_t hash, const void* probe_key)
{
  uint32_t pos = ideal_pos(l, hash);
  for (uint32_t p = 0; p <= l->index.max_probe; ++p) {
    uint32_t s = l->index.cells[pos].slot;
    if (s != LRU_NIL && l->slots[s].hash == hash &&
        l->ops.eq(l->slots[s].value, probe_key, l->ops.user))
      return pos;
    pos = (pos + 1u) & l->index.mask;
  }
  return LRU_NIL;
}

// --- eviction -------------------------------------------------------------

// Free slot `s`'s entry: erase from index, unlink from LRU, return slot
// to the free list, destroy the value. Caller must ensure refcount == 0.
static void
evict_slot(struct lru* l, uint32_t s)
{
  uint32_t pos = index_locate(l, s);
  if (pos != LRU_NIL)
    index_erase(l, pos);

  void* val = l->slots[s].value;
  list_unlink(l, s);
  l->slots[s] = (struct lru_entry){
    .hash = 0,
    .value = NULL,
    .refcount = 0,
    .lru_prev = LRU_NIL,
    .lru_next = LRU_NIL,
  };
  freelist_push(l, s);
  l->size--;
  l->counters.evictions++;
  l->ops.destroy(val, l->ops.user);
}

// Walk the LRU list from tail forward looking for an unpinned entry.
// Returns 0 on success (one entry evicted), -1 if every live entry is
// pinned.
static int
evict_lru_tail(struct lru* l)
{
  uint32_t s = l->list.tail;
  while (s != LRU_NIL && l->slots[s].refcount > 0)
    s = l->slots[s].lru_prev;
  if (s == LRU_NIL)
    return -1;
  evict_slot(l, s);
  return 0;
}

// --- public API -----------------------------------------------------------

struct lru*
lru_create(uint32_t capacity, uint32_t max_probe, const struct lru_ops* ops)
{
  struct lru* l = NULL;
  CHECK_SILENT(error, capacity > 0);
  CHECK_SILENT(error, max_probe > 0);
  CHECK_SILENT(error, ops && ops->eq && ops->destroy);

  l = (struct lru*)calloc(1, sizeof(*l));
  CHECK(error, l);

  uint32_t idx_size = next_pow2_ge(2u * capacity);

  *l = (struct lru){
    .ops = *ops,
    .capacity = capacity,
    .size = 0,
    .index = {
      .cells = NULL,
      .idx_size = idx_size,
      .mask = idx_size - 1u,
      .max_probe = max_probe,
      .max_probe_observed = 0,
    },
    .list = { .head = LRU_NIL, .tail = LRU_NIL },
    .freelist = { .head = LRU_NIL },
    .counters = { 0 },
  };

  l->slots = (struct lru_entry*)calloc(capacity, sizeof(*l->slots));
  CHECK(error, l->slots);
  l->index.cells =
    (struct lru_idx_cell*)malloc(idx_size * sizeof(*l->index.cells));
  CHECK(error, l->index.cells);

  for (uint32_t i = 0; i < idx_size; ++i)
    l->index.cells[i] = (struct lru_idx_cell){ .slot = LRU_NIL };

  // Build the freelist: slot[0] -> slot[1] -> ... -> slot[capacity-1].
  for (uint32_t i = 0; i + 1 < capacity; ++i)
    l->slots[i].lru_next = i + 1;
  l->slots[capacity - 1].lru_next = LRU_NIL;
  l->freelist.head = 0;

  return l;

error:
  if (l) {
    free(l->slots);
    free(l->index.cells);
    free(l);
  }
  return NULL;
}

void
lru_destroy(struct lru* l)
{
  if (!l)
    return;
  for (uint32_t s = l->list.head; s != LRU_NIL; s = l->slots[s].lru_next)
    l->ops.destroy(l->slots[s].value, l->ops.user);
  free(l->slots);
  free(l->index.cells);
  free(l);
}

struct lru_entry*
lru_get(struct lru* l, uint64_t hash, const void* probe_key)
{
  if (!l)
    return NULL;
  uint32_t pos = index_lookup(l, hash, probe_key);
  if (pos == LRU_NIL) {
    l->counters.misses++;
    return NULL;
  }
  uint32_t s = l->index.cells[pos].slot;
  list_promote(l, s);
  l->counters.hits++;
  return &l->slots[s];
}

struct lru_entry*
lru_put(struct lru* l, uint64_t hash, const void* probe_key, void* value)
{
  if (!l || !value) {
    if (l && value)
      l->ops.destroy(value, l->ops.user);
    return NULL;
  }

  // 1. Replace path: same key already present. Refuse if pinned.
  uint32_t pos = index_lookup(l, hash, probe_key);
  if (pos != LRU_NIL) {
    uint32_t s = l->index.cells[pos].slot;
    if (l->slots[s].refcount > 0) {
      // Pinned entry can't be replaced — would invalidate the pin.
      l->counters.put_failures++;
      l->ops.destroy(value, l->ops.user);
      return NULL;
    }
    void* old = l->slots[s].value;
    l->slots[s].value = value;
    list_promote(l, s);
    l->counters.replacements++;
    l->ops.destroy(old, l->ops.user);
    return &l->slots[s];
  }

  // 2. Allocate a slot, evicting the LRU tail if the pool is empty.
  uint32_t s_new = freelist_pop(l);
  while (s_new == LRU_NIL) {
    if (evict_lru_tail(l) != 0) {
      l->counters.put_failures++;
      l->ops.destroy(value, l->ops.user);
      return NULL;
    }
    s_new = freelist_pop(l);
  }
  l->slots[s_new] = (struct lru_entry){
    .hash = hash,
    .value = value,
    .refcount = 0,
    .lru_prev = LRU_NIL,
    .lru_next = LRU_NIL,
  };

  // 3. Robin-hood insert. On cap-exceed, evict and retry. The orphan
  //    is whoever ended up unplaced — could be s_new or a previously-
  //    stored slot.
  for (;;) {
    uint32_t orphan = index_insert_robin(l, s_new);
    if (orphan == LRU_NIL)
      break;
    if (orphan == s_new) {
      // Our value never landed. Evict to free a slot in our chain, retry.
      // If eviction itself fails (everything pinned), give up.
      if (evict_lru_tail(l) != 0) {
        // Fallback: destroy the new value and return our slot.
        void* val = l->slots[s_new].value;
        l->slots[s_new] = (struct lru_entry){
          .hash = 0,
          .value = NULL,
          .refcount = 0,
          .lru_prev = LRU_NIL,
          .lru_next = LRU_NIL,
        };
        freelist_push(l, s_new);
        l->counters.put_failures++;
        l->ops.destroy(val, l->ops.user);
        return NULL;
      }
      continue;
    }
    // Orphan is a previously-existing entry. Evict it (destroy value,
    // unlink from LRU, free slot). Our new slot is now in the index.
    void* val = l->slots[orphan].value;
    list_unlink(l, orphan);
    l->slots[orphan] = (struct lru_entry){
      .hash = 0,
      .value = NULL,
      .refcount = 0,
      .lru_prev = LRU_NIL,
      .lru_next = LRU_NIL,
    };
    freelist_push(l, orphan);
    l->size--;
    l->counters.evictions++;
    l->ops.destroy(val, l->ops.user);
    break;
  }

  // 4. Link new slot at MRU.
  list_push_front(l, s_new);
  l->size++;
  l->counters.insertions++;
  return &l->slots[s_new];
}

void*
lru_entry_value(const struct lru_entry* e)
{
  return e ? e->value : NULL;
}

void
lru_entry_acquire(struct lru_entry* e)
{
  if (e)
    e->refcount++;
}

void
lru_entry_release(struct lru_entry* e)
{
  if (e && e->refcount > 0)
    e->refcount--;
}

void
lru_stats_get(const struct lru* l, struct lru_stats* out)
{
  if (!out)
    return;
  if (!l) {
    *out = (struct lru_stats){ 0 };
    return;
  }
  uint32_t pinned = 0;
  for (uint32_t s = l->list.head; s != LRU_NIL; s = l->slots[s].lru_next) {
    if (l->slots[s].refcount > 0)
      ++pinned;
  }
  *out = (struct lru_stats){
    .hits = l->counters.hits,
    .misses = l->counters.misses,
    .insertions = l->counters.insertions,
    .evictions = l->counters.evictions,
    .replacements = l->counters.replacements,
    .put_failures = l->counters.put_failures,
    .size = l->size,
    .capacity = l->capacity,
    .pinned = pinned,
    .max_probe_observed = l->index.max_probe_observed,
  };
}

void
lru_stats_reset(struct lru* l)
{
  if (!l)
    return;
  l->counters = (struct lru_counters){ 0 };
  l->index.max_probe_observed = 0;
}
