#include "lru.h"

#include "log/log.h"
#include "util/prelude.h"

#include <assert.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define LRU_NIL UINT32_MAX

// Doubly-linked list primitive. Same shape used as both:
//   - the per-entry link (embedded in struct lru_entry as `link`)
//   - the list anchor (a sentinel node owned by struct lru; the list is
//     empty when sentinel.next == &sentinel).
// Using a sentinel removes head/tail special cases from list_*; pointer
// updates are unconditional.
struct lru_list
{
  struct lru_list* prev;
  struct lru_list* next;
};

// Per-slot data — the public lru_entry pointer is just &slots[slot_idx].
// Slots are stable (never moved); the index reorders references to them.
// `link` is in the LRU-order list when the slot is in use, and in the
// freelist when the slot is free.
struct lru_entry
{
  uint64_t hash;
  void* value;
  _Atomic uint32_t refcount;
  struct lru_list link;
};

// Robin-hood hash index over slot indices. Each cell stores a slot
// index into the slots array, or LRU_NIL when empty.
struct lru_index
{
  uint32_t* cells;  // n_cells entries; LRU_NIL = empty
  uint32_t n_cells; // power of two
  uint32_t mask;    // n_cells - 1
  uint32_t max_probe;
  uint32_t max_probe_observed;
};

struct lru
{
  struct lru_ops ops;
  struct lru_entry* slots;
  uint32_t capacity;
  uint32_t size;
  struct lru_index index;
  struct lru_list lru_order; // sentinel; .next = MRU, .prev = LRU
  struct lru_list freelist;  // sentinel; pop/push at .next
  struct lru_counters counters;
};

// --- list primitive (operates on struct lru_list only) -------------------

static void
list_init(struct lru_list* node)
{
  node->prev = node;
  node->next = node;
}

static int
list_empty(const struct lru_list* sentinel)
{
  return sentinel->next == sentinel;
}

// Detach `node` from whatever list it's in. Self-loops the node so a
// second unlink is a no-op (and a subsequent push works).
static void
list_unlink(struct lru_list* node)
{
  node->prev->next = node->next;
  node->next->prev = node->prev;
  node->prev = node;
  node->next = node;
}

// Insert `node` at the front of the list anchored by `sentinel`.
static void
list_push_front(struct lru_list* sentinel, struct lru_list* node)
{
  node->prev = sentinel;
  node->next = sentinel->next;
  sentinel->next->prev = node;
  sentinel->next = node;
}

// Move `node` to the front of `sentinel`'s list. No-op if already front.
static void
list_promote(struct lru_list* sentinel, struct lru_list* node)
{
  if (sentinel->next == node)
    return;
  list_unlink(node);
  list_push_front(sentinel, node);
}

// --- link <-> slot helpers -----------------------------------------------

static struct lru_entry*
link_to_entry(struct lru_list* link)
{
  return container_of(link, struct lru_entry, link);
}

static const struct lru_entry*
link_to_entry_const(const struct lru_list* link)
{
  return container_of(link, const struct lru_entry, link);
}

static uint32_t
entry_to_slot_idx(const struct lru* self, const struct lru_entry* entry)
{
  return (uint32_t)(entry - self->slots);
}

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

static uint32_t
ideal_cell(const struct lru* self, uint64_t hash)
{
  return (uint32_t)(hash & self->index.mask);
}

static uint32_t
probe_dist(const struct lru* self, uint32_t cell_idx, uint32_t ideal)
{
  return (cell_idx - ideal) & self->index.mask;
}

// --- free-list helpers ---------------------------------------------------

// Pop the front slot off the freelist. Returns LRU_NIL if empty.
static uint32_t
freelist_pop(struct lru* self)
{
  if (list_empty(&self->freelist))
    return LRU_NIL;
  struct lru_list* node = self->freelist.next;
  list_unlink(node);
  return entry_to_slot_idx(self, link_to_entry(node));
}

static void
freelist_push(struct lru* self, uint32_t slot_idx)
{
  list_push_front(&self->freelist, &self->slots[slot_idx].link);
}

// Reset the slot's payload to a "free" state. Caller is responsible for
// the link's list membership (unlink first; push to freelist after).
static void
slot_clear(struct lru* self, uint32_t slot_idx)
{
  struct lru_list saved_link = self->slots[slot_idx].link;
  self->slots[slot_idx] = (struct lru_entry){
    .link = saved_link,
  };
}

// --- index helpers -------------------------------------------------------

// Robin-hood insert. Pinned entries (refcount > 0) are immovable; we
// probe past them without considering them for swap. Returns LRU_NIL on
// success, or the index of the orphaned slot on cap-exceed. The orphan
// is guaranteed to have refcount == 0 (we never displace pinned).
static uint32_t
index_insert_robin(struct lru* self, uint32_t slot_to_insert)
{
  uint32_t carry_slot = slot_to_insert;
  uint32_t cell_idx = ideal_cell(self, self->slots[carry_slot].hash);
  for (uint32_t probe = 0; probe <= self->index.max_probe; ++probe) {
    uint32_t incumbent = self->index.cells[cell_idx];
    if (incumbent == LRU_NIL) {
      self->index.cells[cell_idx] = carry_slot;
      if (probe > self->index.max_probe_observed)
        self->index.max_probe_observed = probe;
      return LRU_NIL;
    }
    // relaxed: pin bumps run under this same mutex; only the lock-free
    // release in evict_lru_tail needs acquire.
    if (atomic_load_explicit(&self->slots[incumbent].refcount,
                             memory_order_relaxed) == 0) {
      uint32_t incumbent_ideal = ideal_cell(self, self->slots[incumbent].hash);
      uint32_t incumbent_probe = probe_dist(self, cell_idx, incumbent_ideal);
      if (probe > incumbent_probe) {
        self->index.cells[cell_idx] = carry_slot;
        if (probe > self->index.max_probe_observed)
          self->index.max_probe_observed = probe;
        carry_slot = incumbent;
        probe = incumbent_probe;
      }
    }
    cell_idx = (cell_idx + 1u) & self->index.mask;
  }
  return carry_slot;
}

// Backshift-delete the cell at `cell_idx`. Walks forward shifting unpinned
// entries with probe > 0 backward by one until we hit an empty cell, a
// pinned entry, or an entry with probe == 0. Pinned entries cannot be
// shifted, so a pinned entry adjacent to the deleted cell leaves a gap.
static void
index_erase(struct lru* self, uint32_t cell_idx)
{
  for (;;) {
    uint32_t next_cell = (cell_idx + 1u) & self->index.mask;
    uint32_t next_slot = self->index.cells[next_cell];
    if (next_slot == LRU_NIL) {
      self->index.cells[cell_idx] = LRU_NIL;
      return;
    }
    if (atomic_load_explicit(&self->slots[next_slot].refcount,
                             memory_order_relaxed) > 0) {
      // Pinned entries don't move; leave a gap at `cell_idx`.
      self->index.cells[cell_idx] = LRU_NIL;
      return;
    }
    uint32_t ideal = ideal_cell(self, self->slots[next_slot].hash);
    if (probe_dist(self, next_cell, ideal) == 0) {
      self->index.cells[cell_idx] = LRU_NIL;
      return;
    }
    self->index.cells[cell_idx] = self->index.cells[next_cell];
    cell_idx = next_cell;
  }
}

// Look up `slot_idx`'s current cell position (re-probing from its ideal).
// Returns the cell or LRU_NIL if not found (shouldn't happen for live
// entries).
static uint32_t
index_locate(const struct lru* self, uint32_t slot_idx)
{
  uint32_t cell_idx = ideal_cell(self, self->slots[slot_idx].hash);
  for (uint32_t probe = 0; probe <= self->index.max_probe; ++probe) {
    if (self->index.cells[cell_idx] == slot_idx)
      return cell_idx;
    cell_idx = (cell_idx + 1u) & self->index.mask;
  }
  return LRU_NIL;
}

// Look up an entry by hash + probe key; returns LRU_NIL on miss. Walks
// the full probe range — pin-aware insertion can leave gaps in chains,
// so empty cells do not terminate the search.
static uint32_t
index_lookup(const struct lru* self, uint64_t hash, const void* probe_key)
{
  uint32_t cell_idx = ideal_cell(self, hash);
  for (uint32_t probe = 0; probe <= self->index.max_probe; ++probe) {
    uint32_t slot_idx = self->index.cells[cell_idx];
    if (slot_idx != LRU_NIL && self->slots[slot_idx].hash == hash &&
        self->ops.eq(self->slots[slot_idx].value, probe_key, self->ops.user))
      return cell_idx;
    cell_idx = (cell_idx + 1u) & self->index.mask;
  }
  return LRU_NIL;
}

// --- eviction -------------------------------------------------------------

// Free slot `slot_idx`'s entry: erase from index, unlink from LRU, return
// slot to the free list, destroy the value. Caller must ensure
// refcount == 0.
static void
evict_slot(struct lru* self, uint32_t slot_idx)
{
  uint32_t cell_idx = index_locate(self, slot_idx);
  if (cell_idx != LRU_NIL)
    index_erase(self, cell_idx);

  struct lru_entry* entry = &self->slots[slot_idx];
  void* evicted_value = entry->value;
  list_unlink(&entry->link);
  slot_clear(self, slot_idx);
  freelist_push(self, slot_idx);
  self->size--;
  self->counters.evictions++;
  self->ops.destroy(evicted_value, self->ops.user);
}

// Walk the LRU list from tail toward MRU looking for an unpinned entry.
// Returns 0 on success (one entry evicted), -1 if every live entry is
// pinned.
static int
evict_lru_tail(struct lru* self)
{
  for (struct lru_list* node = self->lru_order.prev; node != &self->lru_order;
       node = node->prev) {
    struct lru_entry* entry = link_to_entry(node);
    // Racing read: a worker may drop refcount to 0 just after we observe
    // > 0, leaving an evictable entry unevicted this pass — benign, the
    // next eviction will catch it.
    if (atomic_load_explicit(&entry->refcount, memory_order_acquire) == 0) {
      evict_slot(self, entry_to_slot_idx(self, entry));
      return 0;
    }
  }
  return -1;
}

// --- public API -----------------------------------------------------------

struct lru*
lru_create(uint32_t capacity, uint32_t max_probe, const struct lru_ops* ops)
{
  struct lru* self = NULL;
  CHECK_SILENT(Error, capacity > 0);
  CHECK_SILENT(Error, max_probe > 0);
  CHECK_SILENT(Error, ops && ops->eq && ops->destroy);

  self = (struct lru*)calloc(1, sizeof(*self));
  CHECK(Error, self);

  uint32_t n_cells = next_pow2_ge(2u * capacity);

  *self = (struct lru){
    .ops = *ops,
    .capacity = capacity,
    .index = {
      .n_cells = n_cells,
      .mask = n_cells - 1u,
      .max_probe = max_probe,
    },
  };
  list_init(&self->lru_order);
  list_init(&self->freelist);

  self->slots = (struct lru_entry*)calloc(capacity, sizeof(*self->slots));
  CHECK(Error, self->slots);
  self->index.cells = (uint32_t*)malloc(n_cells * sizeof(*self->index.cells));
  CHECK(Error, self->index.cells);

  for (uint32_t i = 0; i < n_cells; ++i)
    self->index.cells[i] = LRU_NIL;

  // Seed the freelist with every slot. Push order doesn't matter — the
  // freelist is unordered.
  for (uint32_t i = 0; i < capacity; ++i) {
    list_init(&self->slots[i].link);
    list_push_front(&self->freelist, &self->slots[i].link);
  }

  return self;

Error:
  // lru_destroy walks lru_order; list_init above ran before any CHECK
  // that could land here, so the partial-state walk is well-defined.
  lru_destroy(self);
  return NULL;
}

void
lru_destroy(struct lru* self)
{
  if (!self)
    return;
  for (struct lru_list* node = self->lru_order.next; node != &self->lru_order;
       node = node->next) {
    self->ops.destroy(link_to_entry(node)->value, self->ops.user);
  }
  free(self->slots);
  free(self->index.cells);
  free(self);
}

struct lru_entry*
lru_get(struct lru* self, uint64_t hash, const void* probe_key)
{
  CHECK_SILENT(Miss, self);
  uint32_t cell_idx = index_lookup(self, hash, probe_key);
  CHECK_SILENT(Miss, cell_idx != LRU_NIL);

  uint32_t slot_idx = self->index.cells[cell_idx];
  list_promote(&self->lru_order, &self->slots[slot_idx].link);
  self->counters.hits++;
  return &self->slots[slot_idx];

Miss:
  if (self)
    self->counters.misses++;
  return NULL;
}

struct lru_entry*
lru_peek(struct lru* self, uint64_t hash, const void* probe_key)
{
  if (!self)
    return NULL;
  uint32_t cell_idx = index_lookup(self, hash, probe_key);
  if (cell_idx == LRU_NIL)
    return NULL;
  return &self->slots[self->index.cells[cell_idx]];
}

struct lru_entry*
lru_put(struct lru* self, uint64_t hash, const void* probe_key, void* value)
{
  // Preconditions: NULL self leaks `value` (no destroy callback to call);
  // NULL value is a caller bug. Both are documented in the header.
  CHECK(Fail, self);
  CHECK(Fail, value);

  // 1. Replace path: same key already present. Refuse if pinned.
  uint32_t cell_idx = index_lookup(self, hash, probe_key);
  if (cell_idx != LRU_NIL) {
    uint32_t slot_idx = self->index.cells[cell_idx];
    if (atomic_load_explicit(&self->slots[slot_idx].refcount,
                             memory_order_relaxed) > 0) {
      // Pinned entry can't be replaced — would invalidate the pin.
      self->counters.put_failures++;
      self->ops.destroy(value, self->ops.user);
      return NULL;
    }
    void* old_value = self->slots[slot_idx].value;
    self->slots[slot_idx].value = value;
    list_promote(&self->lru_order, &self->slots[slot_idx].link);
    self->counters.replacements++;
    self->ops.destroy(old_value, self->ops.user);
    return &self->slots[slot_idx];
  }

  // 2. Allocate a slot, evicting the LRU tail if the pool is empty.
  uint32_t new_slot = freelist_pop(self);
  while (new_slot == LRU_NIL) {
    if (evict_lru_tail(self) != 0) {
      self->counters.put_failures++;
      self->ops.destroy(value, self->ops.user);
      return NULL;
    }
    new_slot = freelist_pop(self);
  }
  self->slots[new_slot].hash = hash;
  self->slots[new_slot].value = value;
  atomic_store_explicit(
    &self->slots[new_slot].refcount, 0, memory_order_relaxed);

  // 3. Robin-hood insert. On cap-exceed, evict and retry. The orphan
  //    is whoever ended up unplaced — could be new_slot or a previously-
  //    stored slot.
  for (;;) {
    uint32_t orphan_slot = index_insert_robin(self, new_slot);
    if (orphan_slot == LRU_NIL)
      break;
    if (orphan_slot == new_slot) {
      // Our value never landed. Evict to free a slot in our chain, retry.
      // If eviction itself fails (everything pinned), give up.
      if (evict_lru_tail(self) != 0) {
        // Fallback: destroy the new value and return our slot.
        void* orphan_value = self->slots[new_slot].value;
        slot_clear(self, new_slot);
        freelist_push(self, new_slot);
        self->counters.put_failures++;
        self->ops.destroy(orphan_value, self->ops.user);
        return NULL;
      }
      continue;
    }
    // Orphan is a previously-existing entry. Evict it (destroy value,
    // unlink from LRU, free slot). Our new slot is now in the index.
    void* orphan_value = self->slots[orphan_slot].value;
    list_unlink(&self->slots[orphan_slot].link);
    slot_clear(self, orphan_slot);
    freelist_push(self, orphan_slot);
    self->size--;
    self->counters.evictions++;
    self->ops.destroy(orphan_value, self->ops.user);
    break;
  }

  // 4. Link new slot at MRU.
  list_push_front(&self->lru_order, &self->slots[new_slot].link);
  self->size++;
  self->counters.insertions++;
  return &self->slots[new_slot];

Fail:
  return NULL;
}

void*
lru_entry_value(const struct lru_entry* entry)
{
  return entry ? entry->value : NULL;
}

// Acquire runs under the caller's index mutex (see store_fs_acquire); the
// mutex serializes it against eviction's pinned-check, so relaxed suffices.
void
lru_entry_acquire(struct lru_entry* entry)
{
  if (entry)
    atomic_fetch_add_explicit(&entry->refcount, 1u, memory_order_relaxed);
}

// Release is lock-free (hot path from worker IO completion). acq_rel
// publishes any prior writes performed while the pin was held, so a later
// eviction reading refcount == 0 sees them happen-before.
void
lru_entry_release(struct lru_entry* entry)
{
  if (!entry)
    return;
  uint32_t prev =
    atomic_fetch_sub_explicit(&entry->refcount, 1u, memory_order_acq_rel);
  // Catches double-release on the lock-free path before it wraps.
  assert(prev > 0u);
  (void)prev;
}

void
lru_stats_get(const struct lru* self, struct lru_stats* out)
{
  CHECK(Fail, out);
  CHECK_SILENT(Fail, self);
  uint32_t n_pinned = 0;
  for (const struct lru_list* node = self->lru_order.next;
       node != &self->lru_order;
       node = node->next) {
    if (atomic_load_explicit(&link_to_entry_const(node)->refcount,
                             memory_order_relaxed) > 0)
      ++n_pinned;
  }
  *out = (struct lru_stats){
    .counters = self->counters,
    .size = self->size,
    .capacity = self->capacity,
    .pinned = n_pinned,
    .max_probe_observed = self->index.max_probe_observed,
  };
  return;
Fail:
  if (out)
    *out = (struct lru_stats){ 0 };
}
