#include "prefetch/prefetch_cache.h"

#include "damacy.h"
#include "log/log.h"
#include "platform/platform.h"
#include "util/lru.h"
#include "util/prelude.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#define GATE_ERROR_BIT (UINT64_C(1) << 63)
#define GATE_PENDING_MASK (~GATE_ERROR_BIT)
#define ACTIVE_IDX_UNLINKED UINT32_MAX
#define WAITERS_INLINE 2

struct prefetch_slot
{
  uint64_t key_hash;
  void* key;
  void* value;
  enum prefetch_state state;
  int err;
  uint64_t min_owner_id;
  uint64_t max_owner_id;
  uint32_t generation;

  // Inline storage for the common case; heap takes over on overflow.
  // The first WAITERS_INLINE registers cannot fail.
  struct prefetch_gate* waiters_inline[WAITERS_INLINE];
  struct prefetch_gate** waiters_heap;
  uint32_t n_waiters;
  uint32_t cap_heap;

  struct lru_entry* lru_ent;
  uint32_t active_idx;
  struct prefetch_cache* cache;
};

struct prefetch_cache
{
  struct platform_mutex* lock;
  struct lru* idx;

  uint64_t watermark;
  uint32_t generation_counter;

  // Stable slot indices: eviction must not move a slot, or live handles to
  // siblings misroute.
  struct prefetch_slot** active;
  uint32_t* free_list;
  uint32_t n_free;
  uint32_t n_active;
  uint32_t capacity;

  uint64_t pending_count;
  uint64_t errored_count;

  const struct prefetch_ops* ops;
  struct prefetch_fetcher* fetcher;
  struct prefetch_executor* executor;
};

struct fetch_ctx
{
  struct prefetch_cache* cache;
  struct prefetch_slot* slot;
  uint32_t generation;
};

static void
prefetch_fetch_worker(void* ctx_);

// --- gate ----------------------------------------------------------------

int
prefetch_handle_valid(struct prefetch_handle h)
{
  return h.slot != 0 || h.generation != 0;
}

void
prefetch_gate_init(struct prefetch_gate* g)
{
  if (!g)
    return;
  atomic_store_explicit(&g->state, 0, memory_order_release);
}

int
prefetch_gate_is_ready(const struct prefetch_gate* g)
{
  if (!g)
    return 1;
  uint64_t s = atomic_load_explicit(&g->state, memory_order_acquire);
  return (s & GATE_PENDING_MASK) == 0;
}

int
prefetch_gate_has_error(const struct prefetch_gate* g)
{
  if (!g)
    return 0;
  uint64_t s = atomic_load_explicit(&g->state, memory_order_acquire);
  return (s & GATE_ERROR_BIT) != 0;
}

uint64_t
prefetch_gate_pending(const struct prefetch_gate* g)
{
  if (!g)
    return 0;
  uint64_t s = atomic_load_explicit(&g->state, memory_order_acquire);
  return s & GATE_PENDING_MASK;
}

static void
gate_inc_pending(struct prefetch_gate* g)
{
  if (!g)
    return;
  atomic_fetch_add_explicit(&g->state, 1, memory_order_acq_rel);
}

static void
prefetch_gate_dec_pending(struct prefetch_gate* g)
{
  if (!g)
    return;
  atomic_fetch_sub_explicit(&g->state, 1, memory_order_acq_rel);
}

void
prefetch_gate_set_error(struct prefetch_gate* g)
{
  if (!g)
    return;
  atomic_fetch_or_explicit(&g->state, GATE_ERROR_BIT, memory_order_acq_rel);
}

// --- slot ----------------------------------------------------------------

static int
slot_register_waiter(struct prefetch_slot* s, struct prefetch_gate* gate)
{
  if (!gate)
    return 0;
  if (s->n_waiters < WAITERS_INLINE) {
    s->waiters_inline[s->n_waiters++] = gate;
    return 0;
  }
  uint32_t heap_used = s->n_waiters - WAITERS_INLINE;
  if (heap_used == s->cap_heap) {
    uint32_t new_cap = s->cap_heap ? s->cap_heap * 2u : 4u;
    struct prefetch_gate** w =
      (struct prefetch_gate**)realloc(s->waiters_heap, new_cap * sizeof(*w));
    if (!w)
      return 1;
    s->waiters_heap = w;
    s->cap_heap = new_cap;
  }
  s->waiters_heap[heap_used] = gate;
  s->n_waiters++;
  return 0;
}

static struct prefetch_gate*
slot_waiter(const struct prefetch_slot* s, uint32_t i)
{
  if (i < WAITERS_INLINE)
    return s->waiters_inline[i];
  return s->waiters_heap[i - WAITERS_INLINE];
}

static void
slot_release_waiters(struct prefetch_slot* s, int errored)
{
  for (uint32_t i = 0; i < s->n_waiters; ++i) {
    if (errored)
      prefetch_gate_set_error(slot_waiter(s, i));
    prefetch_gate_dec_pending(slot_waiter(s, i));
  }
  s->n_waiters = 0;
}

// --- lru ops adapter -----------------------------------------------------

static int
lru_slot_eq(const void* slot_value, const void* probe_key, void* user)
{
  struct prefetch_cache* c = (struct prefetch_cache*)user;
  const struct prefetch_slot* s = (const struct prefetch_slot*)slot_value;
  return c->ops->key_eq(c->ops, s->key, probe_key);
}

static void
active_remove(struct prefetch_cache* c, struct prefetch_slot* s)
{
  uint32_t i = s->active_idx;
  c->active[i] = NULL;
  c->free_list[c->n_free++] = i;
  c->n_active--;
  s->active_idx = ACTIVE_IDX_UNLINKED;
}

static void
lru_slot_destroy(void* value, void* user)
{
  struct prefetch_cache* c = (struct prefetch_cache*)user;
  struct prefetch_slot* s = (struct prefetch_slot*)value;
  CHECK(End, s);
  if (s->active_idx != ACTIVE_IDX_UNLINKED) {
    active_remove(c, s);
    if (s->state == PREFETCH_STATE_PENDING)
      c->pending_count--;
    else if (s->state == PREFETCH_STATE_ERROR)
      c->errored_count--;
  }
  if (s->n_waiters) {
    // Waiters' gates have pending counts incremented; drop them as ERROR
    // before free, otherwise any prefetch_gate_is_ready waiter deadlocks.
    log_error("evicting slot with %u live waiters", s->n_waiters);
    slot_release_waiters(s, 1);
  }
  free(s->waiters_heap);
  if (s->key)
    c->ops->key_destroy(c->ops, s->key);
  if (s->value)
    c->ops->value_destroy(c->ops, s->value);
  free(s);
End:
  return;
}

// --- create / destroy ----------------------------------------------------

struct prefetch_cache*
prefetch_cache_create(const struct prefetch_cache_config* cfg)
{
  struct prefetch_cache* self = NULL;

  CHECK(Error, cfg);
  CHECK(Error, cfg->capacity > 0);
  CHECK(Error, cfg->ops);
  CHECK(Error, cfg->ops->key_eq);
  CHECK(Error, cfg->ops->key_clone);
  CHECK(Error, cfg->ops->key_destroy);
  CHECK(Error, cfg->fetcher);
  CHECK(Error, cfg->fetcher->fetch);
  CHECK(Error, cfg->executor);
  CHECK(Error, cfg->executor->post);

  self = (struct prefetch_cache*)malloc(sizeof(*self));
  CHECK(Error, self);
  *self = (struct prefetch_cache){
    .capacity = cfg->capacity,
    .ops = cfg->ops,
    .fetcher = cfg->fetcher,
    .executor = cfg->executor,
  };

  self->lock = platform_mutex_new();
  CHECK(Error, self->lock);

  self->active =
    (struct prefetch_slot**)calloc(cfg->capacity, sizeof(*self->active));
  CHECK(Error, self->active);
  self->free_list = (uint32_t*)malloc(cfg->capacity * sizeof(*self->free_list));
  CHECK(Error, self->free_list);
  for (uint32_t i = 0; i < cfg->capacity; ++i)
    self->free_list[i] = cfg->capacity - 1 - i;
  self->n_free = cfg->capacity;

  struct lru_ops lru_ops = {
    .eq = lru_slot_eq,
    .destroy = lru_slot_destroy,
    .user = self,
  };
  self->idx = lru_create(cfg->capacity, cfg->max_probe, &lru_ops);
  CHECK(Error, self->idx);

  return self;

Error:
  prefetch_cache_destroy(self);
  return NULL;
}

void
prefetch_cache_destroy(struct prefetch_cache* self)
{
  if (!self)
    return;
  // Caller must ensure no fetches are in flight. Release any pins
  // we still hold so lru_destroy can actually free.
  if (self->active) {
    for (uint32_t i = 0; i < self->capacity; ++i) {
      struct prefetch_slot* s = self->active[i];
      if (s && s->lru_ent) {
        lru_entry_release(s->lru_ent);
        s->lru_ent = NULL;
      }
    }
  }
  lru_destroy(self->idx);
  free(self->active);
  free(self->free_list);
  platform_mutex_free(self->lock);
  free(self);
}

// --- request -------------------------------------------------------------

static struct prefetch_handle
make_handle(uint32_t active_idx, uint32_t generation)
{
  return (struct prefetch_handle){ .slot = active_idx + 1,
                                   .generation = generation };
}

struct prefetch_handle
prefetch_cache_request(struct prefetch_cache* self,
                       uint64_t key_hash,
                       const void* key,
                       uint64_t owner_id,
                       struct prefetch_gate* gate)
{
  CHECK(Bad, self);
  CHECK(Bad, key);

  platform_mutex_lock(self->lock);

  struct lru_entry* hit = lru_get(self->idx, key_hash, key);
  if (hit) {
    struct prefetch_slot* s = (struct prefetch_slot*)lru_entry_value(hit);
    if (owner_id > s->max_owner_id)
      s->max_owner_id = owner_id;
    if (owner_id < s->min_owner_id)
      s->min_owner_id = owner_id;

    if (s->state == PREFETCH_STATE_PENDING) {
      gate_inc_pending(gate);
      if (slot_register_waiter(s, gate)) {
        prefetch_gate_dec_pending(gate);
        platform_mutex_unlock(self->lock);
        return PREFETCH_HANDLE_NONE;
      }
    } else if (s->state == PREFETCH_STATE_ERROR) {
      prefetch_gate_set_error(gate);
    }

    struct prefetch_handle h = make_handle(s->active_idx, s->generation);
    platform_mutex_unlock(self->lock);
    return h;
  }

  void* key_copy = self->ops->key_clone(self->ops, key);
  if (!key_copy) {
    platform_mutex_unlock(self->lock);
    return PREFETCH_HANDLE_NONE;
  }
  struct prefetch_slot* s = (struct prefetch_slot*)malloc(sizeof(*s));
  if (!s) {
    self->ops->key_destroy(self->ops, key_copy);
    platform_mutex_unlock(self->lock);
    return PREFETCH_HANDLE_NONE;
  }
  *s = (struct prefetch_slot){
    .key_hash = key_hash,
    .key = key_copy,
    .state = PREFETCH_STATE_PENDING,
    .min_owner_id = owner_id,
    .max_owner_id = owner_id,
    .generation = ++self->generation_counter,
    .cache = self,
    .active_idx = ACTIVE_IDX_UNLINKED,
  };

  struct lru_entry* ent = lru_put(self->idx, key_hash, key, s);
  if (!ent) {
    // lru_put has already called lru_slot_destroy on s.
    platform_mutex_unlock(self->lock);
    return PREFETCH_HANDLE_NONE;
  }
  s->lru_ent = ent;
  lru_entry_acquire_locked(ent);

  uint32_t idx = self->free_list[--self->n_free];
  s->active_idx = idx;
  self->active[idx] = s;
  self->n_active++;
  self->pending_count++;

  // First waiter uses inline storage; can't fail.
  gate_inc_pending(gate);
  (void)slot_register_waiter(s, gate);

  struct fetch_ctx* ctx = (struct fetch_ctx*)malloc(sizeof(*ctx));
  if (!ctx) {
    // Mark errored so consumers don't hang on a pending slot with no
    // fetch in flight.
    s->state = PREFETCH_STATE_ERROR;
    s->err = DAMACY_OOM;
    self->pending_count--;
    self->errored_count++;
    slot_release_waiters(s, 1);
    struct prefetch_handle h = make_handle(s->active_idx, s->generation);
    platform_mutex_unlock(self->lock);
    return h;
  }
  *ctx = (struct fetch_ctx){
    .cache = self,
    .slot = s,
    .generation = s->generation,
  };

  uint32_t handle_active_idx = s->active_idx;
  uint32_t handle_gen = s->generation;

  platform_mutex_unlock(self->lock);

  if (self->executor->post(self->executor, prefetch_fetch_worker, ctx, free)) {
    platform_mutex_lock(self->lock);
    if (s->generation == ctx->generation &&
        s->state == PREFETCH_STATE_PENDING) {
      s->state = PREFETCH_STATE_ERROR;
      s->err = DAMACY_AGAIN;
      self->pending_count--;
      self->errored_count++;
      slot_release_waiters(s, 1);
    }
    platform_mutex_unlock(self->lock);
    free(ctx);
  }

  return make_handle(handle_active_idx, handle_gen);

Bad:
  return PREFETCH_HANDLE_NONE;
}

// --- fetch worker --------------------------------------------------------

static void
prefetch_fetch_worker(void* ctx_)
{
  struct fetch_ctx* ctx = (struct fetch_ctx*)ctx_;
  struct prefetch_cache* c = ctx->cache;
  struct prefetch_slot* s = ctx->slot;

  void* value = NULL;
  int err = 0;
  int rc = c->fetcher->fetch(c->fetcher, s->key, &value, &err);

  platform_mutex_lock(c->lock);

  if (s->generation != ctx->generation) {
    // Slot was evicted; the fetch result is orphaned.
    if (rc == 0 && value)
      c->ops->value_destroy(c->ops, value);
    platform_mutex_unlock(c->lock);
    return;
  }

  if (rc == 0) {
    s->state = PREFETCH_STATE_READY;
    s->value = value;
    c->pending_count--;
    slot_release_waiters(s, 0);
  } else {
    s->state = PREFETCH_STATE_ERROR;
    s->err = err;
    c->pending_count--;
    c->errored_count++;
    slot_release_waiters(s, 1);
  }

  platform_mutex_unlock(c->lock);
}

// --- handle deref --------------------------------------------------------

static struct prefetch_slot*
resolve_handle(const struct prefetch_cache* c, struct prefetch_handle h)
{
  if (h.slot == 0)
    return NULL;
  uint32_t idx = h.slot - 1;
  if (idx >= c->capacity)
    return NULL;
  struct prefetch_slot* s = c->active[idx];
  if (!s || s->generation != h.generation)
    return NULL;
  return s;
}

const void*
prefetch_cache_try_get(const struct prefetch_cache* c, struct prefetch_handle h)
{
  CHECK(Bad, c);
  platform_mutex_lock(c->lock);
  const void* value = NULL;
  struct prefetch_slot* s = resolve_handle(c, h);
  if (s && s->state == PREFETCH_STATE_READY)
    value = s->value;
  platform_mutex_unlock(c->lock);
  return value;
Bad:
  return NULL;
}

const void*
prefetch_cache_peek(const struct prefetch_cache* c,
                    uint64_t key_hash,
                    const void* key)
{
  CHECK(Bad, c);
  CHECK(Bad, key);
  platform_mutex_lock(c->lock);
  const void* value = NULL;
  struct lru_entry* hit = lru_peek(c->idx, key_hash, key);
  if (hit) {
    const struct prefetch_slot* s =
      (const struct prefetch_slot*)lru_entry_value(hit);
    if (s->state == PREFETCH_STATE_READY)
      value = s->value;
  }
  platform_mutex_unlock(c->lock);
  return value;
Bad:
  return NULL;
}

enum prefetch_state
prefetch_cache_query(const struct prefetch_cache* c,
                     struct prefetch_handle h,
                     const void** out_value,
                     int* out_err)
{
  if (out_value)
    *out_value = NULL;
  if (out_err)
    *out_err = 0;
  CHECK(Bad, c);
  platform_mutex_lock(c->lock);
  enum prefetch_state st = PREFETCH_STATE_PENDING;
  struct prefetch_slot* s = resolve_handle(c, h);
  if (s) {
    st = s->state;
    if (out_value && st == PREFETCH_STATE_READY)
      *out_value = s->value;
    if (out_err && st == PREFETCH_STATE_ERROR)
      *out_err = s->err;
  }
  platform_mutex_unlock(c->lock);
  return st;
Bad:
  return PREFETCH_STATE_PENDING;
}

// --- watermark -----------------------------------------------------------

void
prefetch_cache_advance_watermark(struct prefetch_cache* self,
                                 uint64_t new_watermark)
{
  CHECK(End, self);
  platform_mutex_lock(self->lock);
  if (new_watermark <= self->watermark) {
    platform_mutex_unlock(self->lock);
    return;
  }
  self->watermark = new_watermark;
  // Release pins; lru defers eviction until the slots are needed.
  // PENDING slots keep their pin: prefetch_fetch_worker still holds a
  // raw pointer to the slot and dereferences it on completion. Releasing
  // here would let eviction free the slot mid-fetch.
  for (uint32_t i = 0; i < self->capacity; ++i) {
    struct prefetch_slot* s = self->active[i];
    if (s && s->state != PREFETCH_STATE_PENDING &&
        s->max_owner_id < new_watermark && s->lru_ent) {
      lru_entry_release(s->lru_ent);
      s->lru_ent = NULL;
    }
  }
  platform_mutex_unlock(self->lock);
End:
  return;
}

// --- stats ---------------------------------------------------------------

void
prefetch_cache_stats_get(const struct prefetch_cache* self,
                         struct prefetch_cache_stats* out)
{
  CHECK(End, out);
  memset(out, 0, sizeof(*out));
  CHECK(End, self);
  platform_mutex_lock(self->lock);
  struct lru_stats lru_st;
  lru_stats_get(self->idx, &lru_st);
  out->counters = lru_st.counters;
  out->size = lru_st.size;
  out->capacity = self->capacity;
  out->watermark = self->watermark;
  out->pending = self->pending_count;
  out->errored = self->errored_count;
  platform_mutex_unlock(self->lock);
End:
  return;
}
