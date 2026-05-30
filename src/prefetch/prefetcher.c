#include "prefetch/prefetcher.h"

#include "log/log.h"
#include "lookahead/lookahead.h"
#include "platform/platform.h"
#include "prefetch/chunk_layout.h"
#include "prefetch/shard_index.h"
#include "util/hash.h"
#include "util/prelude.h"
#include "zarr/sample_shard_iterator.h"
#include "zarr/zarr_metadata.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

struct prefetcher;
struct prefetcher_slot;

struct prefetcher_state
{
  const char* name;
  void (*advance)(struct prefetcher*, struct prefetcher_slot*);
};

static void
advance_from_meta(struct prefetcher* p, struct prefetcher_slot* s);
static void
advance_from_shard(struct prefetcher* p, struct prefetcher_slot* s);
static void
advance_from_layout(struct prefetcher* p, struct prefetcher_slot* s);

#define DEFINE_PREFETCHER_STATE(id, advance_fn)                                \
  static const struct prefetcher_state state_##id = { .name = #id,             \
                                                      .advance = advance_fn }

DEFINE_PREFETCHER_STATE(free, NULL);
DEFINE_PREFETCHER_STATE(pending_meta, advance_from_meta);
DEFINE_PREFETCHER_STATE(pending_shards, advance_from_shard);
DEFINE_PREFETCHER_STATE(pending_layout, advance_from_layout);
DEFINE_PREFETCHER_STATE(ready, NULL);
DEFINE_PREFETCHER_STATE(error, NULL);

struct prefetcher_slot
{
  const struct prefetcher_state* state;
  int err_code;
  char* uri;
  struct damacy_aabb aabb;
  uint64_t sample_seq;
  struct prefetch_gate* gate;
  struct prefetch_handle h_meta;
  struct prefetch_handle* h_shards;
  uint64_t* shard_coords; // flat [n_shards][rank]
  uint32_t n_shards;
  struct prefetch_handle h_layout;
};

struct prefetcher_owner_entry
{
  uint64_t owner_id;
  struct prefetch_gate gate;
  uint32_t refcount;
  int in_use;
};

struct prefetcher
{
  struct damacy_lookahead* lookahead;
  struct prefetch_cache* array_meta_cache;
  struct prefetch_cache* shard_index_cache;
  struct prefetch_cache* chunk_layout_cache;

  struct prefetcher_slot* slots;
  uint32_t capacity;

  struct prefetcher_owner_entry* owners;
  uint32_t owner_capacity;

  struct platform_mutex* lock;
  struct platform_thread* worker;
  _Atomic int stop;
  _Atomic int started;

  // Closes the drain TOCTOU between lookahead_pop and admit_locked.
  _Atomic uint32_t in_transit;

  uint64_t submitted;
  uint64_t ready;
  uint64_t errored;
  uint64_t next_consume_seq;
};

static struct prefetcher_owner_entry*
owner_lookup_locked(struct prefetcher* p, uint64_t owner_id)
{
  for (uint32_t i = 0; i < p->owner_capacity; ++i)
    if (p->owners[i].in_use && p->owners[i].owner_id == owner_id)
      return &p->owners[i];
  return NULL;
}

// Caller must pair with a refcount drop (owner_unref_locked); a failed admit
// converts the slot to ERROR which still holds the ref until pop.
static struct prefetcher_owner_entry*
owner_get_or_create_locked(struct prefetcher* p, uint64_t owner_id)
{
  struct prefetcher_owner_entry* e = owner_lookup_locked(p, owner_id);
  if (e) {
    e->refcount++;
    return e;
  }
  for (uint32_t i = 0; i < p->owner_capacity; ++i) {
    if (!p->owners[i].in_use) {
      e = &p->owners[i];
      e->owner_id = owner_id;
      e->in_use = 1;
      e->refcount = 1;
      prefetch_gate_init(&e->gate);
      return e;
    }
  }
  return NULL;
}

// Reuse is gated on gate.pending: cache-side waiters reference the gate
// independently of refcount and would alias it on the next admission.
static int
owner_try_free_locked(struct prefetcher_owner_entry* e)
{
  if (e->refcount != 0 || prefetch_gate_pending(&e->gate) != 0)
    return 0;
  e->in_use = 0;
  return 1;
}

static void
owner_unref_entry_locked(struct prefetcher_owner_entry* e)
{
  if (e->refcount > 0)
    e->refcount--;
  owner_try_free_locked(e);
}

static void
owner_unref_locked(struct prefetcher* p, uint64_t owner_id)
{
  struct prefetcher_owner_entry* e = owner_lookup_locked(p, owner_id);
  if (e)
    owner_unref_entry_locked(e);
}

static void
owner_sweep_locked(struct prefetcher* p)
{
  for (uint32_t i = 0; i < p->owner_capacity; ++i) {
    struct prefetcher_owner_entry* e = &p->owners[i];
    if (e->in_use && e->refcount == 0)
      owner_try_free_locked(e);
  }
}

static int
slot_free(const struct prefetcher_slot* s)
{
  return !s->state || s->state == &state_free;
}

static int
slot_active(const struct prefetcher_slot* s)
{
  return s->state && s->state->advance != NULL;
}

static int
slot_terminal(const struct prefetcher_slot* s)
{
  return s->state == &state_ready || s->state == &state_error;
}

static int
slot_failed(const struct prefetcher_slot* s)
{
  return s->state == &state_error;
}

static void
slot_set_state(struct prefetcher_slot* s, const struct prefetcher_state* state)
{
  s->state = state;
}

static void
slot_reset_free(struct prefetcher_slot* s)
{
  *s = (struct prefetcher_slot){ .state = &state_free };
}

static enum prefetcher_result
slot_result(const struct prefetcher_slot* s)
{
  return slot_failed(s) ? PREFETCHER_RESULT_ERROR : PREFETCHER_RESULT_READY;
}

static void
slot_fail(struct prefetcher* p, struct prefetcher_slot* s, int err_code)
{
  if (s->gate)
    prefetch_gate_set_error(s->gate);
  slot_set_state(s, &state_error);
  s->err_code = err_code ? err_code : DAMACY_INVAL;
  p->errored++;
}

static void
slot_mark_ready(struct prefetcher* p, struct prefetcher_slot* s)
{
  slot_set_state(s, &state_ready);
  p->ready++;
}

static struct prefetch_handle
request_chunk_layout(struct prefetcher* p, struct prefetcher_slot* s)
{
  struct chunk_layout_key key = {
    .uri = s->uri,
    .rank = s->aabb.rank,
    .n_shards = s->n_shards,
    .shard_coords = s->shard_coords,
    .h_shards = s->h_shards,
  };
  return prefetch_cache_request(p->chunk_layout_cache,
                                chunk_layout_key_hash(&key),
                                &key,
                                s->sample_seq,
                                s->gate);
}

static void
advance_from_meta(struct prefetcher* p, struct prefetcher_slot* s)
{
  int err = 0;
  const void* value = NULL;
  enum prefetch_state st =
    prefetch_cache_query(p->array_meta_cache, s->h_meta, &value, &err);
  if (st == PREFETCH_STATE_PENDING)
    return;
  if (st == PREFETCH_STATE_ERROR)
    goto Bad;
  const struct zarr_metadata* meta = (const struct zarr_metadata*)value;
  CHECK(Bad, meta);

  struct sample_shard_iterator it;
  CHECK(Bad, sample_shard_iterator_init(&it, meta, &s->aabb) == 0);

  uint64_t n = 1;
  for (uint8_t d = 0; d < it.rank; ++d)
    n *= (it.shard_end[d] - it.shard_beg[d]);

  if (n == 0) {
    s->n_shards = 0;
    s->h_layout = request_chunk_layout(p, s);
    if (!prefetch_handle_valid(s->h_layout)) {
      err = DAMACY_OOM;
      goto Bad;
    }
    slot_set_state(s, &state_pending_layout);
    return;
  }

  s->h_shards =
    (struct prefetch_handle*)calloc((size_t)n, sizeof(*s->h_shards));
  if (!s->h_shards) {
    err = DAMACY_OOM;
    goto Bad;
  }
  s->shard_coords =
    (uint64_t*)calloc((size_t)n * meta->rank, sizeof(*s->shard_coords));
  if (!s->shard_coords) {
    err = DAMACY_OOM;
    goto Bad;
  }

  struct shard_index_key probe = { .uri = s->uri, .rank = meta->rank };
  uint32_t i = 0;
  while (sample_shard_iterator_next(&it, probe.shard_coord)) {
    memcpy(&s->shard_coords[(size_t)i * meta->rank],
           probe.shard_coord,
           (size_t)meta->rank * sizeof(uint64_t));
    s->h_shards[i] = prefetch_cache_request(p->shard_index_cache,
                                            shard_index_key_hash(&probe),
                                            &probe,
                                            s->sample_seq,
                                            s->gate);
    if (!prefetch_handle_valid(s->h_shards[i])) {
      // The K successful requests already registered the gate as a waiter
      // (for PENDING/new slots). Their cache workers will dec_pending when
      // they resolve. owner_try_free_locked gates the entry on
      // gate_pending == 0, so recycling waits naturally.
      s->n_shards = i;
      err = DAMACY_OOM;
      goto Bad;
    }
    i++;
  }
  s->n_shards = (uint32_t)n;
  slot_set_state(s, &state_pending_shards);
  return;
Bad:
  slot_fail(p, s, err);
}

static void
advance_from_shard(struct prefetcher* p, struct prefetcher_slot* s)
{
  int err = 0;
  for (uint32_t i = 0; i < s->n_shards; ++i) {
    int per_shard_err = 0;
    enum prefetch_state st = prefetch_cache_query(
      p->shard_index_cache, s->h_shards[i], NULL, &per_shard_err);
    if (st == PREFETCH_STATE_PENDING)
      return;
    if (st == PREFETCH_STATE_ERROR && per_shard_err != DAMACY_NOTFOUND) {
      // Missing shards (NOTFOUND) are normal — the planner emits fill.
      // Other errors fail the whole sample.
      err = per_shard_err;
      goto Bad;
    }
  }

  s->h_layout = request_chunk_layout(p, s);
  if (!prefetch_handle_valid(s->h_layout)) {
    err = DAMACY_OOM;
    goto Bad;
  }
  slot_set_state(s, &state_pending_layout);
  return;
Bad:
  slot_fail(p, s, err);
}

static void
advance_from_layout(struct prefetcher* p, struct prefetcher_slot* s)
{
  int err = 0;
  enum prefetch_state st =
    prefetch_cache_query(p->chunk_layout_cache, s->h_layout, NULL, &err);
  if (st == PREFETCH_STATE_PENDING)
    return;
  if (st == PREFETCH_STATE_ERROR)
    goto Bad;
  slot_mark_ready(p, s);
  return;
Bad:
  slot_fail(p, s, err);
}

// Lock order: p->lock outermost. advance_all may take cache locks inside;
// fetch workers (running on the io_queue) must not touch p->lock.
static void
advance_all(struct prefetcher* p)
{
  for (uint32_t i = 0; i < p->capacity; ++i) {
    struct prefetcher_slot* s = &p->slots[i];
    if (slot_active(s))
      s->state->advance(p, s);
  }
}

static void
scan_slots_locked(struct prefetcher* p,
                  struct prefetcher_slot** out_free,
                  int* out_has_in_flight)
{
  *out_free = NULL;
  *out_has_in_flight = 0;
  for (uint32_t i = 0; i < p->capacity; ++i) {
    struct prefetcher_slot* s = &p->slots[i];
    if (!*out_free && slot_free(s))
      *out_free = s;
    else if (slot_active(s))
      *out_has_in_flight = 1;
  }
}

// URI ownership transfers into the slot (and on to prefetcher_ready).
static void
emit_error_slot_locked(struct prefetcher* p,
                       struct prefetcher_slot* slot,
                       struct damacy_sample_slot* popped,
                       struct prefetch_gate* gate,
                       int err_code)
{
  *slot = (struct prefetcher_slot){
    .uri = popped->uri,
    .aabb = popped->aabb,
    .sample_seq = popped->sample_seq,
    .gate = gate,
  };
  slot_fail(p, slot, err_code);
}

static void
admit_locked(struct prefetcher* p,
             struct prefetcher_slot* slot,
             struct damacy_sample_slot* popped)
{
  CHECK(Drop, slot);

  struct prefetcher_owner_entry* owner =
    owner_get_or_create_locked(p, popped->sample_seq);
  if (!owner) {
    // gate=NULL: no owner entry allocated. The ERROR slot still preserves
    // sample_seq ordering for the ready-prefix consumer.
    emit_error_slot_locked(p, slot, popped, NULL, DAMACY_OOM);
    return;
  }
  struct prefetch_gate* gate = &owner->gate;
  struct prefetch_handle h = prefetch_cache_request(p->array_meta_cache,
                                                    hash_fnv1a_str(popped->uri),
                                                    popped->uri,
                                                    popped->sample_seq,
                                                    gate);
  if (!prefetch_handle_valid(h)) {
    // ERROR slot retains the owner ref; pop releases it like a normal slot.
    emit_error_slot_locked(p, slot, popped, gate, DAMACY_OOM);
    return;
  }
  *slot = (struct prefetcher_slot){
    .uri = popped->uri,
    .aabb = popped->aabb,
    .sample_seq = popped->sample_seq,
    .gate = gate,
    .h_meta = h,
  };
  slot_set_state(slot, &state_pending_meta);
  p->submitted++;
  return;
Drop:
  free(popped->uri);
  p->errored++;
}

static void
worker_fn(void* arg)
{
  struct prefetcher* p = (struct prefetcher*)arg;
  while (!atomic_load_explicit(&p->stop, memory_order_acquire)) {
    platform_mutex_lock(p->lock);
    advance_all(p);
    owner_sweep_locked(p);
    struct prefetcher_slot* slot;
    int has_in_flight;
    scan_slots_locked(p, &slot, &has_in_flight);
    platform_mutex_unlock(p->lock);

    struct damacy_sample_slot popped = { 0 };
    int popped_ok = 0;
    // Timed wait when in-flight work needs periodic state advance; pure block
    // when admission is the only thing keeping us busy.
    if (slot && !has_in_flight)
      popped_ok = lookahead_pop_blocking(p->lookahead, &popped);
    else if (slot)
      popped_ok = lookahead_pop_blocking_timeout(p->lookahead, &popped, 1);
    else
      // TODO(perf): replace polling with a condvar signaled from
      // pop_terminal_slot_locked when a slot frees up.
      platform_sleep_ns(1000000);

    if (popped_ok) {
      atomic_fetch_add_explicit(&p->in_transit, 1, memory_order_acq_rel);
      platform_mutex_lock(p->lock);
      // scan_slots_locked only signals popped_ok when a FREE slot exists.
      admit_locked(p, slot, &popped);
      atomic_fetch_sub_explicit(&p->in_transit, 1, memory_order_acq_rel);
      platform_mutex_unlock(p->lock);
    }
  }
}

struct prefetcher*
prefetcher_create(const struct prefetcher_config* cfg)
{
  struct prefetcher* self = NULL;
  CHECK(Error, cfg);
  CHECK(Error, cfg->capacity > 0);
  CHECK(Error, cfg->owner_capacity > 0);
  CHECK(Error, cfg->lookahead);
  CHECK(Error, cfg->array_meta_cache);
  CHECK(Error, cfg->shard_index_cache);
  CHECK(Error, cfg->chunk_layout_cache);

  self = (struct prefetcher*)malloc(sizeof(*self));
  CHECK(Error, self);
  *self = (struct prefetcher){
    .lookahead = cfg->lookahead,
    .array_meta_cache = cfg->array_meta_cache,
    .shard_index_cache = cfg->shard_index_cache,
    .chunk_layout_cache = cfg->chunk_layout_cache,
    .capacity = cfg->capacity,
    .owner_capacity = cfg->owner_capacity,
  };
  self->slots =
    (struct prefetcher_slot*)calloc(cfg->capacity, sizeof(*self->slots));
  CHECK(Error, self->slots);
  for (uint32_t i = 0; i < cfg->capacity; ++i)
    slot_reset_free(&self->slots[i]);
  self->owners = (struct prefetcher_owner_entry*)calloc(cfg->owner_capacity,
                                                        sizeof(*self->owners));
  CHECK(Error, self->owners);
  self->lock = platform_mutex_new();
  CHECK(Error, self->lock);
  return self;

Error:
  prefetcher_destroy(self);
  return NULL;
}

void
prefetcher_destroy(struct prefetcher* self)
{
  if (!self)
    return;
  prefetcher_stop(self);
  if (self->slots) {
    for (uint32_t i = 0; i < self->capacity; ++i) {
      free(self->slots[i].uri);
      free(self->slots[i].h_shards);
      free(self->slots[i].shard_coords);
    }
    free(self->slots);
  }
  free(self->owners);
  platform_mutex_free(self->lock);
  free(self);
}

int
prefetcher_start(struct prefetcher* self)
{
  CHECK(Bad, self);
  int expected = 0;
  if (!atomic_compare_exchange_strong(&self->started, &expected, 1))
    return 0;
  self->worker = platform_thread_start(worker_fn, self);
  CHECK(Bad, self->worker);
  return 0;
Bad:
  return 1;
}

void
prefetcher_stop(struct prefetcher* self)
{
  if (!self)
    return;
  int was_started =
    atomic_exchange_explicit(&self->started, 0, memory_order_acq_rel);
  if (!was_started)
    return;
  atomic_store_explicit(&self->stop, 1, memory_order_release);
  lookahead_signal_stop(self->lookahead);
  if (self->worker) {
    platform_thread_join(self->worker);
    self->worker = NULL;
  }
}

enum damacy_status
prefetcher_drain(struct prefetcher* self)
{
  CHECK(Bad, self);
  for (;;) {
    if (lookahead_size(self->lookahead) == 0 &&
        atomic_load_explicit(&self->in_transit, memory_order_acquire) == 0) {
      int active = 0;
      platform_mutex_lock(self->lock);
      for (uint32_t i = 0; i < self->capacity; ++i) {
        if (slot_active(&self->slots[i])) {
          active = 1;
          break;
        }
      }
      platform_mutex_unlock(self->lock);
      if (!active)
        return DAMACY_OK;
    }
    platform_sleep_ns(500000);
  }
Bad:
  return DAMACY_INVAL;
}

static int
find_terminal_sample_locked(struct prefetcher* self, uint64_t sample_seq)
{
  for (uint32_t i = 0; i < self->capacity; ++i) {
    struct prefetcher_slot* s = &self->slots[i];
    if (slot_terminal(s) && s->sample_seq == sample_seq)
      return (int)i;
  }
  return -1;
}

static int
pop_terminal_slot_idx_locked(struct prefetcher* self,
                             uint32_t slot_idx,
                             struct prefetcher_ready* out)
{
  if (slot_idx >= self->capacity || !slot_terminal(&self->slots[slot_idx]))
    return 0;

  struct prefetcher_slot* s = &self->slots[slot_idx];
  *out = (struct prefetcher_ready){
    .result = slot_result(s),
    .err_code = s->err_code,
    .uri = s->uri,
    .aabb = s->aabb,
    .sample_seq = s->sample_seq,
    .h_meta = s->h_meta,
    .h_shards = s->h_shards,
    .n_shards = s->n_shards,
    .h_layout = s->h_layout,
  };
  uint64_t owner_id = s->sample_seq;
  free(s->shard_coords);
  slot_reset_free(s);
  owner_unref_locked(self, owner_id);
  return 1;
}

static uint32_t
ready_prefix_count_locked(struct prefetcher* self)
{
  uint32_t n = 0;
  for (;;) {
    int slot_idx =
      find_terminal_sample_locked(self, self->next_consume_seq + n);
    if (slot_idx < 0)
      break;
    n++;
  }
  return n;
}

int
prefetcher_pop_ready(struct prefetcher* self, struct prefetcher_ready* out)
{
  CHECK(Bad, self);
  CHECK(Bad, out);
  *out = (struct prefetcher_ready){ 0 };
  platform_mutex_lock(self->lock);
  int slot_idx = find_terminal_sample_locked(self, self->next_consume_seq);
  int found = 0;
  if (slot_idx >= 0) {
    found = pop_terminal_slot_idx_locked(self, (uint32_t)slot_idx, out);
    if (found)
      self->next_consume_seq++;
  }
  platform_mutex_unlock(self->lock);
  return found;
Bad:
  return 0;
}

int
prefetcher_take_ready_wave(struct prefetcher* self,
                           uint32_t max_samples,
                           struct prefetcher_wave_ticket* ticket,
                           struct prefetcher_ready* out)
{
  CHECK(Bad, self);
  CHECK(Bad, ticket);
  CHECK(Bad, out);
  CHECK(Bad, max_samples > 0);
  for (uint32_t i = 0; i < max_samples; ++i)
    out[i] = (struct prefetcher_ready){ 0 };

  platform_mutex_lock(self->lock);
  uint32_t n = ready_prefix_count_locked(self);
  if (n > max_samples)
    n = max_samples;
  if (n == 0) {
    platform_mutex_unlock(self->lock);
    *ticket = (struct prefetcher_wave_ticket){ 0 };
    return 0;
  }
  *ticket = (struct prefetcher_wave_ticket){
    .sample_seq_begin = self->next_consume_seq,
    .n_samples = n,
  };
  for (uint32_t i = 0; i < n; ++i) {
    int slot_idx = find_terminal_sample_locked(self, self->next_consume_seq);
    if (slot_idx < 0 ||
        !pop_terminal_slot_idx_locked(self, (uint32_t)slot_idx, &out[i])) {
      platform_mutex_unlock(self->lock);
      for (uint32_t j = 0; j < i; ++j)
        prefetcher_ready_free(&out[j]);
      *ticket = (struct prefetcher_wave_ticket){ 0 };
      return 0;
    }
    self->next_consume_seq++;
  }
  platform_mutex_unlock(self->lock);
  return 1;
Bad:
  return 0;
}

uint32_t
prefetcher_ready_prefix_count(struct prefetcher* self)
{
  CHECK(Bad, self);
  platform_mutex_lock(self->lock);
  uint32_t n = ready_prefix_count_locked(self);
  platform_mutex_unlock(self->lock);
  return n;
Bad:
  return 0;
}

uint32_t
prefetcher_in_flight(struct prefetcher* self)
{
  CHECK(Bad, self);
  // Include in_transit (popped from lookahead, not yet admitted) so a
  // 0 return reliably means "no further state transitions will happen
  // from already-queued work".
  uint32_t n = atomic_load_explicit(&self->in_transit, memory_order_acquire);
  platform_mutex_lock(self->lock);
  for (uint32_t i = 0; i < self->capacity; ++i)
    if (slot_active(&self->slots[i]))
      n++;
  platform_mutex_unlock(self->lock);
  return n;
Bad:
  return 0;
}

// ERROR slots count as "ready" so pop doesn't latch AGAIN before the
// scheduler drains the failure into damacy::failed_status.
int
prefetcher_has_ready(struct prefetcher* self)
{
  CHECK(Bad, self);
  platform_mutex_lock(self->lock);
  int found = find_terminal_sample_locked(self, self->next_consume_seq) >= 0;
  platform_mutex_unlock(self->lock);
  return found;
Bad:
  return 0;
}

void
prefetcher_ready_free(struct prefetcher_ready* r)
{
  if (!r)
    return;
  free(r->uri);
  free(r->h_shards);
  *r = (struct prefetcher_ready){ 0 };
}

const struct prefetch_gate*
prefetcher_sample_gate(struct prefetcher* self, uint64_t sample_seq)
{
  CHECK(Bad, self);
  platform_mutex_lock(self->lock);
  struct prefetcher_owner_entry* e = owner_lookup_locked(self, sample_seq);
  const struct prefetch_gate* g = e ? &e->gate : NULL;
  platform_mutex_unlock(self->lock);
  return g;
Bad:
  return NULL;
}

void
prefetcher_advance_watermark(struct prefetcher* self, uint64_t watermark)
{
  if (!self)
    return;
  prefetch_cache_advance_watermark(self->array_meta_cache, watermark);
  prefetch_cache_advance_watermark(self->shard_index_cache, watermark);
  prefetch_cache_advance_watermark(self->chunk_layout_cache, watermark);
}

void
prefetcher_stats_get(const struct prefetcher* self,
                     struct prefetcher_stats* out)
{
  CHECK(End, out);
  *out = (struct prefetcher_stats){ 0 };
  CHECK(End, self);
  platform_mutex_lock(self->lock);
  out->submitted = self->submitted;
  out->ready = self->ready;
  out->errored = self->errored;
  uint32_t in_flight = 0;
  for (uint32_t i = 0; i < self->capacity; ++i)
    if (slot_active(&self->slots[i]))
      in_flight++;
  out->in_flight = in_flight;
  platform_mutex_unlock(self->lock);
End:
  return;
}
