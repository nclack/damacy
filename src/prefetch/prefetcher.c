#include "prefetch/prefetcher.h"

#include "log/log.h"
#include "lookahead/lookahead.h"
#include "platform/platform.h"
#include "prefetch/shard_index.h"
#include "util/hash.h"
#include "util/prelude.h"
#include "zarr/sample_shard_iterator.h"
#include "zarr/zarr_metadata.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

struct prefetcher_slot
{
  enum prefetcher_sample_state state;
  int err_code;
  char* uri;
  struct damacy_aabb aabb;
  uint64_t batch_id;
  // Monotonic admission sequence. pop returns the smallest admit_seq
  // among terminal slots so plan_reserve sees samples in push order.
  uint64_t admit_seq;
  struct prefetch_gate* gate;
  struct prefetch_handle h_meta;
  struct prefetch_handle* h_shards;
  uint32_t n_shards;
  struct prefetch_handle h_layout;
};

struct prefetcher_batch_entry
{
  uint64_t batch_id;
  struct prefetch_gate gate;
  uint32_t refcount;
  int release_pending;
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

  struct prefetcher_batch_entry* batches;
  uint32_t batch_capacity;

  struct platform_mutex* lock;
  struct platform_thread* worker;
  _Atomic int stop;
  _Atomic int started;

  // Closes the drain TOCTOU between lookahead_pop and admit_locked.
  _Atomic uint32_t in_transit;

  uint64_t submitted;
  uint64_t ready;
  uint64_t errored;
  uint64_t next_admit_seq;
};

static struct prefetcher_batch_entry*
batch_lookup_locked(struct prefetcher* p, uint64_t batch_id)
{
  for (uint32_t i = 0; i < p->batch_capacity; ++i)
    if (p->batches[i].in_use && p->batches[i].batch_id == batch_id)
      return &p->batches[i];
  return NULL;
}

// Caller must pair with a refcount drop (batch_unref_locked); a failed admit
// converts the slot to ERROR which still holds the ref until pop.
static struct prefetcher_batch_entry*
batch_get_or_create_locked(struct prefetcher* p, uint64_t batch_id)
{
  struct prefetcher_batch_entry* e = batch_lookup_locked(p, batch_id);
  if (e) {
    e->refcount++;
    return e;
  }
  for (uint32_t i = 0; i < p->batch_capacity; ++i) {
    if (!p->batches[i].in_use) {
      e = &p->batches[i];
      e->batch_id = batch_id;
      e->in_use = 1;
      e->refcount = 1;
      e->release_pending = 0;
      prefetch_gate_init(&e->gate);
      return e;
    }
  }
  return NULL;
}

// Reuse is gated on gate.pending: cache-side waiters reference the gate
// independently of refcount and would alias it on the next admission.
static int
batch_try_free_locked(struct prefetcher_batch_entry* e)
{
  if (e->refcount != 0 || prefetch_gate_pending(&e->gate) != 0)
    return 0;
  e->in_use = 0;
  return 1;
}

static void
batch_unref_entry_locked(struct prefetcher_batch_entry* e)
{
  if (e->refcount > 0)
    e->refcount--;
  if (e->release_pending)
    batch_try_free_locked(e);
}

static void
batch_unref_locked(struct prefetcher* p, uint64_t batch_id)
{
  struct prefetcher_batch_entry* e = batch_lookup_locked(p, batch_id);
  if (e)
    batch_unref_entry_locked(e);
}

static void
batch_sweep_locked(struct prefetcher* p)
{
  for (uint32_t i = 0; i < p->batch_capacity; ++i) {
    struct prefetcher_batch_entry* e = &p->batches[i];
    if (e->in_use && e->release_pending && e->refcount == 0)
      batch_try_free_locked(e);
  }
}

static int
slot_active(const struct prefetcher_slot* s)
{
  return s->state == PREFETCHER_PENDING_ARRAY_META ||
         s->state == PREFETCHER_PENDING_SHARD_INDEX ||
         s->state == PREFETCHER_PENDING_CHUNK_LAYOUT;
}

static void
fail_slot(struct prefetcher* p, struct prefetcher_slot* s, int err_code)
{
  prefetch_gate_set_error(s->gate);
  s->state = PREFETCHER_ERROR;
  s->err_code = err_code ? err_code : DAMACY_INVAL;
  p->errored++;
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
    s->h_layout = prefetch_cache_request(p->chunk_layout_cache,
                                         hash_fnv1a_str(s->uri),
                                         s->uri,
                                         s->batch_id,
                                         s->gate);
    if (!prefetch_handle_valid(s->h_layout)) {
      err = DAMACY_OOM;
      goto Bad;
    }
    s->state = PREFETCHER_PENDING_CHUNK_LAYOUT;
    return;
  }

  s->h_shards =
    (struct prefetch_handle*)calloc((size_t)n, sizeof(*s->h_shards));
  if (!s->h_shards) {
    err = DAMACY_OOM;
    goto Bad;
  }

  struct shard_index_key probe = { .uri = s->uri, .rank = meta->rank };
  uint32_t i = 0;
  while (sample_shard_iterator_next(&it, probe.shard_coord)) {
    s->h_shards[i] = prefetch_cache_request(p->shard_index_cache,
                                            shard_index_key_hash(&probe),
                                            &probe,
                                            s->batch_id,
                                            s->gate);
    if (!prefetch_handle_valid(s->h_shards[i])) {
      // The K successful requests already registered the gate as a waiter
      // (for PENDING/new slots). Their cache workers will dec_pending when
      // they resolve. batch_try_free_locked gates the entry on
      // gate_pending == 0, so recycling waits naturally.
      s->n_shards = i;
      err = DAMACY_OOM;
      goto Bad;
    }
    i++;
  }
  s->n_shards = (uint32_t)n;
  s->state = PREFETCHER_PENDING_SHARD_INDEX;
  return;
Bad:
  fail_slot(p, s, err);
}

static void
advance_from_shard(struct prefetcher* p, struct prefetcher_slot* s)
{
  int err = 0;
  for (uint32_t i = 0; i < s->n_shards; ++i) {
    enum prefetch_state st =
      prefetch_cache_query(p->shard_index_cache, s->h_shards[i], NULL, &err);
    if (st == PREFETCH_STATE_PENDING)
      return;
    if (st == PREFETCH_STATE_ERROR)
      goto Bad;
  }

  s->h_layout = prefetch_cache_request(p->chunk_layout_cache,
                                       hash_fnv1a_str(s->uri),
                                       s->uri,
                                       s->batch_id,
                                       s->gate);
  if (!prefetch_handle_valid(s->h_layout)) {
    err = DAMACY_OOM;
    goto Bad;
  }
  s->state = PREFETCHER_PENDING_CHUNK_LAYOUT;
  return;
Bad:
  fail_slot(p, s, err);
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
  s->state = PREFETCHER_READY;
  p->ready++;
  return;
Bad:
  fail_slot(p, s, err);
}

// Lock order: p->lock outermost. advance_all may take cache locks inside;
// fetch workers (running on the io_queue) must not touch p->lock.
static void
advance_all(struct prefetcher* p)
{
  for (uint32_t i = 0; i < p->capacity; ++i) {
    struct prefetcher_slot* s = &p->slots[i];
    switch (s->state) {
      case PREFETCHER_PENDING_ARRAY_META:
        advance_from_meta(p, s);
        break;
      case PREFETCHER_PENDING_SHARD_INDEX:
        advance_from_shard(p, s);
        break;
      case PREFETCHER_PENDING_CHUNK_LAYOUT:
        advance_from_layout(p, s);
        break;
      default:
        break;
    }
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
    if (!*out_free && s->state == PREFETCHER_FREE)
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
  prefetch_gate_set_error(gate);
  // Assign admit_seq so pop_terminal_slot_locked returns this ERROR slot
  // in push order; without this it lands at admit_seq=0 and pops first.
  *slot = (struct prefetcher_slot){
    .state = PREFETCHER_ERROR,
    .err_code = err_code,
    .uri = popped->uri,
    .aabb = popped->aabb,
    .batch_id = popped->batch_id,
    .admit_seq = p->next_admit_seq++,
    .gate = gate,
  };
  p->errored++;
}

static void
admit_locked(struct prefetcher* p,
             struct prefetcher_slot* slot,
             struct damacy_sample_slot* popped)
{
  CHECK(Drop, slot);

  struct prefetcher_batch_entry* be =
    batch_get_or_create_locked(p, popped->batch_id);
  if (!be) {
    // gate=NULL: no batch entry allocated; see prefetcher_batch_gate doc
    // for the saturation-error contract.
    emit_error_slot_locked(p, slot, popped, NULL, DAMACY_OOM);
    return;
  }
  struct prefetch_gate* gate = &be->gate;
  struct prefetch_handle h = prefetch_cache_request(p->array_meta_cache,
                                                    hash_fnv1a_str(popped->uri),
                                                    popped->uri,
                                                    popped->batch_id,
                                                    gate);
  if (!prefetch_handle_valid(h)) {
    // ERROR slot retains the batch ref; pop releases it like a normal slot.
    emit_error_slot_locked(p, slot, popped, gate, DAMACY_OOM);
    return;
  }
  *slot = (struct prefetcher_slot){
    .state = PREFETCHER_PENDING_ARRAY_META,
    .uri = popped->uri,
    .aabb = popped->aabb,
    .batch_id = popped->batch_id,
    .admit_seq = p->next_admit_seq++,
    .gate = gate,
    .h_meta = h,
  };
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
    batch_sweep_locked(p);
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
  CHECK(Error, cfg->batch_capacity > 0);
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
    .batch_capacity = cfg->batch_capacity,
  };
  self->slots =
    (struct prefetcher_slot*)calloc(cfg->capacity, sizeof(*self->slots));
  CHECK(Error, self->slots);
  self->batches = (struct prefetcher_batch_entry*)calloc(
    cfg->batch_capacity, sizeof(*self->batches));
  CHECK(Error, self->batches);
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
    }
    free(self->slots);
  }
  free(self->batches);
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
pop_terminal_slot_locked(struct prefetcher* self,
                         const uint64_t* match_batch_id,
                         struct prefetcher_ready* out)
{
  uint32_t best = UINT32_MAX;
  uint64_t best_seq = 0;
  for (uint32_t i = 0; i < self->capacity; ++i) {
    struct prefetcher_slot* s = &self->slots[i];
    if (s->state != PREFETCHER_READY && s->state != PREFETCHER_ERROR)
      continue;
    if (match_batch_id && s->batch_id != *match_batch_id)
      continue;
    if (best == UINT32_MAX || s->admit_seq < best_seq) {
      best = i;
      best_seq = s->admit_seq;
    }
  }
  if (best == UINT32_MAX)
    return 0;

  struct prefetcher_slot* s = &self->slots[best];
  *out = (struct prefetcher_ready){
    .state = s->state,
    .err_code = s->err_code,
    .uri = s->uri,
    .aabb = s->aabb,
    .batch_id = s->batch_id,
    .h_meta = s->h_meta,
    .h_shards = s->h_shards,
    .n_shards = s->n_shards,
    .h_layout = s->h_layout,
  };
  uint64_t batch_id = s->batch_id;
  *s = (struct prefetcher_slot){ .state = PREFETCHER_FREE };
  batch_unref_locked(self, batch_id);
  return 1;
}

int
prefetcher_pop_ready(struct prefetcher* self, struct prefetcher_ready* out)
{
  CHECK(Bad, self);
  CHECK(Bad, out);
  *out = (struct prefetcher_ready){ 0 };
  platform_mutex_lock(self->lock);
  int found = pop_terminal_slot_locked(self, NULL, out);
  platform_mutex_unlock(self->lock);
  return found;
Bad:
  return 0;
}

int
prefetcher_pop_ready_for_batch(struct prefetcher* self,
                               uint64_t batch_id,
                               struct prefetcher_ready* out)
{
  CHECK(Bad, self);
  CHECK(Bad, out);
  *out = (struct prefetcher_ready){ 0 };
  platform_mutex_lock(self->lock);
  int found = pop_terminal_slot_locked(self, &batch_id, out);
  platform_mutex_unlock(self->lock);
  return found;
Bad:
  return 0;
}

uint32_t
prefetcher_ready_count_for_batch(struct prefetcher* self, uint64_t batch_id)
{
  CHECK(Bad, self);
  platform_mutex_lock(self->lock);
  uint32_t n = 0;
  for (uint32_t i = 0; i < self->capacity; ++i) {
    const struct prefetcher_slot* s = &self->slots[i];
    if ((s->state == PREFETCHER_READY || s->state == PREFETCHER_ERROR) &&
        s->batch_id == batch_id)
      n++;
  }
  platform_mutex_unlock(self->lock);
  return n;
Bad:
  return 0;
}

int
prefetcher_batch_full_ready(struct prefetcher* self,
                            uint64_t batch_id,
                            uint32_t n)
{
  return prefetcher_ready_count_for_batch(self, batch_id) >= n;
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

int
prefetcher_has_ready(struct prefetcher* self)
{
  CHECK(Bad, self);
  platform_mutex_lock(self->lock);
  int found = 0;
  for (uint32_t i = 0; i < self->capacity; ++i) {
    const struct prefetcher_slot* s = &self->slots[i];
    if (s->state == PREFETCHER_READY || s->state == PREFETCHER_ERROR) {
      found = 1;
      break;
    }
  }
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
prefetcher_batch_gate(struct prefetcher* self, uint64_t batch_id)
{
  CHECK(Bad, self);
  platform_mutex_lock(self->lock);
  struct prefetcher_batch_entry* e = batch_lookup_locked(self, batch_id);
  const struct prefetch_gate* g = e ? &e->gate : NULL;
  platform_mutex_unlock(self->lock);
  return g;
Bad:
  return NULL;
}

void
prefetcher_release_batch(struct prefetcher* self, uint64_t batch_id)
{
  if (!self)
    return;
  platform_mutex_lock(self->lock);
  struct prefetcher_batch_entry* e = batch_lookup_locked(self, batch_id);
  if (e) {
    e->release_pending = 1;
    batch_try_free_locked(e);
  }
  platform_mutex_unlock(self->lock);
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
