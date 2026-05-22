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
  char* uri;
  struct damacy_aabb aabb;
  uint64_t batch_id;
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
  struct damacy_lookahead* la;
  struct prefetch_cache* amc;
  struct prefetch_cache* sic;
  struct prefetch_cache* clc;

  struct prefetcher_slot* slots;
  uint32_t capacity;

  struct prefetcher_batch_entry* batches;
  uint32_t batch_capacity;

  struct platform_mutex* lock;
  struct platform_thread* worker;
  _Atomic int stop;
  _Atomic int started;

  uint64_t submitted;
  uint64_t ready;
  uint64_t errored;
};

static struct prefetcher_batch_entry*
batch_lookup_locked(struct prefetcher* p, uint64_t batch_id)
{
  for (uint32_t i = 0; i < p->batch_capacity; ++i)
    if (p->batches[i].in_use && p->batches[i].batch_id == batch_id)
      return &p->batches[i];
  return NULL;
}

// Caller must pair with a refcount drop (batch_unref_locked or inline at admit
// rollback). *out_was_new lets admit free a freshly-created entry directly on
// failure — release_pending is 0, so plain unref would leak it.
static struct prefetcher_batch_entry*
batch_get_or_create_locked(struct prefetcher* p,
                           uint64_t batch_id,
                           int* out_was_new)
{
  *out_was_new = 0;
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
      *out_was_new = 1;
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
fail_slot(struct prefetcher* p, struct prefetcher_slot* s)
{
  // Catches prefetcher-origin errors that bypass the cache (alloc fail,
  // saturation).
  prefetch_gate_set_error(s->gate);
  s->state = PREFETCHER_ERROR;
  p->errored++;
}

static void
advance_from_meta(struct prefetcher* p, struct prefetcher_slot* s)
{
  int err = 0;
  enum prefetch_state st = prefetch_cache_query(p->amc, s->h_meta, NULL, &err);
  if (st == PREFETCH_STATE_PENDING)
    return;
  if (st == PREFETCH_STATE_ERROR)
    goto Bad;

  const struct zarr_metadata* meta =
    (const struct zarr_metadata*)prefetch_cache_try_get(p->amc, s->h_meta);
  CHECK(Bad, meta);

  struct sample_shard_iterator it;
  CHECK(Bad, sample_shard_iterator_init(&it, meta, &s->aabb) == 0);

  uint64_t n = 1;
  for (uint8_t d = 0; d < it.rank; ++d)
    n *= (it.shard_end[d] - it.shard_beg[d]);

  if (n == 0) {
    s->n_shards = 0;
    s->h_layout = prefetch_cache_request(
      p->clc, hash_fnv1a_str(s->uri), s->uri, s->batch_id, s->gate);
    CHECK(Bad, prefetch_handle_valid(s->h_layout));
    s->state = PREFETCHER_PENDING_CHUNK_LAYOUT;
    return;
  }

  s->h_shards =
    (struct prefetch_handle*)calloc((size_t)n, sizeof(*s->h_shards));
  CHECK(Bad, s->h_shards);

  struct shard_index_key probe = { .uri = s->uri, .rank = meta->rank };
  uint32_t i = 0;
  while (sample_shard_iterator_next(&it, probe.shard_coord)) {
    s->h_shards[i] = prefetch_cache_request(
      p->sic, shard_index_key_hash(&probe), &probe, s->batch_id, s->gate);
    CHECK(Bad, prefetch_handle_valid(s->h_shards[i]));
    i++;
  }
  s->n_shards = (uint32_t)n;
  s->state = PREFETCHER_PENDING_SHARD_INDEX;
  return;
Bad:
  fail_slot(p, s);
}

static void
advance_from_shard(struct prefetcher* p, struct prefetcher_slot* s)
{
  for (uint32_t i = 0; i < s->n_shards; ++i) {
    int err = 0;
    enum prefetch_state st =
      prefetch_cache_query(p->sic, s->h_shards[i], NULL, &err);
    if (st == PREFETCH_STATE_PENDING)
      return;
    if (st == PREFETCH_STATE_ERROR)
      goto Bad;
  }

  s->h_layout = prefetch_cache_request(
    p->clc, hash_fnv1a_str(s->uri), s->uri, s->batch_id, s->gate);
  CHECK(Bad, prefetch_handle_valid(s->h_layout));
  s->state = PREFETCHER_PENDING_CHUNK_LAYOUT;
  return;
Bad:
  fail_slot(p, s);
}

static void
advance_from_layout(struct prefetcher* p, struct prefetcher_slot* s)
{
  int err = 0;
  enum prefetch_state st =
    prefetch_cache_query(p->clc, s->h_layout, NULL, &err);
  if (st == PREFETCH_STATE_PENDING)
    return;
  if (st == PREFETCH_STATE_ERROR)
    goto Bad;
  s->state = PREFETCHER_READY;
  p->ready++;
  return;
Bad:
  fail_slot(p, s);
}

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

static void
admit_locked(struct prefetcher* p,
             struct prefetcher_slot* slot,
             struct damacy_sample_slot* popped)
{
  int was_new = 0;
  struct prefetcher_batch_entry* be =
    batch_get_or_create_locked(p, popped->batch_id, &was_new);
  struct prefetch_gate* gate = be ? &be->gate : NULL;
  struct prefetch_handle h = prefetch_cache_request(
    p->amc, hash_fnv1a_str(popped->uri), popped->uri, popped->batch_id, gate);
  CHECK(Bad, prefetch_handle_valid(h));
  *slot = (struct prefetcher_slot){
    .state = PREFETCHER_PENDING_ARRAY_META,
    .uri = popped->uri,
    .aabb = popped->aabb,
    .batch_id = popped->batch_id,
    .gate = gate,
    .h_meta = h,
  };
  p->submitted++;
  return;
Bad:
  if (be) {
    if (be->refcount > 0)
      be->refcount--;
    if (was_new || be->release_pending)
      batch_try_free_locked(be);
  }
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
      popped_ok = lookahead_pop_blocking(p->la, &popped);
    else if (slot)
      popped_ok = lookahead_pop_blocking_timeout(p->la, &popped, 1);
    else
      platform_sleep_ns(1000000);

    if (popped_ok) {
      platform_mutex_lock(p->lock);
      admit_locked(p, slot, &popped);
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
    .la = cfg->lookahead,
    .amc = cfg->array_meta_cache,
    .sic = cfg->shard_index_cache,
    .clc = cfg->chunk_layout_cache,
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
  lookahead_signal_stop(self->la);
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
    if (lookahead_size(self->la) == 0) {
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

int
prefetcher_pop_ready(struct prefetcher* self, struct prefetcher_ready* out)
{
  CHECK(Bad, self);
  CHECK(Bad, out);
  *out = (struct prefetcher_ready){ 0 };
  int found = 0;
  platform_mutex_lock(self->lock);
  for (uint32_t i = 0; i < self->capacity; ++i) {
    struct prefetcher_slot* s = &self->slots[i];
    if (s->state != PREFETCHER_READY && s->state != PREFETCHER_ERROR)
      continue;
    *out = (struct prefetcher_ready){
      .state = s->state,
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
    found = 1;
    break;
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
  prefetch_cache_advance_watermark(self->amc, watermark);
  prefetch_cache_advance_watermark(self->sic, watermark);
  prefetch_cache_advance_watermark(self->clc, watermark);
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
