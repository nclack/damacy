#include "util/pool.h"

#include "log/log.h"
#include "platform/platform.h"
#include "util/prelude.h"

#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct pool_slot
{
  struct pool_slot* next;
};

struct pool
{
  struct platform_mutex* lock;
  unsigned char* storage;
  struct pool_slot* head;
  size_t slot_size;
  size_t capacity;
  size_t in_use;
};

static size_t
pool_round_slot(size_t elem_size)
{
  size_t a = _Alignof(max_align_t);
  if (a < sizeof(void*))
    a = sizeof(void*);
  if (elem_size < sizeof(void*))
    elem_size = sizeof(void*);
  return (elem_size + a - 1) & ~(a - 1);
}

struct pool*
pool_create(size_t elem_size, size_t capacity)
{
  struct pool* p = NULL;
  CHECK(Fail, elem_size > 0);
  CHECK(Fail, capacity > 0);

  p = (struct pool*)calloc(1, sizeof(*p));
  CHECK(Fail, p);

  p->slot_size = pool_round_slot(elem_size);
  p->capacity = capacity;

  // Overflow guard on the multiply.
  CHECK(Fail, p->slot_size == 0 || capacity <= (SIZE_MAX / p->slot_size));

  p->storage = (unsigned char*)calloc(capacity, p->slot_size);
  CHECK(Fail, p->storage);

  p->lock = platform_mutex_new();
  CHECK(Fail, p->lock);

  for (size_t i = 0; i + 1 < capacity; ++i) {
    struct pool_slot* s = (struct pool_slot*)(p->storage + i * p->slot_size);
    s->next = (struct pool_slot*)(p->storage + (i + 1) * p->slot_size);
  }
  {
    struct pool_slot* last =
      (struct pool_slot*)(p->storage + (capacity - 1) * p->slot_size);
    last->next = NULL;
  }
  p->head = (struct pool_slot*)p->storage;
  return p;

Fail:
  pool_destroy(p);
  return NULL;
}

void
pool_destroy(struct pool* p)
{
  if (!p)
    return;
  // Outstanding slots are interior pointers into p->storage; freeing it
  // would dangle them. Leak everything instead so live callers stay safe.
  CHECK(Leak, p->in_use == 0);
  platform_mutex_free(p->lock);
  free(p->storage);
  free(p);
  return;
Leak:
  return;
}

static bool
pool_owns_unlocked(const struct pool* p, const void* ptr)
{
  // storage is immutable after pool_create; no lock needed.
  if (!p || !ptr || !p->storage)
    return false;
  const unsigned char* q = (const unsigned char*)ptr;
  if (q < p->storage || q >= p->storage + p->capacity * p->slot_size)
    return false;
  return ((size_t)(q - p->storage)) % p->slot_size == 0;
}

void*
pool_alloc(struct pool* p)
{
  if (!p)
    return NULL;
  struct pool_slot* slot = NULL;
  platform_mutex_lock(p->lock);
  if (p->head) {
    slot = p->head;
    p->head = slot->next;
    ++p->in_use;
  }
  platform_mutex_unlock(p->lock);
  if (!slot)
    return NULL;
  memset(slot, 0, p->slot_size);
  return slot;
}

void
pool_free(struct pool* p, void* ptr)
{
  if (!p || !ptr)
    return;
  CHECK(End, pool_owns_unlocked(p, ptr));
  struct pool_slot* slot = (struct pool_slot*)ptr;
  platform_mutex_lock(p->lock);
  CHECK(Unlock, p->in_use > 0);
  slot->next = p->head;
  p->head = slot;
  --p->in_use;
Unlock:
  platform_mutex_unlock(p->lock);
End:
  return;
}

size_t
pool_in_use(struct pool* p)
{
  if (!p)
    return 0;
  platform_mutex_lock(p->lock);
  size_t n = p->in_use;
  platform_mutex_unlock(p->lock);
  return n;
}

size_t
pool_capacity(const struct pool* p)
{
  return p ? p->capacity : 0;
}
