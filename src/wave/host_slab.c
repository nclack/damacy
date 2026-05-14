#include "host_slab.h"

#include "damacy_limits.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

int
slot_init(struct host_slab_slot* slot, uint64_t cap)
{
  memset(slot, 0, sizeof(*slot));
  slot->cap = cap;
  if (cuMemAllocHost(&slot->buf, cap) != CUDA_SUCCESS)
    goto Error;
  slot->store_reads = (struct store_read*)calloc(DAMACY_MAX_CHUNKS_PER_WAVE,
                                                 sizeof(struct store_read));
  if (!slot->store_reads)
    goto Error;
  return 0;
Error:
  if (slot->buf) {
    cuMemFreeHost(slot->buf);
    slot->buf = NULL;
  }
  free(slot->store_reads);
  slot->store_reads = NULL;
  return 1;
}

void
slot_destroy(struct host_slab_slot* slot, int cuda_skip)
{
  if (!slot)
    return;
  if (!cuda_skip && slot->buf)
    cuMemFreeHost(slot->buf);
  slot->buf = NULL;
  free(slot->store_reads);
  slot->store_reads = NULL;
  slot->state = SLOT_FREE;
}

void
slot_release(struct host_slab_slot* slot)
{
  slot->state = SLOT_FREE;
  slot->used_bytes = 0;
  slot->n_chunks = 0;
  slot->io_bytes = 0;
  slot->batch_pool_slot = 0;
  slot->batch_chunk_offset = 0;
  slot->io_event = (struct store_event){ 0 };
}

int
host_slab_find_free(const struct host_slab_slot* slots, uint8_t n)
{
  for (uint8_t s = 0; s < n; ++s)
    if (slots[s].state == SLOT_FREE)
      return s;
  return -1;
}

int
host_slab_find_ready(const struct host_slab_slot* slots, uint8_t n)
{
  for (uint8_t s = 0; s < n; ++s)
    if (slots[s].state == SLOT_READY)
      return s;
  return -1;
}

int
host_slab_any_in_flight(const struct host_slab_slot* slots, uint8_t n)
{
  for (uint8_t s = 0; s < n; ++s)
    if (slots[s].state != SLOT_FREE)
      return 1;
  return 0;
}

int
host_slab_any_free(const struct host_slab_slot* slots, uint8_t n)
{
  for (uint8_t s = 0; s < n; ++s)
    if (slots[s].state == SLOT_FREE)
      return 1;
  return 0;
}
