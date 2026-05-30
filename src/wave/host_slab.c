#include "host_slab.h"

#include "damacy_limits.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

int
slot_init(struct host_slab_slot* slot,
          uint32_t max_chunks_per_wave,
          uint64_t host_cap,
          uint64_t dev_cap)
{
  memset(slot, 0, sizeof(*slot));
  // Logical cap is the same regardless of where the bytes land.
  slot->cap = host_cap > 0 ? host_cap : dev_cap;
  if (slot->cap == 0)
    return 1;
  if (host_cap > 0 && cuMemAllocHost(&slot->buf, host_cap) != CUDA_SUCCESS)
    goto Error;
  if (dev_cap > 0) {
    CUdeviceptr dptr = 0;
    if (cuMemAlloc(&dptr, dev_cap) != CUDA_SUCCESS)
      goto Error;
    slot->dev_buf = (void*)(uintptr_t)dptr;
  }
  slot->store_reads = (struct store_read*)calloc((size_t)max_chunks_per_wave,
                                                 sizeof(struct store_read));
  if (!slot->store_reads)
    goto Error;
  return 0;
Error:
  if (slot->buf) {
    cuMemFreeHost(slot->buf);
    slot->buf = NULL;
  }
  if (slot->dev_buf) {
    cuMemFree((CUdeviceptr)(uintptr_t)slot->dev_buf);
    slot->dev_buf = NULL;
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
  if (!cuda_skip) {
    if (slot->buf)
      cuMemFreeHost(slot->buf);
    if (slot->dev_buf)
      cuMemFree((CUdeviceptr)(uintptr_t)slot->dev_buf);
  }
  slot->buf = NULL;
  slot->dev_buf = NULL;
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
  slot->render_job_idx = 0;
  slot->batch_pool_slot = 0;
  slot->batch_chunk_offset = 0;
  slot->io_event = (struct store_event){ 0 };
  slot->is_fill_wave = 0;
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
