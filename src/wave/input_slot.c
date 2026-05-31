#include "input_slot.h"

#include "damacy_limits.h"
#include "render_job/render_job.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>

int
input_slot_init(struct input_slot* slot,
                uint32_t max_chunks_per_wave,
                uint64_t host_cap,
                uint64_t dev_cap)
{
  memset(slot, 0, sizeof(*slot));
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
input_slot_destroy(struct input_slot* slot, int cuda_skip)
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
input_slot_begin_reservation(struct input_slot* slot,
                             const struct wave_desc* desc)
{
  platform_toc(&slot->io_clock);
  slot->is_fill_wave = desc->is_fill_wave;
  slot->render_job_idx = desc->render_job_idx;
  slot->batch_pool_slot = desc->batch_pool_slot;
  slot->batch_chunk_offset = desc->batch_chunk_offset;
  slot->n_chunks = desc->n_chunks;
  slot->used_bytes = desc->input_used_bytes;
  slot->io_bytes = desc->io_bytes;
  slot->state = SLOT_RESERVED;
}

void
input_slot_commit_io(struct input_slot* slot, struct store_event ev)
{
  slot->io_event = ev;
  slot->state = SLOT_IO;
}

void
input_slot_mark_ready(struct input_slot* slot)
{
  slot->io_event = (struct store_event){ 0 };
  slot->io_ms = platform_toc(&slot->io_clock) * 1000.0f;
  slot->state = SLOT_READY;
}

void
input_slot_mark_busy(struct input_slot* slot)
{
  slot->state = SLOT_BUSY;
}

float
input_slot_bind_wait_ms(struct input_slot* slot)
{
  return platform_toc(&slot->io_clock) * 1000.0f;
}

void
input_slot_release(struct input_slot* slot)
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
input_slot_find_free(const struct input_slot* slots, uint8_t n)
{
  for (uint8_t s = 0; s < n; ++s)
    if (slots[s].state == SLOT_FREE)
      return s;
  return -1;
}

int
input_slot_find_ready(const struct input_slot* slots, uint8_t n)
{
  for (uint8_t s = 0; s < n; ++s)
    if (slots[s].state == SLOT_READY)
      return s;
  return -1;
}

int
input_slot_any_in_flight(const struct input_slot* slots, uint8_t n)
{
  for (uint8_t s = 0; s < n; ++s)
    if (slots[s].state != SLOT_FREE)
      return 1;
  return 0;
}

int
input_slot_any_free(const struct input_slot* slots, uint8_t n)
{
  for (uint8_t s = 0; s < n; ++s)
    if (slots[s].state == SLOT_FREE)
      return 1;
  return 0;
}
