#pragma once

#include "platform/platform.h"
#include "store/store.h"

#include <stdint.h>

struct wave_desc;

enum slot_state
{
  SLOT_FREE = 0,
  SLOT_PEELING,
  SLOT_IO,
  SLOT_READY,
  SLOT_BUSY,
};

struct input_slot
{
  enum slot_state state;
  void* buf;
  void* dev_buf;
  uint64_t cap;
  uint64_t used_bytes;

  uint16_t render_job_idx;
  uint16_t batch_pool_slot;
  uint32_t batch_chunk_offset;
  uint32_t n_chunks;

  struct store_read* store_reads;
  struct store_event io_event;
  uint8_t is_fill_wave;

  struct platform_clock io_clock;
  float io_ms;
  uint64_t io_bytes;
};

int
input_slot_init(struct input_slot* slot,
                uint32_t max_chunks_per_wave,
                uint64_t host_cap,
                uint64_t dev_cap);

void
input_slot_destroy(struct input_slot* slot, int cuda_skip);

void
input_slot_begin_peel(struct input_slot* slot, const struct wave_desc* desc);

void
input_slot_commit_io(struct input_slot* slot, struct store_event ev);

void
input_slot_mark_ready(struct input_slot* slot);

void
input_slot_mark_busy(struct input_slot* slot);

float
input_slot_bind_wait_ms(struct input_slot* slot);

void
input_slot_release(struct input_slot* slot);

int
input_slot_find_free(const struct input_slot* slots, uint8_t n);
int
input_slot_find_ready(const struct input_slot* slots, uint8_t n);
int
input_slot_any_in_flight(const struct input_slot* slots, uint8_t n);
int
input_slot_any_free(const struct input_slot* slots, uint8_t n);
