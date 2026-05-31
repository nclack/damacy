#pragma once

#include "damacy.h"
#include "store/store.h"

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

struct damacy_wave;
struct input_slot;

struct input_transfer_resources
{
  uint64_t slot_host_bytes;
  uint64_t slot_device_bytes;
  uint64_t wave_device_bytes;
  uint64_t device_staging_bytes;
  uint8_t device_staging_buffers;
};

struct input_transfer_ops
{
  const char* name;
  struct input_transfer_resources (*resources)(uint8_t n_slots,
                                               uint64_t bytes_per_wave);
  void (*bind_stream)(struct store* store, CUstream stream);
  void* (*read_base)(struct input_slot* slot);
  void* (*wave_input)(struct damacy_wave* wave, const struct input_slot* slot);
  struct store_event (*submit_reads)(struct store* store,
                                     const struct store_read* reads,
                                     size_t n_reads);
  enum damacy_status (*queue_input)(CUstream stream,
                                    struct damacy_wave* wave,
                                    uint8_t* queued_stream_work);
  enum damacy_status (*slot_reuse_ready)(const struct damacy_wave* wave,
                                         int* ready);
};

const struct input_transfer_ops*
input_transfer_host_staging(void);

const struct input_transfer_ops*
input_transfer_gds(void);

struct input_transfer_resources
input_transfer_resources(const struct input_transfer_ops* ops,
                         uint8_t n_slots,
                         uint64_t bytes_per_wave);
