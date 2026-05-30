// Host-staging and GDS compressed-input transfer operations.
#pragma once

#include "damacy.h"
#include "store/store.h"

#include <cuda.h>
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

struct input_transfer_submit_state
{
  uint8_t stream_work_queued;
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
                                     struct store_read* reads,
                                     uint32_t n_reads);
  enum damacy_status (*queue_ready)(CUstream stream,
                                    struct damacy_wave* wave,
                                    struct input_transfer_submit_state* state);
  CUevent (*slot_release_gate)(const struct damacy_wave* wave);
};

const struct input_transfer_ops*
input_transfer_h2d(void);

const struct input_transfer_ops*
input_transfer_gds(void);

struct input_transfer_resources
input_transfer_resources(const struct input_transfer_ops* ops,
                         uint8_t n_slots,
                         uint64_t bytes_per_wave);
