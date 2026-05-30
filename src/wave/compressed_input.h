// Compressed-input transfer geometry.
#pragma once

#include "damacy_limits.h"

#include <stdint.h>

enum compressed_input_mode
{
  COMPRESSED_INPUT_H2D = 0,
  COMPRESSED_INPUT_GDS = 1,
};

struct compressed_input_resources
{
  uint64_t slot_host_bytes;
  uint64_t slot_device_bytes;
  uint64_t wave_device_bytes;
  uint64_t gpu_budget_bytes;
  uint8_t device_instances;
};

static inline uint8_t
compressed_input_device_instances(enum compressed_input_mode mode,
                                  uint8_t n_slots)
{
  return mode == COMPRESSED_INPUT_GDS ? n_slots : DAMACY_N_WAVES;
}

static inline struct compressed_input_resources
compressed_input_resources(enum compressed_input_mode mode,
                           uint8_t n_slots,
                           uint64_t bytes_per_wave)
{
  const uint8_t instances = compressed_input_device_instances(mode, n_slots);
  struct compressed_input_resources r = {
    .device_instances = instances,
    .gpu_budget_bytes = (uint64_t)instances * bytes_per_wave,
  };
  if (mode == COMPRESSED_INPUT_GDS) {
    r.slot_device_bytes = bytes_per_wave;
  } else {
    r.slot_host_bytes = bytes_per_wave;
    r.wave_device_bytes = bytes_per_wave;
  }
  return r;
}
