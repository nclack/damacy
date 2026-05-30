// Compressed-input transfer geometry.
#pragma once

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

uint8_t
compressed_input_device_instances(enum compressed_input_mode mode,
                                  uint8_t n_slots);

struct compressed_input_resources
compressed_input_resources(enum compressed_input_mode mode,
                           uint8_t n_slots,
                           uint64_t bytes_per_wave);
