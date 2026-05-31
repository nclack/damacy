#pragma once

#include "damacy.h"
#include "render_job/render_job.h"
#include "store/store.h"

#include <stdint.h>

struct wave_pool;

struct wave_input_reservation
{
  uint8_t active;
  uint8_t committed;
  int input_slot_idx;
  uint32_t n_reads;
  struct wave_desc desc;
};

enum damacy_status
wave_input_reserve(struct wave_pool* wp,
                   uint16_t render_job_idx,
                   struct wave_input_reservation* out);

enum damacy_status
wave_input_submit(struct wave_pool* wp,
                  const struct wave_input_reservation* t,
                  struct store_event* out);

enum damacy_status
wave_input_commit(struct wave_pool* wp,
                  struct wave_input_reservation* t,
                  enum damacy_status submit_status,
                  struct store_event ev,
                  int* changed);
