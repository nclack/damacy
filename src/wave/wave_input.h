#pragma once

#include "damacy.h"
#include "render_job/render_job.h"
#include "store/store.h"

#include <stdint.h>

struct wave_pool;

struct wave_input_reservation
{
  int input_slot_idx;
  uint32_t n_reads;
  struct wave_desc desc;
  uint8_t committed;
};

struct wave_input_reservation
wave_input_reserve(struct wave_pool* wp,
                   uint16_t render_job_idx,
                   enum damacy_status* err);

struct store_event
wave_input_submit(struct wave_pool* wp, const struct wave_input_reservation* t);

enum damacy_status
wave_input_commit(struct wave_pool* wp,
                  struct wave_input_reservation* t,
                  struct store_event ev,
                  int* changed);
