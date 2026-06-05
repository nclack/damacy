#pragma once

#include "damacy.h"
#include "render_job/render_job.h"
#include "store/store.h"

#include <stdint.h>

struct input_slot;
struct wave_pool;

struct wave_input_reservation
{
  uint8_t has_slot;
  uint8_t finalized;
  int input_slot_idx;
  uint32_t n_reads;
  struct wave_desc desc;
};

int
wave_input_reservation_has_slot(const struct wave_input_reservation* r);

int
wave_input_reservation_slot_index(const struct wave_input_reservation* r);

enum damacy_status
wave_input_reserve(struct wave_pool* wp,
                   uint16_t render_job_idx,
                   struct wave_input_reservation* out);

struct store_submit_result
wave_input_submit(struct wave_pool* wp, const struct wave_input_reservation* t);

enum damacy_status
wave_input_commit(struct wave_pool* wp,
                  struct wave_input_reservation* t,
                  struct store_submit_result submit,
                  int* changed);

enum damacy_status
wave_input_rollback_slot(struct wave_pool* wp,
                         struct input_slot* slot,
                         int* changed);
