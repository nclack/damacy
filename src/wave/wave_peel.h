#pragma once

#include "damacy.h"
#include "render_job/render_job.h"
#include "store/store.h"

#include <stdint.h>

struct wave_pool;

struct wave_pool_peel_ticket
{
  int slot_idx;
  uint32_t n_reads;
  struct wave_desc desc;
  uint8_t consumed;
};

struct wave_pool_peel_ticket
wave_pool_peel_reserve(struct wave_pool* wp,
                       uint16_t render_job_idx,
                       enum damacy_status* err);

struct store_event
wave_pool_peel_submit(struct wave_pool* wp,
                      const struct wave_pool_peel_ticket* t);

enum damacy_status
wave_pool_peel_commit(struct wave_pool* wp,
                      struct wave_pool_peel_ticket* t,
                      struct store_event ev,
                      int* changed);
