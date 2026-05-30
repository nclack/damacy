#pragma once

#include "damacy.h"

#include <stdint.h>

struct platform_mutex;
struct platform_cond;

struct damacy_sample_slot
{
  char* uri;
  struct damacy_aabb aabb;
  uint64_t sample_seq;
};

struct damacy_lookahead
{
  struct damacy_sample_slot* slots;
  uint32_t cap;
  uint32_t head;
  uint32_t tail;
  uint32_t size;
  struct platform_mutex* lock;
  struct platform_cond* cond;
  int stop_signaled;
};

void
sample_slot_clear(struct damacy_sample_slot* slot);

int
lookahead_init(struct damacy_lookahead* la, uint32_t cap);

void
lookahead_destroy(struct damacy_lookahead* la);

int
lookahead_push(struct damacy_lookahead* la, const struct damacy_sample* sample);

int
lookahead_push_with_sample_seq(struct damacy_lookahead* la,
                               const struct damacy_sample* sample,
                               uint64_t sample_seq);

void
lookahead_drain(struct damacy_lookahead* la,
                struct damacy_sample_slot* out,
                uint32_t n);

// Returns 1 on pop, 0 when stopped with empty queue.
int
lookahead_pop_blocking(struct damacy_lookahead* la,
                       struct damacy_sample_slot* out);

int
lookahead_pop_blocking_timeout(struct damacy_lookahead* la,
                               struct damacy_sample_slot* out,
                               int timeout_ms);

// Non-blocking. Returns 1 on pop, 0 when empty.
int
lookahead_try_pop(struct damacy_lookahead* la, struct damacy_sample_slot* out);

// Existing samples still pop; empty + stopped returns 0.
void
lookahead_signal_stop(struct damacy_lookahead* la);

uint32_t
lookahead_size(struct damacy_lookahead* la);
