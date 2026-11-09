#pragma once

#include "util/slice.h"
#include <stdint.h>

struct cold_read;

struct cold_read_counts
{
  uint64_t reads;
  uint64_t bytes;
  uint64_t writes;
};

struct cold_read_result
{
  uint64_t passes;
  uint64_t files;
  uint64_t pages;
  uint64_t resident_pages_before;
  uint64_t resident_pages_after;
  double preparation_ms;
  double idle_seconds;
  struct cold_read_counts device;
  struct cold_read_counts idle;
};

int
cold_read_create(struct cslice source, struct cold_read** out);
int
cold_read_begin(struct cold_read* state);
int
cold_read_end(struct cold_read* state);
int
cold_read_idle(struct cold_read* state);
void
cold_read_result(struct cold_read* state, struct cold_read_result* out);
void
cold_read_destroy(struct cold_read* state);
