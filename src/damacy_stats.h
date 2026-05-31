#pragma once

#include "damacy.h"

#include <stdint.h>

void
metric_init(struct damacy_metric* m, const char* name);

// Stall-style metrics pass 0 for both byte counters.
void
metric_record(struct damacy_metric* m, float ms, uint64_t bin, uint64_t bout);

void
stats_init(struct damacy_stats* s);

struct damacy_metric*
stats_input_transfer(struct damacy_stats* s);

const struct damacy_metric*
stats_input_transfer_const(const struct damacy_stats* s);
