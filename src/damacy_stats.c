#include "damacy_stats.h"

#include <string.h>
#include <time.h>

uint64_t
monotonic_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void
metric_init(struct damacy_metric* m, const char* name)
{
  m->name = name;
  m->ms = 0.f;
  m->best_ms = 1e30f;
  m->input_bytes = 0;
  m->output_bytes = 0;
  m->count = 0;
}

void
metric_record(struct damacy_metric* m, float ms, uint64_t bin, uint64_t bout)
{
  m->ms += ms;
  if (ms < m->best_ms)
    m->best_ms = ms;
  m->input_bytes += (double)bin;
  m->output_bytes += (double)bout;
  m->count += 1;
}

void
stats_init(struct damacy_stats* s)
{
  // memset first; metric_init then overwrites best_ms to a large sentinel.
  memset(s, 0, sizeof(*s));
  metric_init(&s->plan, "plan");
  metric_init(&s->io, "io");
  metric_init(&s->h2d, "h2d");
  metric_init(&s->decode, "decode");
  metric_init(&s->post_decode, "post_decode");
  metric_init(&s->decode_gap, "decode_gap");
  metric_init(&s->decompress_parse, "decompress.parse");
  metric_init(&s->assemble, "assemble");
  metric_init(&s->pop_wait, "pop_wait");
  metric_init(&s->flush_wait, "flush_wait");
}
