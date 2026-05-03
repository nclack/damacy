// damacy stub implementation. The public surface is wired but every
// fallible path is a no-op or returns DAMACY_AGAIN. Real planner / IO /
// CUDA work lands in subsequent build-order steps (see
// docs/api-design-internals-draft.md).
#include "damacy.h"

#include <stdlib.h>
#include <string.h>

struct damacy
{
  struct damacy_config cfg;
  enum damacy_status failed_status; // DAMACY_OK if healthy
  uint64_t next_batch_id;
};

static enum damacy_status
validate_config(const struct damacy_config* cfg)
{
  if (!cfg)
    return DAMACY_INVAL;
  if (cfg->batch_size == 0)
    return DAMACY_INVAL;
  if (cfg->lookahead_batches < 2)
    return DAMACY_INVAL;
  if (cfg->n_io_threads == 0)
    return DAMACY_INVAL;
  if (cfg->host_buffer_bytes == 0 || cfg->device_buffer_bytes == 0)
    return DAMACY_INVAL;
  if (cfg->n_zarrs_meta_cache == 0 || cfg->n_shards_meta_cache == 0)
    return DAMACY_INVAL;
  return DAMACY_OK;
}

enum damacy_status
damacy_create(const struct damacy_config* cfg, struct damacy** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  enum damacy_status s = validate_config(cfg);
  if (s != DAMACY_OK)
    return s;

  struct damacy* d = (struct damacy*)calloc(1, sizeof(*d));
  if (!d)
    return DAMACY_OOM;
  d->cfg = *cfg;
  d->failed_status = DAMACY_OK;
  d->next_batch_id = 0;
  *out = d;
  return DAMACY_OK;
}

void
damacy_destroy(struct damacy* d)
{
  if (!d)
    return;
  free(d);
}

struct damacy_push_result
damacy_push(struct damacy* d, struct damacy_sample_slice samples)
{
  struct damacy_push_result r = { .unconsumed = samples, .status = DAMACY_OK };
  if (!d) {
    r.status = DAMACY_INVAL;
    return r;
  }
  if (d->failed_status != DAMACY_OK) {
    r.status = DAMACY_SHUTDOWN;
    return r;
  }
  if (samples.beg > samples.end) {
    r.status = DAMACY_INVAL;
    return r;
  }
  // Stub: pretend we consumed everything. A real implementation would
  // copy samples into the lookahead queue and run the planner.
  r.unconsumed.beg = samples.end;
  return r;
}

enum damacy_status
damacy_pop(struct damacy* d, struct damacy_batch** out)
{
  if (!d || !out)
    return DAMACY_INVAL;
  *out = NULL;
  if (d->failed_status != DAMACY_OK)
    return d->failed_status;
  // Stub: never has a batch ready.
  return DAMACY_AGAIN;
}

void
damacy_release(struct damacy* d, struct damacy_batch* b)
{
  (void)d;
  (void)b;
}

enum damacy_status
damacy_flush(struct damacy* d)
{
  if (!d)
    return DAMACY_INVAL;
  if (d->failed_status != DAMACY_OK)
    return d->failed_status;
  return DAMACY_OK;
}

void
damacy_batch_info(const struct damacy_batch* b, struct damacy_batch_info* out)
{
  (void)b;
  if (out)
    memset(out, 0, sizeof(*out));
}

void
damacy_stats_get(const struct damacy* d, struct damacy_stats* out)
{
  (void)d;
  if (out)
    memset(out, 0, sizeof(*out));
}

void
damacy_stats_reset(struct damacy* d)
{
  (void)d;
}
