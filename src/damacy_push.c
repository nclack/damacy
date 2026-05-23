#include "damacy.h"

#include "damacy_config.h"
#include "damacy_internal.h"
#include "zarr/zarr_metadata.h"

// 1 if `aabb` extents (assumed same rank as cfg) match cfg->sample_shape.
static int
sample_aabb_extents_match_cfg(const struct damacy_config* cfg,
                              const struct damacy_aabb* aabb)
{
  for (uint8_t d = 0; d < cfg->sample_rank; ++d) {
    int64_t extent = aabb->dims[d].end - aabb->dims[d].beg;
    if (extent != cfg->sample_shape[d])
      return 0;
  }
  return 1;
}

static enum damacy_status
push_one(struct damacy* self, const struct damacy_sample* sample)
{
  if (!sample->uri)
    return DAMACY_INVAL;
  if (sample->aabb.rank == 0 || sample->aabb.rank > DAMACY_MAX_RANK)
    return DAMACY_RANK;

  struct zarr_metadata meta;
  enum damacy_status ms =
    zarr_meta_cache_get(self->meta_cache, sample->uri, &meta);
  if (ms != DAMACY_OK)
    return ms;

  if (!cast_path_supported(self->cfg.dtype, meta.dtype))
    return DAMACY_DTYPE;
  if (sample->aabb.rank != meta.rank)
    return DAMACY_RANK;
  if (sample->aabb.rank != self->cfg.sample_rank)
    return DAMACY_RANK;
  if (!sample_aabb_extents_match_cfg(&self->cfg, &sample->aabb))
    return DAMACY_INVAL;

  if (lookahead_push(&self->lookahead, sample))
    return DAMACY_OOM;
  return DAMACY_OK;
}

struct damacy_push_result
damacy_push(struct damacy* self, struct damacy_sample_slice samples)
{
  struct damacy_push_result r = { .unconsumed = samples, .status = DAMACY_OK };
  if (!self) {
    r.status = DAMACY_INVAL;
    return r;
  }
  if (samples.beg > samples.end) {
    r.status = DAMACY_INVAL;
    return r;
  }
  // No ctx_guard: push touches no CUDA. The lock guards lookahead +
  // meta/shape checks against the worker's plan_into_slot drain.
  scheduler_lock(self->sched);
  if (self->failed_status != DAMACY_OK) {
    r.status = DAMACY_SHUTDOWN;
    goto Done;
  }
  for (const struct damacy_sample* s = samples.beg; s != samples.end; ++s) {
    if (self->lookahead.size == self->lookahead.cap) {
      r.unconsumed.beg = s;
      r.status = DAMACY_AGAIN;
      goto Done;
    }
    enum damacy_status ps = push_one(self, s);
    if (ps != DAMACY_OK) {
      r.unconsumed.beg = s;
      r.status = ps;
      goto Done;
    }
  }
  r.unconsumed.beg = samples.end;
Done:
  scheduler_unlock(self->sched);
  return r;
}
