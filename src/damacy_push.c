#include "damacy.h"

#include "damacy_internal.h"

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

// Cfg-only validations; URI / dtype / per-array rank checks surface
// asynchronously through prefetch / plan / pop.
static enum damacy_status
push_one(struct damacy* self, const struct damacy_sample* sample)
{
  if (!sample->uri)
    return DAMACY_INVAL;
  if (sample->aabb.rank == 0 || sample->aabb.rank > DAMACY_MAX_RANK)
    return DAMACY_RANK;
  if (sample->aabb.rank != self->cfg.sample_rank)
    return DAMACY_RANK;
  if (!sample_aabb_extents_match_cfg(&self->cfg, &sample->aabb))
    return DAMACY_INVAL;

  uint64_t batch_id = self->pushed_samples / self->cfg.batch_size;
  if (lookahead_push_with_batch(&self->lookahead, sample, batch_id))
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
  // No ctx_guard: push touches no CUDA. The lock pairs lookahead_push
  // with the worker's plan_reserve drain.
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
    self->pushed_samples++;
  }
  r.unconsumed.beg = samples.end;
Done:
  scheduler_unlock(self->sched);
  return r;
}
