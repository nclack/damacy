#include "damacy_internal.h"
#include "damacy_stats.h"
#include "expect.h"

#include <stdio.h>

static int
test_plan_commit_releases_unsealed_slot(void)
{
  struct damacy d = { 0 };
  stats_init(&d.stats);

  struct damacy_batch_slot* slot = &d.batch_pool.slots[0];
  struct render_job* job = render_job_pool_for_batch_slot(&d.render_jobs, 0);
  EXPECT(job);

  slot->state = BATCH_PLANNING;
  slot->batch_id = 7;
  slot->sample_seq_begin = 11;
  slot->n_samples = 2;

  job->state = RENDER_JOB_READY;
  job->batch_pool_slot = 0;
  job->batch_id = 7;
  job->n_chunks = 3;

  int changed = 0;
  EXPECT(plan_commit(&d, 0, DAMACY_OK, 0.25f, &changed) == DAMACY_INVAL);
  EXPECT(d.failed_status == DAMACY_INVAL);
  EXPECT(changed == 1);
  EXPECT(slot->state == BATCH_FREE);
  EXPECT(slot->batch_id == 0);
  EXPECT(slot->sample_seq_begin == 0);
  EXPECT(slot->n_samples == 0);
  EXPECT(job->state == RENDER_JOB_FREE);
  EXPECT(job->batch_id == 0);
  EXPECT(job->n_chunks == 0);
  EXPECT(d.stats.plan.count == 1);
  return 0;
}

int
main(void)
{
  RUN(test_plan_commit_releases_unsealed_slot);
  printf("all damacy_plan tests passed\n");
  return 0;
}
