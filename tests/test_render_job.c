// Unit tests for render_job's host-side ownership and cursor helpers.

#include "expect.h"
#include "render_job/render_job.h"

#include <stdio.h>
#include <string.h>

static void
wire_job_storage(struct render_job* job,
                 struct read_op* reads,
                 struct chunk_plan* chunks,
                 struct sample_plan* samples,
                 struct read_op_group* groups)
{
  memset(job, 0, sizeof(*job));
  job->read_ops = reads;
  job->chunk_plans = chunks;
  job->sample_plans = samples;
  job->read_op_groups = groups;
}

static int
test_planner_output_borrows_job_storage(void)
{
  struct render_job job;
  struct read_op reads[2];
  struct chunk_plan chunks[2];
  struct sample_plan samples[2];
  struct read_op_group groups[2];
  wire_job_storage(&job, reads, chunks, samples, groups);

  struct planner_output out = render_job_planner_output(&job, 2);
  EXPECT(out.read_ops == reads);
  EXPECT(out.chunk_plans == chunks);
  EXPECT(out.sample_plans == samples);
  EXPECT(out.read_op_groups == groups);
  EXPECT(out.read_ops_cap == DAMACY_MAX_CHUNKS_PER_BATCH);
  EXPECT(out.sample_plans_cap == 2);
  return 0;
}

static int
test_commit_and_find_oldest_work(void)
{
  struct render_job_pool pool;
  memset(&pool, 0, sizeof(pool));
  struct planner_output out = {
    .n_chunk_plans = 3,
    .n_chunks_to_load = 2,
    .n_loads_issued = 1,
    .n_sample_plans = 2,
    .n_read_op_groups = 2,
  };

  struct render_job* j0 = render_job_pool_for_batch_slot(&pool, 0);
  struct render_job* j1 = render_job_pool_for_batch_slot(&pool, 1);
  EXPECT(j0);
  EXPECT(j1);
  render_job_commit_plan(j0, 0, 11, &out);
  render_job_commit_plan(j1, 1, 10, &out);
  EXPECT(render_job_has_work(j0));
  EXPECT(find_render_job_with_work(&pool) == 1);

  j1->n_chunks_dispatched = 3;
  EXPECT(find_render_job_with_work(&pool) == 0);

  render_job_finish(j0);
  EXPECT(j0->state == RENDER_JOB_FREE);
  EXPECT(j0->n_chunks == 0);
  EXPECT(find_render_job_with_work(&pool) == -1);
  return 0;
}

int
main(void)
{
  RUN(test_planner_output_borrows_job_storage);
  RUN(test_commit_and_find_oldest_work);
  printf("all render_job tests passed\n");
  return 0;
}
