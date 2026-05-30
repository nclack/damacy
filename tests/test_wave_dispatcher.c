// Unit tests for the wave-building boundary that consumes render_job
// cursors and emits wave_desc records.

#include "expect.h"
#include "render_job/render_job.h"
#include "store/store.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct fixture
{
  struct render_job job;
  struct read_op reads[4];
  struct chunk_plan chunks[4];
  struct read_op_group groups[4];
  struct store_read store_reads[4];
  uint8_t host_buf[256];
  uint8_t dev_buf[256];
};

static void
init_fixture(struct fixture* f)
{
  memset(f, 0, sizeof(*f));
  f->job.state = RENDER_JOB_READY;
  f->job.batch_pool_slot = 1;
  f->job.batch_id = 9;
  f->job.read_ops = f->reads;
  f->job.chunk_plans = f->chunks;
  f->job.read_op_groups = f->groups;
  f->job.n_chunks = 3;
  f->job.n_read_op_groups = 3;

  f->reads[0] = (struct read_op){ .shard_path = "a", .nbytes = 64 };
  f->reads[1] = (struct read_op){ .shard_path = "b", .nbytes = 64 };
  f->reads[2] = (struct read_op){ .shard_path = "c", .nbytes = 64 };
  f->chunks[0] = (struct chunk_plan){ .read_op_idx = 0,
                                      .decompressed_nbytes = 40,
                                      .batch_pool_slot = 1 };
  f->chunks[1] = (struct chunk_plan){ .read_op_idx = 1,
                                      .decompressed_nbytes = 50,
                                      .batch_pool_slot = 1 };
  f->chunks[2] = (struct chunk_plan){ .read_op_idx = 2,
                                      .decompressed_nbytes = 60,
                                      .batch_pool_slot = 1 };
  f->groups[0] = (struct read_op_group){
    .read_op_idx = 0, .first_chunk = 0, .n_chunks = 1, .total_decompressed = 40
  };
  f->groups[1] = (struct read_op_group){
    .read_op_idx = 1, .first_chunk = 1, .n_chunks = 1, .total_decompressed = 50
  };
  f->groups[2] = (struct read_op_group){
    .read_op_idx = 2, .first_chunk = 2, .n_chunks = 1, .total_decompressed = 60
  };
}

static int
test_reserve_commits_cursor_and_desc(void)
{
  struct fixture f;
  init_fixture(&f);
  const struct wave_pack_limits limits = {
    .host_cap = sizeof(f.host_buf),
    .dev_decompressed_cap = 200,
    .max_chunks_per_wave = 2,
    .use_gds = 0,
  };
  struct wave_desc desc;
  EXPECT(wave_dispatcher_reserve(
           &f.job, 0, &limits, f.store_reads, f.host_buf, f.dev_buf, &desc) ==
         DAMACY_OK);
  EXPECT(desc.render_job_idx == 0);
  EXPECT(desc.batch_pool_slot == 1);
  EXPECT(desc.batch_chunk_offset == 0);
  EXPECT(desc.n_chunks == 2);
  EXPECT(desc.n_reads == 2);
  EXPECT(desc.host_used_bytes == 128);
  EXPECT(f.job.n_chunks_dispatched == 2);
  EXPECT(f.job.n_groups_dispatched == 2);
  EXPECT(f.store_reads[0].dst == f.host_buf);
  EXPECT(f.chunks[1].dev_decompressed_offset == 40);
  return 0;
}

static int
test_reserve_rolls_back_cursor(void)
{
  struct fixture f;
  init_fixture(&f);
  const struct wave_pack_limits limits = {
    .host_cap = sizeof(f.host_buf),
    .dev_decompressed_cap = 200,
    .max_chunks_per_wave = 2,
    .use_gds = 0,
  };
  struct wave_desc desc;
  EXPECT(wave_dispatcher_reserve(
           &f.job, 0, &limits, f.store_reads, f.host_buf, f.dev_buf, &desc) ==
         DAMACY_OK);
  render_job_rollback_wave(&f.job, &desc);
  EXPECT(f.job.n_chunks_dispatched == 0);
  EXPECT(f.job.n_groups_dispatched == 0);
  return 0;
}

static int
test_first_group_too_large_errors_without_advancing(void)
{
  struct fixture f;
  init_fixture(&f);
  const struct wave_pack_limits limits = {
    .host_cap = sizeof(f.host_buf),
    .dev_decompressed_cap = 20,
    .max_chunks_per_wave = 2,
    .use_gds = 0,
  };
  struct wave_desc desc;
  EXPECT(wave_dispatcher_reserve(
           &f.job, 0, &limits, f.store_reads, f.host_buf, f.dev_buf, &desc) ==
         DAMACY_BUDGET);
  EXPECT(f.job.n_chunks_dispatched == 0);
  EXPECT(f.job.n_groups_dispatched == 0);
  return 0;
}

static int
test_gds_uses_device_staging_target(void)
{
  struct fixture f;
  init_fixture(&f);
  const struct wave_pack_limits limits = {
    .host_cap = sizeof(f.host_buf),
    .dev_decompressed_cap = 200,
    .max_chunks_per_wave = 1,
    .use_gds = 1,
  };
  struct wave_desc desc;
  EXPECT(wave_dispatcher_reserve(
           &f.job, 0, &limits, f.store_reads, f.host_buf, f.dev_buf, &desc) ==
         DAMACY_OK);
  EXPECT(f.store_reads[0].dst == f.dev_buf);
  EXPECT(desc.n_reads == 1);
  return 0;
}

int
main(void)
{
  RUN(test_reserve_commits_cursor_and_desc);
  RUN(test_reserve_rolls_back_cursor);
  RUN(test_first_group_too_large_errors_without_advancing);
  RUN(test_gds_uses_device_staging_target);
  printf("all wave_dispatcher tests passed\n");
  return 0;
}
