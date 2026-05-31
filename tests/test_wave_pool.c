// Unit tests for the host-side parts of src/wave/wave_pool.c.
// wave_pool_init needs CUDA (creates streams); input_reserve and
// input_commit don't — they touch stats, render_job cursors, input_slot
// state, and chunk_plans / read_op_groups, all stack-constructible.

#include "batch_pool/batch_pool.h"
#include "damacy.h"
#include "damacy_limits.h"
#include "damacy_log.h"
#include "expect.h"
#include "planner/planner.h"
#include "render_job/render_job.h"
#include "store/store.h"
#include "wave/input_slot.h"
#include "wave/wave_input.h"
#include "wave/wave_pool.h"
#include "zarr/zarr_metadata.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static struct render_job*
job0(struct render_job_pool* jobs)
{
  return render_job_pool_for_batch_slot(jobs, 0);
}

// Construct just enough wave_pool state for input_commit to run. The slot
// is wired as if input_reserve had just been called: SLOT_RESERVED, n_chunks
// set, batch_pool_slot points at slot 0 of the batch pool. Counters are
// pre-incremented to reflect a completed reserve.
static void
setup_post_reserve(struct wave_pool* wp,
                   struct damacy_batch_pool* batch_pool,
                   struct render_job_pool* jobs,
                   struct damacy_stats* stats,
                   struct input_slot* slot,
                   uint32_t n_chunks)
{
  memset(wp, 0, sizeof(*wp));
  memset(batch_pool, 0, sizeof(*batch_pool));
  memset(jobs, 0, sizeof(*jobs));
  memset(stats, 0, sizeof(*stats));
  memset(slot, 0, sizeof(*slot));

  wp->pool = batch_pool;
  wp->render_jobs = jobs;
  wp->stats = stats;
  wp->slots[0] = *slot; // placeholder; we use the pointer below

  // Wire the fields input_commit actually reads.
  wp->slots[0].state = SLOT_RESERVED;
  wp->slots[0].render_job_idx = 0;
  wp->slots[0].batch_pool_slot = 0;
  wp->slots[0].n_chunks = n_chunks;

  job0(jobs)->n_chunks_dispatched = n_chunks;
  stats->waves_emitted = 1;
  stats->chunks_dispatched = n_chunks;
}

static int
test_input_commit_rollback_once(void)
{
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct render_job_pool jobs;
  struct damacy_stats stats;
  struct input_slot slot;
  const uint32_t n_chunks = 3;
  setup_post_reserve(&wp, &batch_pool, &jobs, &stats, &slot, n_chunks);

  // n_reads > 0 + ev.seq == 0 → submit-failure rollback path.
  struct wave_input_reservation t = { .active = 1,
                                      .input_slot_idx = 0,
                                      .n_reads = 2,
                                      .desc = { .render_job_idx = 0,
                                                .n_chunks = n_chunks,
                                                .prev_n_groups_dispatched = 0 },
                                      .committed = 0 };
  struct store_event ev = { .seq = 0 };

  int changed = 0;
  EXPECT(wave_input_commit(&wp, &t, ev, &changed) == DAMACY_IO);
  EXPECT(changed == 1);
  EXPECT(t.committed == 1);
  EXPECT(stats.waves_emitted == 0);
  EXPECT(stats.chunks_dispatched == 0);
  EXPECT(job0(&jobs)->n_chunks_dispatched == 0);
  EXPECT(wp.slots[0].state == SLOT_FREE);

  // Second call must NOT decrement again — that would underflow.
  // Re-entry intentionally trips a log_error; quiet the sink across it.
  changed = 0;
  damacy_log_set_quiet(1);
  EXPECT(wave_input_commit(&wp, &t, ev, &changed) == DAMACY_OK);
  damacy_log_set_quiet(0);
  EXPECT(changed == 0);
  EXPECT(stats.waves_emitted == 0);
  EXPECT(stats.chunks_dispatched == 0);
  EXPECT(job0(&jobs)->n_chunks_dispatched == 0);
  return 0;
}

static int
test_input_commit_success_then_recommit(void)
{
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct render_job_pool jobs;
  struct damacy_stats stats;
  struct input_slot slot;
  const uint32_t n_chunks = 4;
  setup_post_reserve(&wp, &batch_pool, &jobs, &stats, &slot, n_chunks);

  // Successful submit: ev.seq != 0 → slot advances to SLOT_IO.
  struct wave_input_reservation t = {
    .active = 1, .input_slot_idx = 0, .n_reads = 3, .committed = 0
  };
  struct store_event ev = { .seq = 42 };

  int changed = 0;
  EXPECT(wave_input_commit(&wp, &t, ev, &changed) == DAMACY_OK);
  EXPECT(changed == 1);
  EXPECT(t.committed == 1);
  EXPECT(wp.slots[0].state == SLOT_IO);
  EXPECT(wp.slots[0].io_event.seq == 42);
  // Success path leaves counters as the reserve set them.
  EXPECT(stats.waves_emitted == 1);
  EXPECT(stats.chunks_dispatched == n_chunks);

  // Second call: a stray re-commit (e.g. error retry) must not touch state.
  // Re-entry intentionally trips a log_error; quiet the sink across it.
  struct store_event ev2 = { .seq = 0 };
  changed = 0;
  damacy_log_set_quiet(1);
  EXPECT(wave_input_commit(&wp, &t, ev2, &changed) == DAMACY_OK);
  damacy_log_set_quiet(0);
  EXPECT(changed == 0);
  EXPECT(wp.slots[0].state == SLOT_IO);
  EXPECT(stats.waves_emitted == 1);
  EXPECT(stats.chunks_dispatched == n_chunks);
  return 0;
}

// Submit-failure rollback must restore n_groups_dispatched from the
// snapshot the ticket captured at reserve time — without it the next
// input dispatch would skip past the groups this wave never actually issued.
static int
test_input_commit_rolls_back_groups(void)
{
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct render_job_pool jobs;
  struct damacy_stats stats;
  struct input_slot slot;
  const uint32_t n_chunks = 5;
  setup_post_reserve(&wp, &batch_pool, &jobs, &stats, &slot, n_chunks);

  // Simulate reserve having advanced from group 7 to group 9.
  job0(&jobs)->n_groups_dispatched = 9;

  struct wave_input_reservation t = { .active = 1,
                                      .input_slot_idx = 0,
                                      .n_reads = 1,
                                      .desc = { .render_job_idx = 0,
                                                .n_chunks = n_chunks,
                                                .prev_n_groups_dispatched = 7 },
                                      .committed = 0 };
  struct store_event ev = { .seq = 0 };

  int changed = 0;
  EXPECT(wave_input_commit(&wp, &t, ev, &changed) == DAMACY_IO);
  EXPECT(changed == 1);
  EXPECT(job0(&jobs)->n_groups_dispatched == 7);
  return 0;
}

// Stack-constructible scaffold for input_reserve. All planner-output
// arrays are inline so a test can populate groups/chunks/read_ops and
// fire reserve without touching CUDA or the heap.
struct reserve_fixture
{
  struct wave_pool wp;
  struct damacy_batch_pool pool;
  struct render_job_pool jobs;
  struct damacy_stats stats;
  uint8_t host_buf[8192];
  struct store_read store_reads[DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE];
  struct read_op read_ops[8];
  struct chunk_plan chunk_plans[8];
  struct read_op_group groups[4];
};

static void
init_reserve_fixture(struct reserve_fixture* f,
                     uint64_t host_cap,
                     uint64_t dev_cap)
{
  memset(f, 0, sizeof(*f));
  f->wp.pool = &f->pool;
  f->wp.render_jobs = &f->jobs;
  f->wp.stats = &f->stats;
  f->wp.n_slots = 1;
  f->wp.input = input_transfer_host_staging();
  f->wp.max_chunks_per_wave = DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE;
  f->wp.waves[0].dev_decompressed_cap = dev_cap;
  f->wp.slots[0].state = SLOT_FREE;
  f->wp.slots[0].cap = host_cap;
  f->wp.slots[0].buf = f->host_buf;
  f->wp.slots[0].store_reads = f->store_reads;
  struct render_job* job = job0(&f->jobs);
  job->state = RENDER_JOB_READY;
  job->batch_pool_slot = 0;
  job->chunk_plans = f->chunk_plans;
  job->read_ops = f->read_ops;
  job->read_op_groups = f->groups;
}

// dev_cap blocks group 1 after group 0 lands; group 1 must defer whole.
static int
test_input_reserve_defers_oversize_group(void)
{
  struct reserve_fixture f;
  init_reserve_fixture(&f, 4096, 200);

  f.read_ops[0].shard_path = "a";
  f.read_ops[0].nbytes = 64;
  f.read_ops[1].shard_path = "b";
  f.read_ops[1].nbytes = 64;
  f.chunk_plans[0].read_op_idx = 0;
  f.chunk_plans[0].decompressed_nbytes = 100;
  f.chunk_plans[1].read_op_idx = 1;
  f.chunk_plans[1].decompressed_nbytes = 200;
  f.groups[0] = (struct read_op_group){
    .read_op_idx = 0, .first_chunk = 0, .n_chunks = 1, .total_decompressed = 100
  };
  f.groups[1] = (struct read_op_group){
    .read_op_idx = 1, .first_chunk = 1, .n_chunks = 1, .total_decompressed = 200
  };
  job0(&f.jobs)->n_chunks = 2;
  job0(&f.jobs)->n_read_op_groups = 2;

  struct wave_input_reservation t = { 0 };
  enum damacy_status status = wave_input_reserve(&f.wp, 0, &t);

  EXPECT(status == DAMACY_OK);
  EXPECT(t.active == 1);
  EXPECT(t.input_slot_idx == 0);
  EXPECT(t.n_reads == 1);
  EXPECT(t.desc.prev_n_groups_dispatched == 0);
  EXPECT(job0(&f.jobs)->n_chunks_dispatched == 1);
  EXPECT(job0(&f.jobs)->n_groups_dispatched == 1);
  EXPECT(f.wp.slots[0].n_chunks == 1);
  EXPECT(f.wp.slots[0].state == SLOT_RESERVED);
  return 0;
}

// First group exceeds dev_cap on a fresh wave → DAMACY_BUDGET.
static int
test_input_reserve_errors_when_first_group_too_big(void)
{
  struct reserve_fixture f;
  init_reserve_fixture(&f, 4096, 50);

  f.read_ops[0].shard_path = "a";
  f.read_ops[0].nbytes = 64;
  f.chunk_plans[0].read_op_idx = 0;
  f.chunk_plans[0].decompressed_nbytes = 100;
  f.groups[0] = (struct read_op_group){
    .read_op_idx = 0, .first_chunk = 0, .n_chunks = 1, .total_decompressed = 100
  };
  job0(&f.jobs)->n_chunks = 1;
  job0(&f.jobs)->n_read_op_groups = 1;

  struct wave_input_reservation t = { 0 };
  damacy_log_set_quiet(1);
  enum damacy_status status = wave_input_reserve(&f.wp, 0, &t);
  damacy_log_set_quiet(0);

  EXPECT(status == DAMACY_BUDGET);
  EXPECT(t.active == 0);
  EXPECT(job0(&f.jobs)->n_chunks_dispatched == 0);
  EXPECT(job0(&f.jobs)->n_groups_dispatched == 0);
  return 0;
}

static int
test_input_reserve_inactive_when_not_ready(void)
{
  struct reserve_fixture f;
  init_reserve_fixture(&f, 4096, 200);

  struct wave_input_reservation t = { 0 };
  EXPECT(wave_input_reserve(&f.wp, 0, &t) == DAMACY_OK);
  EXPECT(t.active == 0);

  job0(&f.jobs)->n_chunks = 1;
  f.wp.slots[0].state = SLOT_BUSY;
  EXPECT(wave_input_reserve(&f.wp, 0, &t) == DAMACY_OK);
  EXPECT(t.active == 0);
  return 0;
}

static int
test_input_commit_noop_ticket(void)
{
  // Inactive reservations must not touch state.
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct render_job_pool jobs;
  struct damacy_stats stats;
  memset(&wp, 0, sizeof(wp));
  memset(&batch_pool, 0, sizeof(batch_pool));
  memset(&jobs, 0, sizeof(jobs));
  memset(&stats, 0, sizeof(stats));
  wp.pool = &batch_pool;
  wp.render_jobs = &jobs;
  wp.stats = &stats;

  struct wave_input_reservation t = {
    .active = 0, .input_slot_idx = 0, .n_reads = 0, .committed = 0
  };
  struct store_event ev = { .seq = 0 };
  int changed = 0;
  EXPECT(wave_input_commit(&wp, &t, ev, &changed) == DAMACY_OK);
  EXPECT(changed == 0);
  EXPECT(stats.waves_emitted == 0);
  EXPECT(stats.chunks_dispatched == 0);
  return 0;
}

// Unprobed BLOSC_ZSTD reaching the sizer means the eligibility gate
// failed to reject it; prepare_decode_caps treats that as DAMACY_INVAL.
static int
test_chunk_substreams_unprobed_blosc(void)
{
  struct sample_plan sp_unprobed = { .layout_probed = 0 };
  struct sample_plan sp_probed = { .layout_probed = 1,
                                   .layout = { .nblocks = 5 } };

  uint32_t n = 0xDEADBEEF;
  {
    struct chunk_plan c = { .codec_id = (uint8_t)CODEC_BLOSC_ZSTD,
                            .is_fill = 0 };
    EXPECT(chunk_substreams_upper_bound(&c, &sp_unprobed, &n) == DAMACY_INVAL);
    EXPECT(n == 0xDEADBEEF);
  }
  {
    struct chunk_plan c = { .codec_id = (uint8_t)CODEC_BLOSC_ZSTD,
                            .is_fill = 0 };
    EXPECT(chunk_substreams_upper_bound(&c, &sp_probed, &n) == DAMACY_OK);
    EXPECT(n == 5);
  }
  {
    struct chunk_plan c = { .codec_id = (uint8_t)CODEC_BLOSC_ZSTD,
                            .is_fill = 1 };
    EXPECT(chunk_substreams_upper_bound(&c, &sp_unprobed, &n) == DAMACY_OK);
    EXPECT(n == 0);
  }
  {
    struct chunk_plan c = { .codec_id = (uint8_t)CODEC_ZSTD, .is_fill = 0 };
    EXPECT(chunk_substreams_upper_bound(&c, &sp_unprobed, &n) == DAMACY_OK);
    EXPECT(n == 1);
  }
  return 0;
}

int
main(void)
{
  RUN(test_input_commit_rollback_once);
  RUN(test_input_commit_success_then_recommit);
  RUN(test_input_commit_rolls_back_groups);
  RUN(test_input_reserve_defers_oversize_group);
  RUN(test_input_reserve_errors_when_first_group_too_big);
  RUN(test_input_reserve_inactive_when_not_ready);
  RUN(test_input_commit_noop_ticket);
  RUN(test_chunk_substreams_unprobed_blosc);
  printf("all wave_pool tests passed\n");
  return 0;
}
