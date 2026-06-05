// CUDA-backed guardrails for wave_pool_advance phase ordering.

#include "batch_pool/batch_pool.h"
#include "cuda_init.h"
#include "damacy.h"
#include "damacy_limits.h"
#include "damacy_log.h"
#include "expect.h"
#include "gpu_budget/gpu_budget.h"
#include "render_job/render_job.h"
#include "wave/input_slot.h"
#include "wave/wave_pool.h"
#include "zarr/zarr_metadata.h"

#include <cuda.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static volatile int g_host_func_ran;

static int
create_wave_events(struct damacy_wave* wave)
{
  return cuEventCreate(&wave->ev.input_start, CU_EVENT_DEFAULT) !=
           CUDA_SUCCESS ||
         cuEventCreate(&wave->ev.input_transfer_done, CU_EVENT_DEFAULT) !=
           CUDA_SUCCESS ||
         cuEventCreate(&wave->ev.input_parse_done, CU_EVENT_DEFAULT) !=
           CUDA_SUCCESS ||
         cuEventCreate(&wave->ev.decomp_start, CU_EVENT_DEFAULT) !=
           CUDA_SUCCESS ||
         cuEventCreate(&wave->ev.decode_done, CU_EVENT_DEFAULT) !=
           CUDA_SUCCESS ||
         cuEventCreate(&wave->ev.decomp_end, CU_EVENT_DEFAULT) !=
           CUDA_SUCCESS ||
         cuEventCreate(&wave->ev.asm_start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
         cuEventCreate(&wave->ev.asm_end, CU_EVENT_DEFAULT) != CUDA_SUCCESS;
}

static void
destroy_wave_events(struct damacy_wave* wave)
{
  CUevent* events[] = {
    &wave->ev.input_start,      &wave->ev.input_transfer_done,
    &wave->ev.input_parse_done, &wave->ev.decomp_start,
    &wave->ev.decode_done,      &wave->ev.decomp_end,
    &wave->ev.asm_start,        &wave->ev.asm_end
  };
  for (size_t i = 0; i < sizeof(events) / sizeof(events[0]); ++i) {
    if (*events[i]) {
      cuEventDestroy(*events[i]);
      *events[i] = NULL;
    }
  }
}

static int
record_retired_wave_events(struct damacy_wave* wave, CUstream stream)
{
  return cuEventRecord(wave->ev.input_start, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.input_transfer_done, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.input_parse_done, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.decomp_start, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.decode_done, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.decomp_end, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.asm_start, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.asm_end, stream) != CUDA_SUCCESS ||
         cuStreamSynchronize(stream) != CUDA_SUCCESS;
}

static void CUDA_CB
sleeping_host_func(void* user_data)
{
  (void)user_data;
  const struct timespec delay = { .tv_sec = 0, .tv_nsec = 50 * 1000 * 1000 };
  nanosleep(&delay, NULL);
  g_host_func_ran = 1;
}

static int
test_freed_wave_does_not_bind_until_next_tick(void)
{
  CUstream stream = NULL;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);

  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct render_job_pool jobs;
  struct damacy_stats stats;
  struct blosc1_totals totals[DAMACY_N_WAVES] = { 0 };
  memset(&wp, 0, sizeof(wp));
  memset(&batch_pool, 0, sizeof(batch_pool));
  memset(&jobs, 0, sizeof(jobs));
  memset(&stats, 0, sizeof(stats));

  wp.pool = &batch_pool;
  wp.render_jobs = &jobs;
  wp.stats = &stats;
  wp.n_slots = 1;

  for (int w = 0; w < DAMACY_N_WAVES; ++w) {
    struct damacy_wave* wave = &wp.waves[w];
    EXPECT(create_wave_events(wave) == 0);
    EXPECT(record_retired_wave_events(wave, stream) == 0);
    wave->state = WAVE_POST;
    wave->render_job_idx = (uint16_t)w;
    wave->batch_pool_slot = (uint16_t)w;
    wave->n_chunks = 1;
    wave->bound_slot = -1;
    wave->h_blosc1_totals = &totals[w];

    batch_pool.slots[w].state = BATCH_RENDERING;
    batch_pool.slots[w].chunks_remaining = 1;
  }

  // A ready slot is available, but both waves are occupied when the bind phase
  // runs. The waves retire later in this tick; the slot must remain ready until
  // the next call instead of rebinding a same-tick WAVE_POST -> WAVE_FREE wave.
  wp.slots[0].state = SLOT_READY;

  int changed = 0;
  EXPECT(wave_pool_advance(&wp, &changed) == DAMACY_OK);
  EXPECT(changed == 1);
  for (int w = 0; w < DAMACY_N_WAVES; ++w) {
    EXPECT(wp.waves[w].state == WAVE_FREE);
    EXPECT(batch_pool.slots[w].state == BATCH_READY);
  }
  EXPECT(wp.slots[0].state == SLOT_READY);
  EXPECT(wp.slots[0].state != SLOT_BUSY);

  for (int w = 0; w < DAMACY_N_WAVES; ++w)
    destroy_wave_events(&wp.waves[w]);
  cuStreamDestroy(stream);
  return 0;
}

static int
test_failed_host_staging_submit_drains_before_unbind(void)
{
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct render_job_pool jobs;
  struct damacy_stats stats;
  struct gpu_budget* budget = gpu_budget_new(UINT64_MAX);
  struct chunk_plan chunk_plans[1];
  struct sample_plan sample_plans[1];
  memset(&wp, 0, sizeof(wp));
  memset(&batch_pool, 0, sizeof(batch_pool));
  memset(&jobs, 0, sizeof(jobs));
  memset(&stats, 0, sizeof(stats));
  memset(chunk_plans, 0, sizeof(chunk_plans));
  memset(sample_plans, 0, sizeof(sample_plans));
  EXPECT(budget != NULL);

  EXPECT(wave_pool_init(&wp,
                        &batch_pool,
                        &jobs,
                        NULL,
                        &stats,
                        DAMACY_F32,
                        DAMACY_N_WAVES,
                        1,
                        1,
                        4096,
                        4096,
                        4096,
                        input_transfer_host_staging(),
                        0,
                        budget) == 0);

  struct render_job* job = render_job_pool_get(&jobs, 0);
  EXPECT(job != NULL);
  job->chunk_plans = chunk_plans;
  job->sample_plans = sample_plans;
  chunk_plans[0].is_fill = 1;
  chunk_plans[0].codec_id = (uint8_t)CODEC_FILL;
  chunk_plans[0].decompressed_nbytes = 1;

  batch_pool.rank = 1;
  wp.slots[0].state = SLOT_READY;
  wp.slots[0].render_job_idx = 0;
  wp.slots[0].batch_pool_slot = 0;
  wp.slots[0].n_chunks = 1;
  wp.slots[0].used_bytes = 1;
  wp.slots[0].io_bytes = 1;

  g_host_func_ran = 0;
  EXPECT(cuLaunchHostFunc(wp.stream_input, sleeping_host_func, NULL) ==
         CUDA_SUCCESS);

  void* saved_parse_chunks = wp.waves[0].d_parse_chunks;
  wp.waves[0].d_parse_chunks = NULL;

  int changed = 0;
  damacy_log_set_quiet(1);
  EXPECT(wave_pool_advance(&wp, &changed) == DAMACY_CUDA);
  damacy_log_set_quiet(0);

  EXPECT(changed == 0);
  EXPECT(g_host_func_ran == 1);
  EXPECT(cuStreamQuery(wp.stream_input) == CUDA_SUCCESS);
  EXPECT(wp.slots[0].state == SLOT_FREE);
  EXPECT(wp.waves[0].state == WAVE_FREE);
  EXPECT(wp.waves[0].bound_slot == -1);
  EXPECT(wp.waves[0].host_input == NULL);

  wp.waves[0].d_parse_chunks = saved_parse_chunks;
  wave_pool_destroy(&wp, 0);
  gpu_budget_destroy(budget);
  return 0;
}

static int
test_failed_bulk_host_staging_submit_drains_before_unbind(void)
{
  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct render_job_pool jobs;
  struct damacy_stats stats;
  struct gpu_budget* budget = gpu_budget_new(UINT64_MAX);
  struct chunk_plan chunk_plans[1];
  struct sample_plan sample_plans[1];
  memset(&wp, 0, sizeof(wp));
  memset(&batch_pool, 0, sizeof(batch_pool));
  memset(&jobs, 0, sizeof(jobs));
  memset(&stats, 0, sizeof(stats));
  memset(chunk_plans, 0, sizeof(chunk_plans));
  memset(sample_plans, 0, sizeof(sample_plans));
  EXPECT(budget != NULL);

  EXPECT(wave_pool_init(&wp,
                        &batch_pool,
                        &jobs,
                        NULL,
                        &stats,
                        DAMACY_F32,
                        DAMACY_N_WAVES,
                        1,
                        1,
                        4096,
                        4096,
                        4096,
                        input_transfer_host_staging(),
                        0,
                        budget) == 0);

  struct render_job* job = render_job_pool_get(&jobs, 0);
  EXPECT(job != NULL);
  job->chunk_plans = chunk_plans;
  job->sample_plans = sample_plans;
  chunk_plans[0].is_fill = 1;
  chunk_plans[0].codec_id = (uint8_t)CODEC_FILL;
  chunk_plans[0].decompressed_nbytes = 1;

  batch_pool.rank = 1;
  wp.slots[0].state = SLOT_READY;
  wp.slots[0].render_job_idx = 0;
  wp.slots[0].batch_pool_slot = 0;
  wp.slots[0].n_chunks = 1;
  wp.slots[0].used_bytes = 1;
  wp.slots[0].io_bytes = 1;

  g_host_func_ran = 0;
  EXPECT(cuLaunchHostFunc(wp.stream_input, sleeping_host_func, NULL) ==
         CUDA_SUCCESS);

  void* saved_dev_compressed = wp.waves[0].dev_compressed;
  void* saved_dev_compressed_owned = wp.waves[0].dev_compressed_owned;
  wp.waves[0].dev_compressed = NULL;
  wp.waves[0].dev_compressed_owned = NULL;

  int changed = 0;
  damacy_log_set_quiet(1);
  EXPECT(wave_pool_advance(&wp, &changed) == DAMACY_CUDA);
  damacy_log_set_quiet(0);

  EXPECT(changed == 0);
  EXPECT(g_host_func_ran == 1);
  EXPECT(cuStreamQuery(wp.stream_input) == CUDA_SUCCESS);
  EXPECT(wp.slots[0].state == SLOT_FREE);
  EXPECT(wp.waves[0].state == WAVE_FREE);
  EXPECT(wp.waves[0].bound_slot == -1);
  EXPECT(wp.waves[0].host_input == NULL);

  wp.waves[0].dev_compressed = saved_dev_compressed;
  wp.waves[0].dev_compressed_owned = saved_dev_compressed_owned;
  wave_pool_destroy(&wp, 0);
  gpu_budget_destroy(budget);
  return 0;
}

static int
test_host_staging_slot_releases_after_input_transfer_done(void)
{
  CUstream stream = NULL;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);

  struct wave_pool wp;
  memset(&wp, 0, sizeof(wp));
  wp.input = input_transfer_host_staging();
  wp.n_slots = 1;
  wp.waves[0].state = WAVE_INPUT;
  wp.waves[0].bound_slot = 0;
  wp.waves[0].dev_compressed = (void*)1;
  wp.waves[0].dev_compressed_owned = (void*)1;
  wp.slots[0].state = SLOT_BUSY;

  EXPECT(create_wave_events(&wp.waves[0]) == 0);
  EXPECT(cuEventRecord(wp.waves[0].ev.input_transfer_done, stream) ==
         CUDA_SUCCESS);
  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  g_host_func_ran = 0;
  EXPECT(cuLaunchHostFunc(stream, sleeping_host_func, NULL) == CUDA_SUCCESS);
  EXPECT(cuEventRecord(wp.waves[0].ev.input_parse_done, stream) ==
         CUDA_SUCCESS);

  int changed = 0;
  EXPECT(wave_pool_advance(&wp, &changed) == DAMACY_OK);
  EXPECT(changed == 1);
  EXPECT(wp.waves[0].bound_slot == -1);
  EXPECT(wp.slots[0].state == SLOT_FREE);
  EXPECT(wp.waves[0].dev_compressed == wp.waves[0].dev_compressed_owned);
  EXPECT(wp.waves[0].state == WAVE_INPUT);
  EXPECT(g_host_func_ran == 0);

  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  destroy_wave_events(&wp.waves[0]);
  cuStreamDestroy(stream);
  return 0;
}

static int
test_gds_slot_stays_bound_before_input_parse_done(void)
{
  CUstream stream = NULL;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);

  struct wave_pool wp;
  memset(&wp, 0, sizeof(wp));
  wp.input = input_transfer_gds();
  wp.n_slots = 1;
  wp.waves[0].state = WAVE_INPUT;
  wp.waves[0].bound_slot = 0;
  wp.slots[0].state = SLOT_BUSY;

  EXPECT(create_wave_events(&wp.waves[0]) == 0);
  EXPECT(cuEventRecord(wp.waves[0].ev.input_transfer_done, stream) ==
         CUDA_SUCCESS);
  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  g_host_func_ran = 0;
  EXPECT(cuLaunchHostFunc(stream, sleeping_host_func, NULL) == CUDA_SUCCESS);
  EXPECT(cuEventRecord(wp.waves[0].ev.input_parse_done, stream) ==
         CUDA_SUCCESS);

  int changed = 0;
  EXPECT(wave_pool_advance(&wp, &changed) == DAMACY_OK);
  EXPECT(changed == 0);
  EXPECT(wp.waves[0].bound_slot == 0);
  EXPECT(wp.slots[0].state == SLOT_BUSY);
  EXPECT(g_host_func_ran == 0);

  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  destroy_wave_events(&wp.waves[0]);
  cuStreamDestroy(stream);
  return 0;
}

static int
test_gds_slot_releases_after_decomp_end(void)
{
  CUstream stream = NULL;
  CUstream asm_stream = NULL;
  EXPECT(cuStreamCreate(&stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);
  EXPECT(cuStreamCreate(&asm_stream, CU_STREAM_DEFAULT) == CUDA_SUCCESS);

  struct wave_pool wp;
  struct damacy_batch_pool batch_pool;
  struct render_job_pool jobs;
  struct damacy_stats stats;
  struct blosc1_totals totals = { 0 };
  memset(&wp, 0, sizeof(wp));
  memset(&batch_pool, 0, sizeof(batch_pool));
  memset(&jobs, 0, sizeof(jobs));
  memset(&stats, 0, sizeof(stats));
  wp.input = input_transfer_gds();
  wp.pool = &batch_pool;
  wp.render_jobs = &jobs;
  wp.stats = &stats;
  wp.n_slots = 1;
  wp.waves[0].state = WAVE_POST;
  wp.waves[0].bound_slot = 0;
  wp.waves[0].dev_compressed = (void*)1;
  wp.waves[0].h_blosc1_totals = &totals;
  wp.waves[0].n_chunks = 1;
  wp.slots[0].state = SLOT_BUSY;
  batch_pool.slots[0].state = BATCH_RENDERING;
  batch_pool.slots[0].chunks_remaining = 1;

  EXPECT(create_wave_events(&wp.waves[0]) == 0);
  EXPECT(cuEventRecord(wp.waves[0].ev.decomp_end, stream) == CUDA_SUCCESS);
  EXPECT(cuStreamSynchronize(stream) == CUDA_SUCCESS);
  g_host_func_ran = 0;
  EXPECT(cuLaunchHostFunc(asm_stream, sleeping_host_func, NULL) ==
         CUDA_SUCCESS);
  EXPECT(cuEventRecord(wp.waves[0].ev.asm_end, asm_stream) == CUDA_SUCCESS);
  int changed = 0;
  EXPECT(wave_pool_advance(&wp, &changed) == DAMACY_OK);
  EXPECT(changed == 1);
  EXPECT(wp.waves[0].bound_slot == -1);
  EXPECT(wp.slots[0].state == SLOT_FREE);
  EXPECT(wp.waves[0].dev_compressed == NULL);

  EXPECT(cuStreamSynchronize(asm_stream) == CUDA_SUCCESS);
  destroy_wave_events(&wp.waves[0]);
  cuStreamDestroy(asm_stream);
  cuStreamDestroy(stream);
  return 0;
}

int
main(void)
{
  if (cuda_init_primary() != 0) {
    fprintf(stderr, "skip: CUDA not available\n");
    return 0;
  }
  RUN(test_freed_wave_does_not_bind_until_next_tick);
  RUN(test_failed_host_staging_submit_drains_before_unbind);
  RUN(test_failed_bulk_host_staging_submit_drains_before_unbind);
  RUN(test_host_staging_slot_releases_after_input_transfer_done);
  RUN(test_gds_slot_stays_bound_before_input_parse_done);
  RUN(test_gds_slot_releases_after_decomp_end);
  printf("all wave_pool_advance tests passed\n");
  return 0;
}
