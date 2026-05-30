// CUDA-backed guardrails for wave_pool_advance phase ordering.

#include "batch_pool/batch_pool.h"
#include "cuda_init.h"
#include "damacy.h"
#include "damacy_limits.h"
#include "expect.h"
#include "render_job/render_job.h"
#include "wave/host_slab.h"
#include "wave/wave_pool.h"

#include <cuda.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int
create_wave_events(struct damacy_wave* wave)
{
  return cuEventCreate(&wave->ev.h2d_start, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
         cuEventCreate(&wave->ev.bulk_h2d_end, CU_EVENT_DEFAULT) !=
           CUDA_SUCCESS ||
         cuEventCreate(&wave->ev.h2d_end, CU_EVENT_DEFAULT) != CUDA_SUCCESS ||
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
  CUevent* events[] = { &wave->ev.h2d_start,   &wave->ev.bulk_h2d_end,
                        &wave->ev.h2d_end,     &wave->ev.decomp_start,
                        &wave->ev.decode_done, &wave->ev.decomp_end,
                        &wave->ev.asm_start,   &wave->ev.asm_end };
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
  return cuEventRecord(wave->ev.h2d_start, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.bulk_h2d_end, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.h2d_end, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.decomp_start, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.decode_done, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.decomp_end, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.asm_start, stream) != CUDA_SUCCESS ||
         cuEventRecord(wave->ev.asm_end, stream) != CUDA_SUCCESS ||
         cuStreamSynchronize(stream) != CUDA_SUCCESS;
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

int
main(void)
{
  if (cuda_init_primary() != 0) {
    fprintf(stderr, "skip: CUDA not available\n");
    return 0;
  }
  RUN(test_freed_wave_does_not_bind_until_next_tick);
  printf("all wave_pool_advance tests passed\n");
  return 0;
}
