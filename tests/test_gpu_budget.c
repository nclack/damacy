// Unit tests for src/gpu_budget/. Verifies that:
//   - gpu_budget_compute returns OK on a sane config
//   - .total equals the sum of its parts (no field drift)
//   - .total scales 2× when host/device buffer bytes double
//   - smaller max_chunk_uncompressed_bytes shrinks .nvcomp_temp
//
// Needs CUDA because wave_predict_bytes routes through
// decoder_*_query_temp_bytes (nvcomp).

#include "cuda_init.h"
#include "damacy.h"
#include "expect.h"
#include "gpu_budget/gpu_budget.h"

#include <stdio.h>

static struct damacy_config
mk_cfg(uint64_t host_bytes, uint64_t dev_bytes, uint32_t chunk_cap)
{
  return (struct damacy_config){
    .batch_size = 4,
    .lookahead_batches = 2,
    .n_io_threads = 1,
    .host_buffer_bytes = host_bytes,
    .device_buffer_bytes = dev_bytes,
    .n_zarrs_meta_cache = 4,
    .n_shards_meta_cache = 4,
    .dtype = DAMACY_F32,
    .max_chunk_uncompressed_bytes = chunk_cap,
    .max_gpu_memory_bytes = 0,
    .device = -1,
    .n_compute_threads = 0,
  };
}

static int
test_total_is_sum_of_parts(void)
{
  struct damacy_config cfg = mk_cfg(2ull << 20, 2ull << 20, 0);
  struct gpu_budget b = { 0 };
  EXPECT(gpu_budget_compute(&cfg, &b) == DAMACY_OK);
  uint64_t expect = b.dev_compressed + b.dev_decompressed +
                    b.dev_unshuffle_scratch + b.blosc1_meta + b.fanout_soa +
                    b.nvcomp_temp + b.batch_metadata;
  EXPECT(b.total == expect);
  // dev_compressed should mirror host_buffer_bytes (2× host_per_wave =
  // host_buffer_bytes); both waves accounted via the 2× in gpu_budget.
  EXPECT(b.dev_compressed == cfg.host_buffer_bytes);
  EXPECT(b.dev_decompressed == cfg.device_buffer_bytes);
  EXPECT(b.dev_unshuffle_scratch == cfg.device_buffer_bytes);
  return 0;
}

static int
test_scales_with_buffers(void)
{
  struct damacy_config small_cfg = mk_cfg(1ull << 20, 1ull << 20, 0);
  struct damacy_config big_cfg = mk_cfg(2ull << 20, 2ull << 20, 0);
  struct gpu_budget small = { 0 }, big = { 0 };
  EXPECT(gpu_budget_compute(&small_cfg, &small) == DAMACY_OK);
  EXPECT(gpu_budget_compute(&big_cfg, &big) == DAMACY_OK);
  // dev_compressed / dev_decompressed / dev_unshuffle_scratch are the
  // dominant axis-aligned terms — they double.
  EXPECT(big.dev_compressed == 2 * small.dev_compressed);
  EXPECT(big.dev_decompressed == 2 * small.dev_decompressed);
  EXPECT(big.dev_unshuffle_scratch == 2 * small.dev_unshuffle_scratch);
  // batch_metadata is independent of buffer sizes; identical across cfgs.
  EXPECT(big.batch_metadata == small.batch_metadata);
  return 0;
}

static int
test_chunk_cap_shrinks_nvcomp_temp(void)
{
  // Same buffer sizes, only the per-chunk cap differs. The smaller cap
  // bounds nvcomp's per-substream allocation, so .nvcomp_temp shrinks.
  struct damacy_config small_cap = mk_cfg(2ull << 20, 2ull << 20, 64ull << 10);
  struct damacy_config big_cap = mk_cfg(2ull << 20, 2ull << 20, 1ull << 20);
  struct gpu_budget small = { 0 }, big = { 0 };
  EXPECT(gpu_budget_compute(&small_cap, &small) == DAMACY_OK);
  EXPECT(gpu_budget_compute(&big_cap, &big) == DAMACY_OK);
  EXPECT(small.nvcomp_temp <= big.nvcomp_temp);
  // Other fields are unaffected by the chunk cap.
  EXPECT(small.dev_compressed == big.dev_compressed);
  EXPECT(small.fanout_soa == big.fanout_soa);
  return 0;
}

int
main(void)
{
  if (cuda_init_primary() != 0) {
    fprintf(stderr, "skip: CUDA not available\n");
    return 0;
  }
  RUN(test_total_is_sum_of_parts);
  RUN(test_scales_with_buffers);
  RUN(test_chunk_cap_shrinks_nvcomp_temp);
  printf("all gpu_budget tests passed\n");
  return 0;
}
