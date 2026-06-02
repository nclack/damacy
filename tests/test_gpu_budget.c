// Unit tests for src/gpu_budget/. Verifies that:
//   - gpu_budget_predict returns OK on a sane config
//   - .total equals the sum of its parts (no field drift)
//   - .total scales with the resolved per-wave geometry
//   - smaller max_chunk_uncompressed_bytes shrinks .nvcomp_temp
//
// Needs CUDA because wave_predict_bytes routes through
// decoder_*_query_temp_bytes (nvcomp).

#include "cuda_init.h"
#include "damacy.h"
#include "expect.h"
#include "wave/wave_budget.h"

#include <stdio.h>

static struct damacy_config
mk_cfg(uint32_t chunk_cap)
{
  return (struct damacy_config){
    .samples_per_batch = 4,
    .lookahead_samples = 8,
    .dtype = DAMACY_F32,
    .device = -1,
    .tuning = {
      .n_io_threads = 1,
      .n_prefetch_threads = 1,
      .n_metadata_io_threads = 1,
      .n_array_meta_cache = 4,
      .n_shard_index_cache = 4,
      .n_chunk_layout_cache = 4,
      .max_chunk_uncompressed_bytes = chunk_cap,
      .max_gpu_memory_bytes = 0,
    },
  };
}

static int
test_total_is_sum_of_parts(void)
{
  struct damacy_config cfg = mk_cfg(0);
  struct gpu_budget_breakdown b = { 0 };
  const uint64_t host = 2ull << 20;
  const uint64_t dev = 2ull << 20;
  const struct input_transfer_resources input = input_transfer_resources(
    input_transfer_host_staging(), DAMACY_N_WAVES, host);
  EXPECT(gpu_budget_predict(&cfg, &input, dev, &b) == DAMACY_OK);
  uint64_t expect = b.dev_compressed + b.dev_decompressed + b.blosc1_meta +
                    b.fanout_soa + b.nvcomp_temp + b.batch_metadata;
  EXPECT(b.total == expect);
  EXPECT(b.dev_compressed == 2 * host);
  EXPECT(b.dev_decompressed == 2 * dev);
  return 0;
}

static int
test_scales_with_buffers(void)
{
  struct damacy_config cfg = mk_cfg(0);
  struct gpu_budget_breakdown small = { 0 }, big = { 0 };
  const struct input_transfer_resources small_input = input_transfer_resources(
    input_transfer_host_staging(), DAMACY_N_WAVES, 1ull << 20);
  const struct input_transfer_resources big_input = input_transfer_resources(
    input_transfer_host_staging(), DAMACY_N_WAVES, 2ull << 20);
  EXPECT(gpu_budget_predict(&cfg, &small_input, 1ull << 20, &small) ==
         DAMACY_OK);
  EXPECT(gpu_budget_predict(&cfg, &big_input, 2ull << 20, &big) == DAMACY_OK);
  EXPECT(big.dev_compressed == 2 * small.dev_compressed);
  EXPECT(big.dev_decompressed == 2 * small.dev_decompressed);
  // batch_metadata is independent of buffer sizes; identical across cfgs.
  EXPECT(big.batch_metadata == small.batch_metadata);
  return 0;
}

static int
test_chunk_cap_shrinks_nvcomp_temp(void)
{
  // Same buffer sizes, only the per-chunk cap differs. The smaller cap
  // bounds nvcomp's per-substream allocation, so .nvcomp_temp shrinks.
  // dev buffer is sized so big_cap's per-wave decompressed arena
  // exceeds small_cap's chunks_cap_bytes (max_chunks_per_wave ×
  // max_chunk_uncompressed_bytes), so the per-substream cap actually
  // moves nvcomp's scratch.
  struct damacy_config small_cap = mk_cfg(64ull << 10);
  struct damacy_config big_cap = mk_cfg(1ull << 20);
  struct gpu_budget_breakdown small = { 0 }, big = { 0 };
  const uint64_t host = 2ull << 20;
  const uint64_t dev = 256ull << 20;
  const struct input_transfer_resources input = input_transfer_resources(
    input_transfer_host_staging(), DAMACY_N_WAVES, host);
  EXPECT(gpu_budget_predict(&small_cap, &input, dev, &small) == DAMACY_OK);
  EXPECT(gpu_budget_predict(&big_cap, &input, dev, &big) == DAMACY_OK);
  EXPECT(small.nvcomp_temp < big.nvcomp_temp);
  EXPECT(small.dev_compressed == big.dev_compressed);
  EXPECT(small.fanout_soa == big.fanout_soa);
  return 0;
}

static int
test_gds_counts_slot_device_staging(void)
{
  struct damacy_config cfg = mk_cfg(0);
  const uint64_t staging = 2ull << 20;
  const uint64_t dev = 2ull << 20;
  struct gpu_budget_breakdown host_staging = { 0 }, gds = { 0 };
  const struct input_transfer_resources host_staging_input =
    input_transfer_resources(input_transfer_host_staging(), 4, staging);
  const struct input_transfer_resources gds_input =
    input_transfer_resources(input_transfer_gds(), 4, staging);
  EXPECT(gpu_budget_predict(&cfg, &host_staging_input, dev, &host_staging) ==
         DAMACY_OK);
  EXPECT(gpu_budget_predict(&cfg, &gds_input, dev, &gds) == DAMACY_OK);
  EXPECT(host_staging.dev_compressed == 2 * staging);
  EXPECT(gds.dev_compressed == 4 * staging);
  EXPECT(gds.total == host_staging.total + 2 * staging);
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
  RUN(test_gds_counts_slot_device_staging);
  printf("all gpu_budget tests passed\n");
  return 0;
}
