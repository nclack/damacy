#include "damacy.h"
#include "damacy_config.h"
#include "damacy_limits.h"
#include "expect.h"

#include <stdlib.h>

static struct damacy_config
mk_cfg(enum damacy_gds_mode mode)
{
  struct damacy_config cfg = { 0 };
  cfg.tuning.enable_gds = mode;
  return cfg;
}

static struct damacy_config
mk_valid_cfg(void)
{
  struct damacy_config cfg = {
    .dtype = DAMACY_F32,
    .sample_rank = 2,
    .samples_per_batch = 4,
    .lookahead_samples = 4,
    .device = -1,
    .tuning = {
      .max_gpu_memory_bytes = 1ull << 30,
      .max_chunk_uncompressed_bytes =
        DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES,
      .max_read_op_bytes = DAMACY_DEFAULT_READ_OP_MAX_BYTES,
      .host_buffer_waves = DAMACY_DEFAULT_HOST_BUFFER_WAVES,
      .max_chunks_per_wave = DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE,
      .max_substreams_per_chunk =
        DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK,
      .n_io_threads = 1,
      .n_prefetch_io_threads = DAMACY_DEFAULT_PREFETCH_IO_THREADS,
      .n_array_meta_cache = 4,
      .n_shard_index_cache = 4,
      .n_chunk_layout_cache = 4,
      .numa_node = -1,
    },
  };
  cfg.sample_shape[0] = 8;
  cfg.sample_shape[1] = 16;
  return cfg;
}

static int
test_resolve_enable_gds_auto_defers_to_env(void)
{
  struct damacy_config cfg = mk_cfg(DAMACY_GDS_AUTO);

  unsetenv("DAMACY_GDS_ENABLE");
  EXPECT(resolve_enable_gds(&cfg) == 0);

  setenv("DAMACY_GDS_ENABLE", "1", 1);
  EXPECT(resolve_enable_gds(&cfg) == 1);

  setenv("DAMACY_GDS_ENABLE", "0", 1);
  EXPECT(resolve_enable_gds(&cfg) == 0);

  setenv("DAMACY_GDS_ENABLE", "true", 1);
  EXPECT(resolve_enable_gds(&cfg) == 0);

  unsetenv("DAMACY_GDS_ENABLE");
  return 0;
}

static int
test_resolve_enable_gds_on_overrides_env(void)
{
  struct damacy_config cfg = mk_cfg(DAMACY_GDS_ON);

  unsetenv("DAMACY_GDS_ENABLE");
  EXPECT(resolve_enable_gds(&cfg) == 1);

  setenv("DAMACY_GDS_ENABLE", "0", 1);
  EXPECT(resolve_enable_gds(&cfg) == 1);

  unsetenv("DAMACY_GDS_ENABLE");
  return 0;
}

static int
test_resolve_enable_gds_off_overrides_env(void)
{
  struct damacy_config cfg = mk_cfg(DAMACY_GDS_OFF);

  unsetenv("DAMACY_GDS_ENABLE");
  EXPECT(resolve_enable_gds(&cfg) == 0);

  setenv("DAMACY_GDS_ENABLE", "1", 1);
  EXPECT(resolve_enable_gds(&cfg) == 0);

  unsetenv("DAMACY_GDS_ENABLE");
  return 0;
}

static int
test_resolve_enable_gds_zero_init_is_auto(void)
{
  struct damacy_config cfg = { 0 };
  setenv("DAMACY_GDS_ENABLE", "1", 1);
  EXPECT(resolve_enable_gds(&cfg) == 1);
  unsetenv("DAMACY_GDS_ENABLE");
  return 0;
}

static int
test_resolve_prefetch_io_threads_default(void)
{
  struct damacy_config cfg = { 0 };
  EXPECT(resolve_n_prefetch_io_threads(&cfg) ==
         DAMACY_DEFAULT_PREFETCH_IO_THREADS);
  return 0;
}

static int
test_resolve_prefetch_io_threads_explicit(void)
{
  struct damacy_config cfg = { 0 };
  cfg.tuning.n_prefetch_io_threads = 7;
  EXPECT(resolve_n_prefetch_io_threads(&cfg) == 7);
  return 0;
}

static int
test_defaults_helper_fills_defaultable_knobs(void)
{
  struct damacy_config cfg = mk_valid_cfg();
  cfg.lookahead_samples = 0;
  cfg.tuning.max_chunk_uncompressed_bytes = 0;
  cfg.tuning.max_read_op_bytes = 0;
  cfg.tuning.host_buffer_waves = 0;
  cfg.tuning.max_chunks_per_wave = 0;
  cfg.tuning.max_substreams_per_chunk = 0;
  cfg.tuning.n_prefetch_io_threads = 0;

  struct damacy_config out = damacy_config_validate_with_defaults(&cfg);
  EXPECT(out.lookahead_samples == 2u * cfg.samples_per_batch);
  EXPECT(out.tuning.max_chunk_uncompressed_bytes ==
         DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES);
  EXPECT(out.tuning.max_read_op_bytes == DAMACY_DEFAULT_READ_OP_MAX_BYTES);
  EXPECT(out.tuning.host_buffer_waves == DAMACY_DEFAULT_HOST_BUFFER_WAVES);
  EXPECT(out.tuning.max_chunks_per_wave == DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE);
  EXPECT(out.tuning.max_substreams_per_chunk ==
         DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK);
  EXPECT(out.tuning.n_prefetch_io_threads ==
         DAMACY_DEFAULT_PREFETCH_IO_THREADS);
  EXPECT(validate_config(&out) == DAMACY_OK);
  return 0;
}

static int
test_validate_rejects_implicit_numeric_defaults(void)
{
  struct damacy_config cfg = mk_valid_cfg();

  cfg.tuning.max_chunk_uncompressed_bytes = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg = mk_valid_cfg();
  cfg.tuning.max_read_op_bytes = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg = mk_valid_cfg();
  cfg.tuning.host_buffer_waves = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg = mk_valid_cfg();
  cfg.tuning.max_chunks_per_wave = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg = mk_valid_cfg();
  cfg.tuning.max_substreams_per_chunk = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg = mk_valid_cfg();
  cfg.tuning.n_prefetch_io_threads = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  return 0;
}

static int
test_validate_accepts_single_batch_lookahead(void)
{
  struct damacy_config cfg = mk_valid_cfg();
  cfg.lookahead_samples = cfg.samples_per_batch;
  EXPECT(validate_config(&cfg) == DAMACY_OK);

  cfg.lookahead_samples = cfg.samples_per_batch - 1u;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  return 0;
}

int
main(void)
{
  RUN(test_resolve_enable_gds_auto_defers_to_env);
  RUN(test_resolve_enable_gds_on_overrides_env);
  RUN(test_resolve_enable_gds_off_overrides_env);
  RUN(test_resolve_enable_gds_zero_init_is_auto);
  RUN(test_resolve_prefetch_io_threads_default);
  RUN(test_resolve_prefetch_io_threads_explicit);
  RUN(test_defaults_helper_fills_defaultable_knobs);
  RUN(test_validate_rejects_implicit_numeric_defaults);
  RUN(test_validate_accepts_single_batch_lookahead);
  log_info("all tests passed");
  return 0;
}
