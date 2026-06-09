#include "damacy.h"
#include "damacy_config.h"
#include "damacy_limits.h"
#include "expect.h"
#include "platform/platform.h"

#include <stdlib.h>

static struct damacy_config
mk_cfg(enum damacy_gds_mode mode)
{
  struct damacy_config cfg = { 0 };
  cfg.tuning.enable_gds = mode;
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
test_tuning_defaults_thread_counts(void)
{
  struct damacy_tuning tuning = damacy_tuning_defaults();
  EXPECT(tuning.n_io_threads == 64);
  EXPECT(tuning.metadata_io_concurrency == 64);
  EXPECT(tuning.max_read_op_bytes == 4 * 1024 * 1024);
  EXPECT(tuning.max_chunk_uncompressed_bytes == 2 * 1024 * 1024);
  EXPECT(tuning.n_array_meta_cache == DAMACY_DEFAULT_ARRAY_META_CACHE);
  EXPECT(tuning.n_shard_index_cache == DAMACY_DEFAULT_SHARD_INDEX_CACHE);
  EXPECT(tuning.n_chunk_layout_cache == DAMACY_DEFAULT_CHUNK_LAYOUT_CACHE);
  return 0;
}

static int
test_resolve_metadata_io_concurrency_explicit(void)
{
  struct damacy_config cfg = { 0 };
  cfg.tuning.metadata_io_concurrency = 11;
  EXPECT(resolve_metadata_io_concurrency(&cfg) == 11);
  return 0;
}

static struct damacy_config
mk_valid_cfg(void)
{
  struct damacy_config cfg = {
    .dtype = DAMACY_F32,
    .sample_shape = { 1 },
    .sample_rank = 1,
    .samples_per_batch = 1,
    .lookahead_samples = 1,
    .device = -1,
  };
  cfg.tuning = damacy_tuning_defaults();
  cfg.tuning.max_gpu_memory_bytes = 1ull << 20;
  return cfg;
}

static int
test_validate_accepts_tuning_defaults(void)
{
  struct damacy_config cfg = mk_valid_cfg();
  EXPECT(validate_config(&cfg) == DAMACY_OK);
  return 0;
}

static int
test_validate_metadata_io_concurrency_reject_zero(void)
{
  struct damacy_config cfg = mk_valid_cfg();
  cfg.tuning.metadata_io_concurrency = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  return 0;
}

static int
test_validate_tuning_fields_reject_out_of_range(void)
{
  struct damacy_config cfg = mk_valid_cfg();
  EXPECT(validate_config(&cfg) == DAMACY_OK);

  cfg = mk_valid_cfg();
  cfg.tuning.max_chunk_uncompressed_bytes = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);

  cfg = mk_valid_cfg();
  cfg.tuning.max_read_op_bytes = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg.tuning.max_read_op_bytes = (uint64_t)UINT32_MAX + 1;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);

  cfg = mk_valid_cfg();
  cfg.tuning.host_buffer_waves = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg.tuning.host_buffer_waves = DAMACY_N_WAVES - 1;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg.tuning.host_buffer_waves = DAMACY_MAX_HOST_BUFFER_WAVES + 1;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);

  cfg = mk_valid_cfg();
  cfg.tuning.max_chunks_per_wave = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg.tuning.max_chunks_per_wave = DAMACY_HARD_MAX_CHUNKS_PER_WAVE + 1;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);

  cfg = mk_valid_cfg();
  cfg.tuning.max_substreams_per_chunk = 0;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg.tuning.max_substreams_per_chunk = DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK + 1;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  return 0;
}

static int
test_resolvers_return_literal_value(void)
{
  struct damacy_config cfg = { 0 };
  cfg.tuning.max_chunk_uncompressed_bytes = 1u << 20;
  cfg.tuning.max_read_op_bytes = 1ull << 21;
  cfg.tuning.host_buffer_waves = DAMACY_N_WAVES + 1;
  cfg.tuning.max_chunks_per_wave = 7;
  cfg.tuning.max_substreams_per_chunk = 9;
  EXPECT(resolve_max_chunk_uncompressed(&cfg) == (1u << 20));
  EXPECT(resolve_max_read_op_bytes(&cfg) == (1ull << 21));
  EXPECT(resolve_host_buffer_waves(&cfg) == DAMACY_N_WAVES + 1);
  EXPECT(resolve_max_chunks_per_wave(&cfg) == 7);
  EXPECT(resolve_max_substreams_per_chunk(&cfg) == 9);
  return 0;
}

static int
test_validate_io_threads_bound_by_machine(void)
{
  uint32_t too_many = (uint32_t)platform_default_thread_count() + 1u;
  struct damacy_config cfg = mk_valid_cfg();
  cfg.tuning.n_io_threads = 1;
  cfg.tuning.metadata_io_concurrency = too_many;
  EXPECT(validate_config(&cfg) == DAMACY_OK);
  cfg.tuning.metadata_io_concurrency = DAMACY_MAX_METADATA_IO_CONCURRENCY + 1u;
  EXPECT(validate_config(&cfg) == DAMACY_INVAL);
  cfg.tuning.metadata_io_concurrency = 1;
  cfg.tuning.n_io_threads = too_many;
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
  RUN(test_tuning_defaults_thread_counts);
  RUN(test_resolve_metadata_io_concurrency_explicit);
  RUN(test_validate_accepts_tuning_defaults);
  RUN(test_validate_metadata_io_concurrency_reject_zero);
  RUN(test_validate_tuning_fields_reject_out_of_range);
  RUN(test_resolvers_return_literal_value);
  RUN(test_validate_io_threads_bound_by_machine);
  log_info("all tests passed");
  return 0;
}
