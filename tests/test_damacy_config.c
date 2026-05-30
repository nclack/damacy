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

int
main(void)
{
  RUN(test_resolve_enable_gds_auto_defers_to_env);
  RUN(test_resolve_enable_gds_on_overrides_env);
  RUN(test_resolve_enable_gds_off_overrides_env);
  RUN(test_resolve_enable_gds_zero_init_is_auto);
  RUN(test_resolve_prefetch_io_threads_default);
  RUN(test_resolve_prefetch_io_threads_explicit);
  log_info("all tests passed");
  return 0;
}
