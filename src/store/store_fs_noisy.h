// Bench-only: store_fs wrapped with a per-read sleep. See issue #79.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct store;
  struct store_fs_config;

  // Pre-read sleep is drawn from lognormal(mu, sigma) where mu and sigma
  // are fit to (mean_ms, p99_ms). jitter_ms is an extra uniform
  // [-jitter_ms, +jitter_ms] term; the sum clamps at zero.
  struct store_fs_noisy_params
  {
    double mean_ms;
    double p99_ms;
    double jitter_ms;
    uint64_t seed;
  };

  struct store* store_fs_noisy_create(const struct store_fs_config* cfg,
                                      const struct store_fs_noisy_params* np);

  // Reads DAMACY_NOISY_{MEAN,P99,JITTER}_MS and DAMACY_NOISY_SEED.
  // Returns 1 iff any of the *_MS knobs is set.
  int store_fs_noisy_params_from_env(struct store_fs_noisy_params* out);

#ifdef __cplusplus
}
#endif
