#include "damacy_pipeline.h"
#include "log/log.h"

enum damacy_status
damacy_cuda_executor_create(struct damacy_reader* reader,
                            const struct damacy_cuda_config* config,
                            struct damacy_executor** out)
{
  (void)reader;
  (void)config;
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  log_error("CUDA support is disabled in this build");
  return DAMACY_CUDA;
}

void
damacy_config_describe(const struct damacy_config* config)
{
  (void)config;
  log_info("CUDA support is disabled in this build");
}
