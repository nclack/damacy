#include "gpu_budget.h"

#include "log/log.h"

#include <stdlib.h>

struct gpu_budget
{
  uint64_t max_bytes;
  uint64_t committed;
};

struct gpu_budget*
gpu_budget_new(uint64_t max_bytes)
{
  struct gpu_budget* b = (struct gpu_budget*)calloc(1, sizeof(*b));
  if (!b)
    return NULL;
  b->max_bytes = max_bytes;
  b->committed = 0;
  return b;
}

void
gpu_budget_destroy(struct gpu_budget* b)
{
  free(b);
}

enum damacy_status
gpu_budget_try_commit(struct gpu_budget* b, uint64_t bytes, const char* tag)
{
  if (!b || bytes == 0)
    return DAMACY_OK;
  if (b->max_bytes > 0 && b->committed + bytes > b->max_bytes) {
    log_error("%s would exceed GPU budget: committed=%llu add=%llu cap=%llu",
              tag ? tag : "gpu_budget",
              (unsigned long long)b->committed,
              (unsigned long long)bytes,
              (unsigned long long)b->max_bytes);
    return DAMACY_OOM;
  }
  b->committed += bytes;
  return DAMACY_OK;
}

void
gpu_budget_commit(struct gpu_budget* b, uint64_t bytes)
{
  if (!b)
    return;
  b->committed += bytes;
}

void
gpu_budget_release(struct gpu_budget* b, uint64_t bytes)
{
  if (!b)
    return;
  if (bytes > b->committed)
    b->committed = 0;
  else
    b->committed -= bytes;
}

uint64_t
gpu_budget_committed(const struct gpu_budget* b)
{
  return b ? b->committed : 0;
}

uint64_t
gpu_budget_max(const struct gpu_budget* b)
{
  return b ? b->max_bytes : 0;
}

uint64_t
gpu_budget_set_committed_for_test(struct gpu_budget* b, uint64_t v)
{
  if (!b)
    return 0;
  const uint64_t prev = b->committed;
  b->committed = v;
  return prev;
}
