// DAMACY_ENABLE_GDS=OFF link target.
#include "store/store_fs_gds.h"

#include <stddef.h>

struct store*
store_fs_gds_create(const struct store_fs_gds_config* cfg)
{
  (void)cfg;
  return NULL;
}

void
store_fs_gds_set_stream(struct store* s, void* stream)
{
  (void)s;
  (void)stream;
}

void
store_fs_gds_stats_get(struct store* s, struct lru_stats* out)
{
  (void)s;
  (void)out;
}
