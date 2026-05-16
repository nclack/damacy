// GPUDirect Storage (cuFile) store. Composes a host store_fs underneath
// for stat/submit/map and overrides submit_dev with cuFileReadAsync.
// store_fs.c has no GDS coupling — damacy.c picks one or the other.
// DAMACY_ENABLE_GDS=OFF links store_fs_gds_stub.c which makes
// store_fs_gds_create return NULL.
#pragma once

struct store;
struct store_fs_config;

#ifdef __cplusplus
extern "C"
{
#endif

  // NULL on driver init failure, missing libcufile, or stub build.
  struct store* store_fs_gds_create(const struct store_fs_config* cfg);

  // cuFileReadAsync rides this stream. Caller owns it.
  void store_fs_gds_set_stream(struct store* s, void* stream);

#ifdef __cplusplus
}
#endif
