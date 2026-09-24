#pragma once

#include "damacy_pipeline.h"
#include "zarr/zarr_metadata.h"

struct plan_array
{
  const char* uri;
  struct zarr_metadata metadata;
};

struct plan_chunk
{
  const char* path;
  uint64_t offset;
  uint64_t coordinate[DAMACY_MAX_RANK];
  uint32_t array;
  uint32_t encoded_bytes;
  uint32_t decoded_bytes;
  uint32_t first_use;
  uint8_t missing;
};

enum plan_operation
{
  PLAN_COPY,
};

struct plan_region
{
  enum plan_operation operation;
  uint32_t array;
  uint32_t sample;
  struct damacy_aabb source;
};

struct plan_use
{
  uint32_t chunk;
  uint32_t region;
  uint32_t next;
};

struct prepared_plan
{
  struct damacy_batch_spec output;
  struct plan_array* arrays;
  struct plan_chunk* chunks;
  struct plan_region* regions;
  struct plan_use* uses;
  uint32_t n_arrays;
  uint32_t n_chunks;
  uint32_t n_regions;
  uint32_t n_uses;
  uint64_t allocated_bytes;
  void* storage;
};

void
prepared_plan_destroy(struct prepared_plan* plan);
