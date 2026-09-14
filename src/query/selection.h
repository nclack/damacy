#pragma once

#include "damacy_pipeline.h"

struct query_index
{
  int64_t source;
  uint32_t output;
};

struct query_axis
{
  const struct query_index* indices;
  uint32_t count;
};

struct selection_grid_axis
{
  const struct query_index* indices;
  uint64_t extent;
  uint64_t begin;
  uint64_t end;
  uint64_t cell;
  uint32_t first;
  uint32_t last;
  uint32_t position;
};

struct selection_grid
{
  struct selection_grid_axis axes[DAMACY_MAX_RANK];
  uint64_t count;
  uint8_t rank;
  uint8_t finished;
};

struct selection_span
{
  uint64_t begin;
  uint64_t count;
};

enum damacy_status
query_validate(const struct damacy_sample* sample,
               const struct damacy_batch_spec* output,
               uint64_t max_index_bytes);

enum damacy_status
query_copy(const struct damacy_sample* sample,
           char** uri,
           struct damacy_aabb* bounds,
           struct query_axis* axes);

uint32_t
query_lower_bound(const struct query_axis* axis, uint64_t source);

struct selection_span
query_chunk_span(const struct query_axis* axis,
                 struct damacy_interval bounds,
                 uint64_t origin,
                 uint64_t extent);

enum damacy_status
selection_grid_init(struct selection_grid* grid,
                    const struct damacy_aabb* bounds,
                    const struct query_axis* axes,
                    const uint64_t* shape,
                    const uint64_t* clip_begin,
                    const uint64_t* clip_end,
                    uint64_t max_cells);

int
selection_grid_next(struct selection_grid* grid, uint64_t* coordinate);
