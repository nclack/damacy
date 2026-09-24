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
  uint64_t cell_size_index;
  struct
  {
    uint64_t beg;
    uint64_t end;
  } span_cell;
  uint64_t current_cell;
  struct
  {
    uint32_t beg;
    uint32_t end;
  } span_entry;
  uint32_t current_entry;
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

int
query_has_indices(const struct damacy_sample* sample);

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
                    const struct damacy_aabb* bounds_index,
                    const struct query_axis* axes,
                    const uint64_t* cell_shape_index,
                    const uint64_t* clip_beg_cell,
                    const uint64_t* clip_end_cell,
                    uint64_t max_cells);

int
selection_grid_next(struct selection_grid* grid, uint64_t* coordinate_cell);
