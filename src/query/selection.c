#include "query/selection.h"

#include <stdlib.h>
#include <string.h>

static enum damacy_status
axis_length(const struct damacy_axis_selection* axis, uint64_t* length)
{
  switch (axis->kind) {
    case DAMACY_AXIS_INTERVAL:
      if (axis->interval.beg < 0 || axis->interval.end <= axis->interval.beg)
        return DAMACY_INVAL;
      *length = (uint64_t)(axis->interval.end - axis->interval.beg);
      return DAMACY_OK;
    case DAMACY_AXIS_INDICES:
      if (!axis->indices.values || !axis->indices.count)
        return DAMACY_INVAL;
      *length = axis->indices.count;
      return DAMACY_OK;
    default:
      return DAMACY_INVAL;
  }
}

enum damacy_status
query_validate(const struct damacy_sample* sample,
               const struct damacy_batch_spec* output,
               uint64_t max_index_bytes)
{
  if (!sample || !sample->uri || !output)
    return DAMACY_INVAL;
  if (!sample->rank || sample->rank > DAMACY_MAX_RANK ||
      sample->rank != output->sample_rank)
    return DAMACY_RANK;
  uint64_t bytes = 0;
  for (uint8_t d = 0; d < sample->rank; ++d) {
    const struct damacy_axis_selection* axis = &sample->axes[d];
    uint64_t length;
    enum damacy_status status = axis_length(axis, &length);
    if (status != DAMACY_OK)
      return status;
    if (length != (uint64_t)output->sample_shape[d])
      return DAMACY_INVAL;
    if (axis->kind == DAMACY_AXIS_INDICES) {
      uint64_t size =
        (uint64_t)axis->indices.count * sizeof(struct query_index);
      if (size > max_index_bytes || bytes > max_index_bytes - size)
        return DAMACY_BUDGET;
      bytes += size;
      for (uint32_t i = 0; i < axis->indices.count; ++i)
        if (axis->indices.values[i] < 0 || axis->indices.values[i] == INT64_MAX)
          return DAMACY_INVAL;
    }
  }
  return DAMACY_OK;
}

static int
compare_indices(const void* left, const void* right)
{
  const struct query_index* a = left;
  const struct query_index* b = right;
  if (a->source != b->source)
    return (a->source > b->source) - (a->source < b->source);
  return (a->output > b->output) - (a->output < b->output);
}

enum damacy_status
query_copy(const struct damacy_sample* sample,
           char** uri,
           struct damacy_aabb* bounds,
           struct query_axis* axes)
{
  if (!sample || !sample->uri || !sample->rank ||
      sample->rank > DAMACY_MAX_RANK)
    return DAMACY_INVAL;
  size_t uri_bytes = strlen(sample->uri) + 1;
  size_t alignment = _Alignof(struct query_index);
  if (uri_bytes > SIZE_MAX - alignment + 1)
    return DAMACY_BUDGET;
  size_t offset = (uri_bytes + alignment - 1) / alignment * alignment;
  uint64_t count = 0;
  for (uint8_t d = 0; d < sample->rank; ++d) {
    uint64_t length;
    enum damacy_status status = axis_length(&sample->axes[d], &length);
    if (status != DAMACY_OK)
      return status;
    if (sample->axes[d].kind == DAMACY_AXIS_INDICES)
      count += length;
  }
  if (count > (SIZE_MAX - offset) / sizeof(struct query_index))
    return DAMACY_BUDGET;
  char* storage = malloc(offset + (size_t)count * sizeof(struct query_index));
  if (!storage)
    return DAMACY_OOM;
  memcpy(storage, sample->uri, uri_bytes);
  *bounds = (struct damacy_aabb){ .rank = sample->rank };
  memset(axes, 0, DAMACY_MAX_RANK * sizeof(*axes));
  struct query_index* cursor = (void*)(storage + offset);
  for (uint8_t d = 0; d < sample->rank; ++d) {
    const struct damacy_axis_selection* axis = &sample->axes[d];
    if (axis->kind == DAMACY_AXIS_INTERVAL) {
      bounds->dims[d] = axis->interval;
      continue;
    }
    uint32_t n = axis->indices.count;
    for (uint32_t i = 0; i < n; ++i) {
      int64_t value = axis->indices.values[i];
      if (value < 0 || value == INT64_MAX) {
        free(storage);
        return DAMACY_INVAL;
      }
      cursor[i] = (struct query_index){ .source = value, .output = i };
    }
    qsort(cursor, n, sizeof(*cursor), compare_indices);
    bounds->dims[d] =
      (struct damacy_interval){ cursor[0].source, cursor[n - 1].source + 1 };
    axes[d] = (struct query_axis){ .indices = cursor, .count = n };
    cursor += n;
  }
  *uri = storage;
  return DAMACY_OK;
}

uint32_t
query_lower_bound(const struct query_axis* axis, uint64_t source)
{
  uint32_t lo = 0;
  uint32_t hi = axis->count;
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if ((uint64_t)axis->indices[mid].source < source)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

struct selection_span
query_chunk_span(const struct query_axis* axis,
                 struct damacy_interval bounds,
                 uint64_t origin,
                 uint64_t extent)
{
  uint64_t end = origin > UINT64_MAX - extent ? UINT64_MAX : origin + extent;
  if (axis->count) {
    uint32_t begin = query_lower_bound(axis, origin);
    return (struct selection_span){ begin,
                                    query_lower_bound(axis, end) - begin };
  }
  uint64_t begin =
    origin > (uint64_t)bounds.beg ? origin : (uint64_t)bounds.beg;
  if (end > (uint64_t)bounds.end)
    end = (uint64_t)bounds.end;
  return (struct selection_span){ begin, end > begin ? end - begin : 0 };
}

static uint64_t
cell_start(uint64_t cell, uint64_t extent)
{
  return cell > UINT64_MAX / extent ? UINT64_MAX : cell * extent;
}

enum damacy_status
selection_grid_init(struct selection_grid* grid,
                    const struct damacy_aabb* bounds,
                    const struct query_axis* axes,
                    const uint64_t* shape,
                    const uint64_t* clip_begin,
                    const uint64_t* clip_end,
                    uint64_t max_cells)
{
  *grid = (struct selection_grid){ .rank = bounds->rank, .count = 1 };
  if (!bounds->rank || bounds->rank > DAMACY_MAX_RANK)
    return DAMACY_RANK;
  for (uint8_t d = 0; d < bounds->rank; ++d) {
    if (bounds->dims[d].beg < 0 || bounds->dims[d].end <= bounds->dims[d].beg ||
        !shape[d])
      return DAMACY_INVAL;
    struct selection_grid_axis* axis = &grid->axes[d];
    axis->extent = shape[d];
    axis->begin = (uint64_t)bounds->dims[d].beg / shape[d];
    axis->end = ((uint64_t)bounds->dims[d].end - 1) / shape[d] + 1;
    if (clip_begin && axis->begin < clip_begin[d])
      axis->begin = clip_begin[d];
    if (clip_end && axis->end > clip_end[d])
      axis->end = clip_end[d];
    uint64_t count = axis->end > axis->begin ? axis->end - axis->begin : 0;
    axis->cell = axis->begin;
    if (axes[d].count && count) {
      axis->indices = axes[d].indices;
      axis->first =
        query_lower_bound(&axes[d], cell_start(axis->begin, shape[d]));
      axis->last = query_lower_bound(&axes[d], cell_start(axis->end, shape[d]));
      axis->position = axis->first;
      count = 0;
      uint64_t previous = UINT64_MAX;
      for (uint32_t i = axis->first; i < axis->last; ++i) {
        uint64_t cell = (uint64_t)axis->indices[i].source / shape[d];
        count += cell != previous;
        previous = cell;
      }
      if (count)
        axis->cell = (uint64_t)axis->indices[axis->first].source / shape[d];
    }
    if (!count) {
      grid->count = 0;
      grid->finished = 1;
    } else {
      if (grid->count > max_cells / count)
        return DAMACY_BUDGET;
      grid->count *= count;
    }
  }
  return DAMACY_OK;
}

int
selection_grid_next(struct selection_grid* grid, uint64_t* coordinate)
{
  if (grid->finished)
    return 0;
  for (uint8_t d = 0; d < grid->rank; ++d)
    coordinate[d] = grid->axes[d].cell;
  for (int d = grid->rank - 1; d >= 0; --d) {
    struct selection_grid_axis* axis = &grid->axes[d];
    if (axis->indices) {
      while (++axis->position < axis->last) {
        uint64_t cell =
          (uint64_t)axis->indices[axis->position].source / axis->extent;
        if (cell != axis->cell) {
          axis->cell = cell;
          return 1;
        }
      }
      axis->position = axis->first;
      axis->cell = (uint64_t)axis->indices[axis->first].source / axis->extent;
    } else {
      if (++axis->cell < axis->end)
        return 1;
      axis->cell = axis->begin;
    }
  }
  grid->finished = 1;
  return 1;
}
