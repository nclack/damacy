#include "expect.h"
#include "query/selection.h"

#include <stdlib.h>
#include <string.h>

static int
test_sparse_grid(void)
{
  int64_t rows[] = { INT64_C(1) << 40, 3, 3, 8 };
  int64_t cols[] = { 9, 0, 9 };
  struct damacy_sample sample = { .uri = "array",
                                  .aabb = { .rank = 2 },
                                  .indices = { { rows, 4 }, { cols, 3 } } };
  struct damacy_batch_spec output = { .sample_rank = 2,
                                      .sample_shape = { 4, 3 },
                                      .samples_per_batch = 1 };
  EXPECT(query_validate(&sample, &output, 7 * sizeof(struct query_index)) ==
         DAMACY_OK);
  EXPECT(query_validate(&sample, &output, 7 * sizeof(struct query_index) - 1) ==
         DAMACY_BUDGET);
  struct damacy_aabb bounds;
  struct query_axis axes[DAMACY_MAX_RANK];
  char* storage = NULL;
  EXPECT(query_copy(&sample, &storage, &bounds, axes) == DAMACY_OK);
  memset(rows, 0, sizeof(rows));
  memset(cols, 0, sizeof(cols));
  EXPECT(!strcmp(storage, "array"));
  EXPECT(axes[0].indices[0].source == 3 && axes[0].indices[0].output == 1);
  EXPECT(axes[0].indices[1].source == 3 && axes[0].indices[1].output == 2);
  EXPECT(axes[0].indices[3].source == (INT64_C(1) << 40));
  uint64_t shape[] = { 4, 4 };
  struct selection_grid grid;
  EXPECT(selection_grid_init(&grid, &bounds, axes, shape, NULL, NULL, 5) ==
         DAMACY_BUDGET);
  EXPECT(selection_grid_init(&grid, &bounds, axes, shape, NULL, NULL, 6) ==
         DAMACY_OK);
  EXPECT(grid.count == 6);
  uint64_t expected_rows[] = { 0, 2, UINT64_C(1) << 38 };
  uint64_t coordinate[DAMACY_MAX_RANK];
  for (uint32_t i = 0; i < 6; ++i) {
    EXPECT(selection_grid_next(&grid, coordinate));
    EXPECT(coordinate[0] == expected_rows[i / 2]);
    EXPECT(coordinate[1] == (i % 2 ? 2 : 0));
  }
  EXPECT(!selection_grid_next(&grid, coordinate));
  uint64_t begin[] = { 1, 0 }, end[] = { 3, 1 };
  EXPECT(selection_grid_init(&grid, &bounds, axes, shape, begin, end, 1) ==
         DAMACY_OK);
  EXPECT(grid.count == 1);
  EXPECT(selection_grid_next(&grid, coordinate));
  EXPECT(coordinate[0] == 2 && coordinate[1] == 0);
  EXPECT(!selection_grid_next(&grid, coordinate));
  struct selection_span span = query_chunk_span(&axes[0], bounds.dims[0], 0, 4);
  EXPECT(span.begin == 0 && span.count == 2);
  span = query_chunk_span(&axes[0], bounds.dims[0], 4, 4);
  EXPECT(span.count == 0);
  free(storage);
  return 0;
}

static int
test_invalid_indices(void)
{
  int64_t values[] = { 0, 1 };
  struct damacy_sample sample = { .uri = "array",
                                  .aabb = { .rank = 1 },
                                  .indices = { { values, 2 } } };
  struct damacy_batch_spec output = { .sample_rank = 1, .sample_shape = { 2 } };
  EXPECT(query_validate(&sample, &output, 1024) == DAMACY_OK);
  values[0] = -1;
  EXPECT(query_validate(&sample, &output, 1024) == DAMACY_INVAL);
  values[0] = INT64_MAX;
  EXPECT(query_validate(&sample, &output, 1024) == DAMACY_INVAL);
  values[0] = 0;
  sample.indices[0].count = 0;
  EXPECT(query_validate(&sample, &output, 1024) == DAMACY_INVAL);
  sample.indices[0] = (struct damacy_index_array){ NULL, 2 };
  EXPECT(query_validate(&sample, &output, 1024) == DAMACY_INVAL);
  sample.indices[0] = (struct damacy_index_array){ values, 1 };
  EXPECT(query_validate(&sample, &output, 1024) == DAMACY_INVAL);
  return 0;
}

static int
test_mixed_grid(void)
{
  int64_t values[] = { 10, 1, 10 };
  struct damacy_sample sample = { .uri = "array",
                                  .aabb = { .rank = 2,
                                            .dims = { { 0, 0 }, { 3, 9 } } },
                                  .indices = { { values, 3 } } };
  struct damacy_aabb bounds;
  struct query_axis axes[DAMACY_MAX_RANK];
  char* storage = NULL;
  EXPECT(query_copy(&sample, &storage, &bounds, axes) == DAMACY_OK);
  struct selection_grid grid;
  uint64_t shape[] = { 4, 4 };
  EXPECT(selection_grid_init(&grid, &bounds, axes, shape, NULL, NULL, 6) ==
         DAMACY_OK);
  EXPECT(grid.count == 6);
  uint64_t coordinate[DAMACY_MAX_RANK];
  for (uint32_t i = 0; i < 6; ++i) {
    EXPECT(selection_grid_next(&grid, coordinate));
    EXPECT(coordinate[0] == (i < 3 ? 0 : 2));
    EXPECT(coordinate[1] == i % 3);
  }
  EXPECT(!selection_grid_next(&grid, coordinate));
  struct selection_span span = query_chunk_span(&axes[1], bounds.dims[1], 0, 4);
  EXPECT(span.begin == 3 && span.count == 1);
  span = query_chunk_span(&axes[1], bounds.dims[1], 8, 4);
  EXPECT(span.begin == 8 && span.count == 1);
  free(storage);
  return 0;
}

int
main(void)
{
  RUN(test_sparse_grid);
  RUN(test_invalid_indices);
  RUN(test_mixed_grid);
  return 0;
}
