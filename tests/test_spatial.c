#include "damacy_spatial.h"
#include "expect.h"
#include "fixture.h"
#include "ngff/ngff.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char group[] =
  "{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":{\"ome\":{"
  "\"version\":\"0.5\",\"multiscales\":[{\"axes\":["
  "{\"name\":\"y\",\"type\":\"space\",\"unit\":\"micrometer\"},"
  "{\"name\":\"x\",\"type\":\"space\",\"unit\":\"micrometer\"}],"
  "\"coordinateTransformations\":[{\"type\":\"scale\",\"scale\":[3,5]},"
  "{\"type\":\"translation\",\"translation\":[80,-20]}],\"datasets\":["
  "{\"path\":\"0\",\"coordinateTransformations\":["
  "{\"type\":\"scale\",\"scale\":[0.5,1]},"
  "{\"type\":\"translation\",\"translation\":[10,20]}]},"
  "{\"path\":\"1\",\"coordinateTransformations\":["
  "{\"type\":\"scale\",\"scale\":[1,2]},"
  "{\"type\":\"translation\",\"translation\":[10.25,20.5]}]},"
  "{\"path\":\"2\",\"coordinateTransformations\":["
  "{\"type\":\"scale\",\"scale\":[2,4]},"
  "{\"type\":\"translation\",\"translation\":[10,20]}]}]}]}}}";

static void
array_json(char* dst, size_t capacity, int size)
{
  snprintf(dst,
           capacity,
           "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[%d,%d],"
           "\"dimension_names\":[\"y\",\"x\"],\"data_type\":\"uint16\","
           "\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_"
           "shape\":[8,8]}},"
           "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{"
           "\"separator\":\"/\"}},"
           "\"fill_value\":0,\"codecs\":[{\"name\":\"bytes\",\"configuration\":"
           "{\"endian\":\"little\"}}]}",
           size,
           size);
}

static struct cslice
text_slice(const char* text)
{
  return (struct cslice){ text, text + strlen(text) };
}

static int
image_create(struct damacy_ngff_image** image)
{
  EXPECT(ngff_parse_group(text_slice(group), "volume", 0, 3, image) ==
         DAMACY_OK);
  for (uint32_t i = 0; i < 3; ++i) {
    char array[1024];
    array_json(array, sizeof(array), 64 >> i);
    EXPECT(ngff_parse_array(text_slice(array), *image, i) == DAMACY_OK);
  }
  return 0;
}

static struct damacy_spatial_query
identity_query(void)
{
  return (struct damacy_spatial_query){
    .output_to_reference = { .linear = { { 1, 0 }, { 0, 1 } } },
    .sampler = { .filter = DAMACY_FILTER_NEAREST,
                 .boundary = DAMACY_BOUNDARY_ERROR },
    .level = DAMACY_LEVEL_AUTO
  };
}

static struct damacy_batch_spec output = { .dtype = DAMACY_F32,
                                           .sample_shape = { 4, 4 },
                                           .sample_rank = 2,
                                           .samples_per_batch = 1 };

static int
test_corner_conversion_and_copy(void)
{
  struct damacy_ngff_image* image;
  EXPECT(!image_create(&image));
  const struct damacy_ngff_info* metadata = damacy_ngff_image_info(image);
  EXPECT(metadata->rank == 2 && metadata->level_count == 3);
  EXPECT(!strcmp(metadata->data_type, "uint16"));
  EXPECT(!strcmp(metadata->axes[0].unit, "micrometer"));
  EXPECT(!strcmp(metadata->levels[1].uri, "volume/1"));
  for (uint8_t d = 0; d < 2; ++d) {
    EXPECT(metadata->levels[0].scale_to_reference[d] == 1);
    EXPECT(metadata->levels[0].origin_reference_index[d] == 0);
    EXPECT(metadata->levels[1].scale_to_reference[d] == 2);
    EXPECT(metadata->levels[1].origin_reference_index[d] == 0);
    EXPECT(metadata->levels[2].origin_reference_index[d] == -1.5);
  }
  struct damacy_spatial_query query = identity_query();
  struct damacy_spatial_resolution* resolved;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  EXPECT(damacy_spatial_resolution_info(resolved)->level == 0);
  struct damacy_sample sample;
  EXPECT(damacy_spatial_resolution_sample(resolved, &sample) == DAMACY_OK);
  EXPECT(sample.axes[0].interval.beg == 0 && sample.axes[0].interval.end == 4);
  damacy_spatial_resolution_destroy(resolved);
  query.output_to_reference.linear[0][0] = 2;
  query.output_to_reference.linear[1][1] = 2;
  query.output_to_reference.offset[0] = 4;
  query.output_to_reference.offset[1] = 6;
  query.sampler.filter = DAMACY_FILTER_LINEAR;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  const struct damacy_spatial_info* info =
    damacy_spatial_resolution_info(resolved);
  EXPECT(info->level == 1 && !info->requires_resampling);
  damacy_ngff_image_destroy(image);
  memset(&query, 0, sizeof(query));
  EXPECT(damacy_spatial_resolution_sample(resolved, &sample) == DAMACY_OK);
  EXPECT(!strcmp(sample.uri, "volume/1"));
  EXPECT(sample.rank == 2 && sample.axes[0].kind == DAMACY_AXIS_INTERVAL);
  EXPECT(sample.axes[0].interval.beg == 2 && sample.axes[0].interval.end == 6);
  EXPECT(sample.axes[1].interval.beg == 3 && sample.axes[1].interval.end == 7);
  damacy_spatial_resolution_destroy(resolved);
  return 0;
}

static int
test_scale_rotation_and_shear(void)
{
  struct damacy_ngff_image* image;
  EXPECT(!image_create(&image));
  struct damacy_spatial_query query = identity_query();
  query.output_to_reference.offset[0] = query.output_to_reference.offset[1] =
    12;
  double s = sqrt(2.0);
  query.output_to_reference.linear[0][0] = s;
  query.output_to_reference.linear[0][1] = -s;
  query.output_to_reference.linear[1][0] = s;
  query.output_to_reference.linear[1][1] = s;
  struct damacy_spatial_resolution* resolved;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  const struct damacy_spatial_info* info =
    damacy_spatial_resolution_info(resolved);
  EXPECT(info->level == 1 && info->requires_resampling);
  EXPECT(fabs(info->output_to_source.linear[0][0] - s / 2) < 1e-15);
  struct damacy_sample sample;
  EXPECT(damacy_spatial_resolution_sample(resolved, &sample) ==
         DAMACY_UNSUPPORTED);
  EXPECT(!sample.uri && !sample.rank);
  damacy_spatial_resolution_destroy(resolved);
  query.output_to_reference.linear[0][0] = 3;
  query.output_to_reference.linear[0][1] = 2;
  query.output_to_reference.linear[1][0] = 2;
  query.output_to_reference.linear[1][1] = 3;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  EXPECT(damacy_spatial_resolution_info(resolved)->level == 0);
  damacy_spatial_resolution_destroy(resolved);
  query = identity_query();
  query.output_to_reference.linear[0][0] = 4;
  query.output_to_reference.linear[1][1] = 4;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  info = damacy_spatial_resolution_info(resolved);
  EXPECT(info->level == 2 && info->requires_resampling);
  EXPECT(info->output_to_source.offset[0] == 0.375);
  damacy_spatial_resolution_destroy(resolved);
  query = identity_query();
  query.output_to_reference.linear[0][0] = 0.5;
  query.output_to_reference.linear[1][1] = 0.5;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  EXPECT(damacy_spatial_resolution_info(resolved)->level == 0);
  damacy_spatial_resolution_destroy(resolved);
  damacy_ngff_image_destroy(image);
  return 0;
}

static int
test_sampler_bounds(void)
{
  struct damacy_ngff_image* image;
  EXPECT(!image_create(&image));
  struct damacy_spatial_query query = identity_query();
  query.output_to_reference.offset[0] = -0.25;
  struct damacy_spatial_resolution* resolved;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  const struct damacy_spatial_info* info =
    damacy_spatial_resolution_info(resolved);
  EXPECT(info->source_bounds_index.dims[0].beg == 0);
  EXPECT(info->source_bounds_index.dims[0].end == 4);
  damacy_spatial_resolution_destroy(resolved);
  query.sampler.filter = DAMACY_FILTER_LINEAR;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_INVAL);
  EXPECT(!resolved);
  query.sampler.boundary = DAMACY_BOUNDARY_CONSTANT;
  query.sampler.constant_value = 17;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  info = damacy_spatial_resolution_info(resolved);
  EXPECT(info->source_bounds_index.dims[0].beg == -1);
  EXPECT(info->source_bounds_index.dims[0].end == 4);
  EXPECT(info->read_bounds_index.dims[0].beg == 0);
  EXPECT(info->read_bounds_index.dims[0].end == 4);
  damacy_spatial_resolution_destroy(resolved);
  query.output_to_reference.offset[0] = -20;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  info = damacy_spatial_resolution_info(resolved);
  EXPECT(info->read_bounds_index.dims[0].beg == 0);
  EXPECT(info->read_bounds_index.dims[0].end == 0);
  damacy_spatial_resolution_destroy(resolved);
  query.sampler.boundary = DAMACY_BOUNDARY_CLAMP;
  query.sampler.constant_value = 0;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_OK);
  info = damacy_spatial_resolution_info(resolved);
  EXPECT(info->read_bounds_index.dims[0].beg == 0);
  EXPECT(info->read_bounds_index.dims[0].end == 1);
  damacy_spatial_resolution_destroy(resolved);
  damacy_ngff_image_destroy(image);
  return 0;
}

static int
test_invalid_queries(void)
{
  struct damacy_ngff_image* image;
  EXPECT(!image_create(&image));
  struct damacy_spatial_query query = identity_query();
  struct damacy_spatial_resolution* resolved;
  double invalid[] = { 0, NAN, INFINITY, 1e100 };
  for (size_t i = 0; i < sizeof(invalid) / sizeof(*invalid); ++i) {
    query.output_to_reference.linear[0][0] = invalid[i];
    EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
           DAMACY_INVAL);
    EXPECT(!resolved);
  }
  query = identity_query();
  query.level = -2;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_INVAL);
  query.level = 3;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_INVAL);
  query.level = 0;
  query.sampler.constant_value = 1;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_INVAL);
  query.sampler.constant_value = 0;
  query.sampler.filter = 0;
  EXPECT(damacy_spatial_resolve(image, &output, &query, &resolved) ==
         DAMACY_INVAL);
  query = identity_query();
  struct damacy_batch_spec wrong = output;
  wrong.sample_rank = 3;
  EXPECT(damacy_spatial_resolve(image, &wrong, &query, &resolved) ==
         DAMACY_RANK);
  damacy_ngff_image_destroy(image);
  return 0;
}

static int
test_metadata_limits_and_load(void)
{
  struct damacy_ngff_image* image = NULL;
  EXPECT(ngff_parse_group(text_slice(group), "v", 0, 2, &image) ==
         DAMACY_BUDGET);
  EXPECT(!image);
  EXPECT(ngff_parse_group(text_slice(group), "v", 1, 3, &image) ==
         DAMACY_INVAL);
  char root[4096];
  const char* scratch = getenv("TMPDIR");
  snprintf(
    root, sizeof(root), "%s/damacy-spatial-XXXXXX", scratch ? scratch : "/tmp");
  EXPECT(mkdtemp(root));
  char path[8192], array[1024];
  snprintf(path, sizeof(path), "%s/zarr.json", root);
  EXPECT(!fixture_write_file(path, group));
  size_t metadata_bytes = strlen(group);
  for (int i = 0; i < 3; ++i) {
    snprintf(path, sizeof(path), "%s/%d", root, i);
    EXPECT(!mkdir(path, 0700));
    snprintf(path, sizeof(path), "%s/%d/zarr.json", root, i);
    array_json(array, sizeof(array), 64 >> i);
    EXPECT(!fixture_write_file(path, array));
    metadata_bytes += strlen(array);
  }
  struct damacy_metadata_reader* reader;
  EXPECT(damacy_file_metadata_reader_create(2, NULL, &reader) == DAMACY_OK);
  struct damacy_ngff_limits limits = { .max_levels = 3,
                                       .max_metadata_bytes = metadata_bytes };
  EXPECT(damacy_ngff_image_load(reader, root, 0, &limits, &image) == DAMACY_OK);
  EXPECT(damacy_ngff_image_info(image)->levels[2].shape[0] == 16);
  damacy_ngff_image_destroy(image);
  limits.max_metadata_bytes--;
  EXPECT(damacy_ngff_image_load(reader, root, 0, &limits, &image) ==
         DAMACY_BUDGET);
  EXPECT(!image);
  limits.max_metadata_bytes = 1;
  EXPECT(damacy_ngff_image_load(reader, root, 0, &limits, &image) ==
         DAMACY_BUDGET);
  EXPECT(!image);
  limits.max_metadata_bytes = metadata_bytes;
  EXPECT(damacy_ngff_image_load(
           reader, "/damacy-missing-image", 0, &limits, &image) ==
         DAMACY_NOTFOUND);
  damacy_metadata_reader_destroy(reader);
  for (int i = 0; i < 3; ++i) {
    snprintf(path, sizeof(path), "%s/%d/zarr.json", root, i);
    EXPECT(!unlink(path));
    snprintf(path, sizeof(path), "%s/%d", root, i);
    EXPECT(!rmdir(path));
  }
  snprintf(path, sizeof(path), "%s/zarr.json", root);
  EXPECT(!unlink(path));
  EXPECT(!rmdir(root));
  return 0;
}

static int
test_collapsed_volume(void)
{
  struct damacy_ngff_level level = { .uri = "volume/0",
                                     .shape = { 8, 8, 8 },
                                     .scale_to_reference = { 1, 1, 1 } };
  struct damacy_ngff_image image = {
    .info = { .rank = 3,
              .data_type = "uint16",
              .axes = { { .name = "z", .kind = DAMACY_NGFF_SPACE },
                        { .name = "y", .kind = DAMACY_NGFF_SPACE },
                        { .name = "x", .kind = DAMACY_NGFF_SPACE } },
              .level_count = 1,
              .levels = &level },
    .levels = &level,
    .dtype = dtype_u16
  };
  struct damacy_batch_spec shape = { .dtype = DAMACY_F32,
                                     .sample_rank = 3,
                                     .sample_shape = { 2, 2, 2 },
                                     .samples_per_batch = 1 };
  struct damacy_spatial_query query = {
    .output_to_reference = { .linear = { { 0.7854589457317591,
                                           0.4315455905112043,
                                           0.5705067473246884 },
                                         { 0.2148316327537341,
                                           0.7753842829888732,
                                           -1.494658594462818 },
                                         { 1.0002905784854932,
                                           1.2069298735000775,
                                           -0.9241518471381295 } },
                             .offset = { 4, 4, 4 } },
    .sampler = { .filter = DAMACY_FILTER_NEAREST,
                 .boundary = DAMACY_BOUNDARY_CONSTANT },
    .level = DAMACY_LEVEL_AUTO
  };
  struct damacy_spatial_resolution* resolved = NULL;
  EXPECT(damacy_spatial_resolve(&image, &shape, &query, &resolved) ==
         DAMACY_INVAL);
  EXPECT(!resolved);
  query.output_to_reference = (struct damacy_affine){
    .linear = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } }
  };
  EXPECT(damacy_spatial_resolve(&image, &shape, &query, &resolved) ==
         DAMACY_OK);
  EXPECT(!damacy_spatial_resolution_info(resolved)->requires_resampling);
  damacy_spatial_resolution_destroy(resolved);
  return 0;
}

int
main(void)
{
  RUN(test_corner_conversion_and_copy);
  RUN(test_scale_rotation_and_shear);
  RUN(test_sampler_bounds);
  RUN(test_invalid_queries);
  RUN(test_collapsed_volume);
  RUN(test_metadata_limits_and_load);
  return 0;
}
