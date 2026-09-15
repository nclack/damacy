#pragma once

#include "damacy_pipeline.h"

#ifdef __cplusplus
extern "C"
{
#endif

  struct damacy_ngff_image;
  struct damacy_spatial_resolution;

  enum damacy_ngff_axis_kind
  {
    DAMACY_NGFF_SPACE = 1,
    DAMACY_NGFF_TIME,
    DAMACY_NGFF_CHANNEL,
  };

  struct damacy_ngff_axis
  {
    const char* name;
    const char* unit;
    enum damacy_ngff_axis_kind kind;
  };

  struct damacy_ngff_level
  {
    const char* uri;
    int64_t shape[DAMACY_MAX_RANK];
    double scale_to_reference[DAMACY_MAX_RANK];
    double origin_reference_index[DAMACY_MAX_RANK];
  };

  struct damacy_ngff_info
  {
    uint8_t rank;
    const char* data_type;
    struct damacy_ngff_axis axes[DAMACY_MAX_RANK];
    uint32_t level_count;
    const struct damacy_ngff_level* levels;
  };

  struct damacy_ngff_limits
  {
    uint32_t max_levels;
    uint64_t max_metadata_bytes;
  };

  struct damacy_affine
  {
    double linear[DAMACY_MAX_RANK][DAMACY_MAX_RANK];
    double offset[DAMACY_MAX_RANK];
  };

  enum damacy_filter
  {
    DAMACY_FILTER_NEAREST = 1,
    DAMACY_FILTER_LINEAR,
  };

  enum damacy_boundary
  {
    DAMACY_BOUNDARY_ERROR = 1,
    DAMACY_BOUNDARY_CONSTANT,
    DAMACY_BOUNDARY_CLAMP,
  };

  struct damacy_sampler
  {
    enum damacy_filter filter;
    enum damacy_boundary boundary;
    double constant_value;
  };

  enum
  {
    DAMACY_LEVEL_AUTO = -1
  };

  struct damacy_spatial_query
  {
    struct damacy_affine output_to_reference;
    struct damacy_sampler sampler;
    int32_t level;
  };

  struct damacy_spatial_info
  {
    const char* uri;
    uint32_t level;
    uint8_t rank;
    int64_t output_shape[DAMACY_MAX_RANK];
    int64_t source_shape[DAMACY_MAX_RANK];
    struct damacy_affine output_to_source;
    struct damacy_sampler sampler;
    struct damacy_aabb source_bounds_index;
    struct damacy_aabb read_bounds_index;
    int requires_resampling;
  };

  enum damacy_status damacy_ngff_image_load(
    struct damacy_metadata_reader* reader,
    const char* uri,
    uint32_t multiscale_index,
    const struct damacy_ngff_limits* limits,
    struct damacy_ngff_image** out);
  const struct damacy_ngff_info* damacy_ngff_image_info(
    const struct damacy_ngff_image* image);
  void damacy_ngff_image_destroy(struct damacy_ngff_image* image);

  enum damacy_status damacy_spatial_resolve(
    const struct damacy_ngff_image* image,
    const struct damacy_batch_spec* output,
    const struct damacy_spatial_query* query,
    struct damacy_spatial_resolution** out);
  const struct damacy_spatial_info* damacy_spatial_resolution_info(
    const struct damacy_spatial_resolution* resolution);
  enum damacy_status damacy_spatial_resolution_sample(
    const struct damacy_spatial_resolution* resolution,
    struct damacy_sample* out);
  void damacy_spatial_resolution_destroy(
    struct damacy_spatial_resolution* resolution);

#ifdef __cplusplus
}
#endif
