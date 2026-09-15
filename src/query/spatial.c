#include "damacy_spatial.h"

#include "damacy_config.h"
#include "ngff/ngff.h"
#include "pipeline/components.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

struct damacy_spatial_resolution
{
  struct damacy_spatial_info info;
  struct damacy_sample sample;
};

static uint8_t
spatial_matrix(const struct damacy_ngff_image* image,
               const struct damacy_affine* transform,
               uint32_t level,
               double matrix[3][3])
{
  uint8_t axes[3], rank = 0;
  for (uint8_t d = 0; d < image->info.rank; ++d)
    if (image->info.axes[d].kind == DAMACY_NGFF_SPACE)
      axes[rank++] = d;
  for (uint8_t i = 0; i < rank; ++i)
    for (uint8_t j = 0; j < rank; ++j)
      matrix[i][j] = transform->linear[axes[i]][axes[j]] /
                     image->levels[level].scale_to_reference[axes[i]];
  return rank;
}

static int
linear_is_invertible(const struct damacy_ngff_image* image,
                     const struct damacy_affine* transform)
{
  double matrix[3][3];
  uint8_t rank = spatial_matrix(image, transform, 0, matrix);
  for (uint8_t i = 0; i < rank; ++i) {
    double scale = 0;
    for (uint8_t j = 0; j < rank; ++j)
      scale = fmax(scale, fabs(matrix[i][j]));
    if (!scale)
      return 0;
    for (uint8_t j = 0; j < rank; ++j)
      matrix[i][j] /= scale;
  }
  for (uint8_t j = 0; j < rank; ++j) {
    uint8_t pivot_row = j, pivot_column = j;
    for (uint8_t i = j; i < rank; ++i)
      for (uint8_t k = j; k < rank; ++k)
        if (fabs(matrix[i][k]) > fabs(matrix[pivot_row][pivot_column])) {
          pivot_row = i;
          pivot_column = k;
        }
    if (fabs(matrix[pivot_row][pivot_column]) <= 64 * DBL_EPSILON)
      return 0;
    for (uint8_t k = j; k < rank; ++k) {
      double value = matrix[j][k];
      matrix[j][k] = matrix[pivot_row][k];
      matrix[pivot_row][k] = value;
    }
    for (uint8_t i = j; i < rank; ++i) {
      double value = matrix[i][j];
      matrix[i][j] = matrix[i][pivot_column];
      matrix[i][pivot_column] = value;
    }
    for (uint8_t i = j + 1; i < rank; ++i) {
      double factor = matrix[i][j] / matrix[j][j];
      for (uint8_t k = j + 1; k < rank; ++k)
        matrix[i][k] -= factor * matrix[j][k];
    }
  }
  return 1;
}

static double
minimum_spacing(const struct damacy_ngff_image* image,
                const struct damacy_affine* transform,
                uint32_t level)
{
  double matrix[3][3];
  uint8_t rank = spatial_matrix(image, transform, level, matrix);
  double scale = 0;
  for (uint8_t i = 0; i < rank; ++i)
    for (uint8_t j = 0; j < rank; ++j)
      scale = fmax(scale, fabs(matrix[i][j]));
  if (!scale)
    return 0;
  for (uint8_t i = 0; i < rank; ++i)
    for (uint8_t j = 0; j < rank; ++j)
      matrix[i][j] /= scale;
  for (int sweep = 0; sweep < 32; ++sweep) {
    int changed = 0;
    for (uint8_t p = 0; p < rank; ++p) {
      for (uint8_t q = p + 1; q < rank; ++q) {
        double a = 0, b = 0, cross = 0;
        for (uint8_t i = 0; i < rank; ++i) {
          a += matrix[i][p] * matrix[i][p];
          b += matrix[i][q] * matrix[i][q];
          cross += matrix[i][p] * matrix[i][q];
        }
        if (!cross || fabs(cross) <= DBL_EPSILON * sqrt(a) * sqrt(b))
          continue;
        double tau = (b - a) / (2 * cross);
        double t = copysign(1, tau) / (fabs(tau) + hypot(1, tau));
        double c = 1 / hypot(1, t), s = t * c;
        for (uint8_t i = 0; i < rank; ++i) {
          double first = matrix[i][p], second = matrix[i][q];
          matrix[i][p] = c * first - s * second;
          matrix[i][q] = s * first + c * second;
        }
        changed = 1;
      }
    }
    if (!changed)
      break;
  }
  double minimum = INFINITY;
  for (uint8_t j = 0; j < rank; ++j) {
    double length = 0;
    for (uint8_t i = 0; i < rank; ++i)
      length = hypot(length, matrix[i][j]);
    minimum = fmin(minimum, length);
  }
  return scale * minimum;
}

static enum damacy_status
validate_query(const struct damacy_ngff_image* image,
               const struct damacy_batch_spec* output,
               const struct damacy_spatial_query* query)
{
  if (!image || !output || !query)
    return DAMACY_INVAL;
  if (output->sample_rank != image->info.rank)
    return DAMACY_RANK;
  int64_t shape[DAMACY_MAX_RANK + 1], strides[DAMACY_MAX_RANK + 1];
  uint64_t bytes;
  enum damacy_status status = batch_spec_layout(output, shape, strides, &bytes);
  if (status != DAMACY_OK)
    return status;
  if (!cast_path_supported(output->dtype, image->dtype))
    return DAMACY_DTYPE;
  if (query->level < DAMACY_LEVEL_AUTO ||
      (query->level >= 0 && (uint32_t)query->level >= image->info.level_count))
    return DAMACY_INVAL;
  const struct damacy_sampler* sampler = &query->sampler;
  if ((sampler->filter != DAMACY_FILTER_NEAREST &&
       sampler->filter != DAMACY_FILTER_LINEAR) ||
      (sampler->boundary != DAMACY_BOUNDARY_ERROR &&
       sampler->boundary != DAMACY_BOUNDARY_CONSTANT &&
       sampler->boundary != DAMACY_BOUNDARY_CLAMP) ||
      !isfinite(sampler->constant_value) ||
      (sampler->boundary != DAMACY_BOUNDARY_CONSTANT &&
       sampler->constant_value))
    return DAMACY_INVAL;
  for (uint8_t i = 0; i < image->info.rank; ++i) {
    double offset = query->output_to_reference.offset[i];
    int space = image->info.axes[i].kind == DAMACY_NGFF_SPACE;
    if (!isfinite(offset) ||
        output->sample_shape[i] > INT64_C(4503599627370495))
      return DAMACY_INVAL;
    if (!space && offset != floor(offset))
      return DAMACY_INVAL;
    for (uint8_t j = 0; j < image->info.rank; ++j) {
      double value = query->output_to_reference.linear[i][j];
      if (!isfinite(value))
        return DAMACY_INVAL;
      if ((!space || image->info.axes[j].kind != DAMACY_NGFF_SPACE) &&
          value != (double)(i == j))
        return DAMACY_INVAL;
    }
  }
  if (!linear_is_invertible(image, &query->output_to_reference))
    return DAMACY_INVAL;
  return DAMACY_OK;
}

static int64_t
clamp_index(int64_t value, int64_t end)
{
  return value < 0 ? 0 : value > end ? end : value;
}

static enum damacy_status
resolve_bounds(const struct damacy_ngff_image* image,
               struct damacy_spatial_info* info)
{
  info->source_bounds_index.rank = info->read_bounds_index.rank = info->rank;
  for (uint8_t i = 0; i < info->rank; ++i) {
    long double beg = info->output_to_source.offset[i], end = beg;
    for (uint8_t j = 0; j < info->rank; ++j) {
      double value = info->output_to_source.linear[i][j];
      long double first = 0.5L * value;
      long double last = ((long double)info->output_shape[j] - 0.5L) * value;
      beg += fminl(first, last);
      end += fmaxl(first, last);
      if (value != (double)(i == j))
        info->requires_resampling = 1;
    }
    double offset = info->output_to_source.offset[i];
    if (offset != floor(offset))
      info->requires_resampling = 1;
    if (!isfinite(beg) || !isfinite(end) || beg < -4503599627370495.0L ||
        end > 4503599627370495.0L)
      return DAMACY_INVAL;
    struct damacy_interval span;
    if (info->sampler.filter == DAMACY_FILTER_NEAREST) {
      span.beg = (int64_t)floorl(beg);
      span.end = (int64_t)floorl(end) + 1;
    } else {
      span.beg = (int64_t)floorl(beg - 0.5L);
      span.end = (int64_t)ceill(end - 0.5L) + 1;
    }
    info->source_bounds_index.dims[i] = span;
    int64_t size = info->source_shape[i];
    if (span.beg < 0 || span.end > size) {
      if (info->sampler.boundary == DAMACY_BOUNDARY_ERROR ||
          image->info.axes[i].kind != DAMACY_NGFF_SPACE)
        return DAMACY_INVAL;
      info->requires_resampling = 1;
    }
    struct damacy_interval read;
    if (info->sampler.boundary == DAMACY_BOUNDARY_CLAMP) {
      read.beg = clamp_index(span.beg, size - 1);
      read.end = clamp_index(span.end - 1, size - 1) + 1;
    } else {
      read.beg = clamp_index(span.beg, size);
      read.end = clamp_index(span.end, size);
    }
    info->read_bounds_index.dims[i] = read;
  }
  return DAMACY_OK;
}

enum damacy_status
damacy_spatial_resolve(const struct damacy_ngff_image* image,
                       const struct damacy_batch_spec* output,
                       const struct damacy_spatial_query* query,
                       struct damacy_spatial_resolution** out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = NULL;
  enum damacy_status status = validate_query(image, output, query);
  if (status != DAMACY_OK)
    return status;
  uint32_t level_index = query->level < 0 ? 0 : (uint32_t)query->level;
  if (query->level == DAMACY_LEVEL_AUTO)
    for (uint32_t i = 1; i < image->info.level_count; ++i)
      if (minimum_spacing(image, &query->output_to_reference, i) >=
          1 - 64 * DBL_EPSILON)
        level_index = i;
  const struct damacy_ngff_level* level = &image->levels[level_index];
  struct damacy_spatial_resolution* resolution = calloc(1, sizeof(*resolution));
  if (!resolution)
    return DAMACY_OOM;
  struct damacy_spatial_info* info = &resolution->info;
  info->level = level_index;
  info->rank = image->info.rank;
  info->sampler = query->sampler;
  for (uint8_t i = 0; i < info->rank; ++i) {
    info->output_shape[i] = output->sample_shape[i];
    info->source_shape[i] = level->shape[i];
    double scale = level->scale_to_reference[i];
    info->output_to_source.offset[i] = (query->output_to_reference.offset[i] -
                                        level->origin_reference_index[i]) /
                                       scale;
    for (uint8_t j = 0; j < info->rank; ++j)
      info->output_to_source.linear[i][j] =
        query->output_to_reference.linear[i][j] / scale;
  }
  status = resolve_bounds(image, info);
  if (status != DAMACY_OK) {
    free(resolution);
    return status;
  }
  char* uri = malloc(strlen(level->uri) + 1);
  if (!uri) {
    free(resolution);
    return DAMACY_OOM;
  }
  strcpy(uri, level->uri);
  info->uri = uri;
  if (!info->requires_resampling) {
    resolution->sample.uri = uri;
    resolution->sample.rank = info->rank;
    for (uint8_t i = 0; i < info->rank; ++i)
      resolution->sample.axes[i] =
        (struct damacy_axis_selection){ .kind = DAMACY_AXIS_INTERVAL,
                                        .interval =
                                          info->source_bounds_index.dims[i] };
  }
  *out = resolution;
  return DAMACY_OK;
}

const struct damacy_spatial_info*
damacy_spatial_resolution_info(
  const struct damacy_spatial_resolution* resolution)
{
  return resolution ? &resolution->info : NULL;
}

enum damacy_status
damacy_spatial_resolution_sample(
  const struct damacy_spatial_resolution* resolution,
  struct damacy_sample* out)
{
  if (!out)
    return DAMACY_INVAL;
  *out = (struct damacy_sample){ 0 };
  if (!resolution)
    return DAMACY_INVAL;
  if (resolution->info.requires_resampling)
    return DAMACY_UNSUPPORTED;
  *out = resolution->sample;
  return DAMACY_OK;
}

void
damacy_spatial_resolution_destroy(struct damacy_spatial_resolution* resolution)
{
  if (resolution) {
    free((void*)resolution->info.uri);
    free(resolution);
  }
}
