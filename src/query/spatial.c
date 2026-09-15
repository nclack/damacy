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

static double
minimum_spacing(const struct damacy_ngff_image* image,
                const struct damacy_affine* transform,
                uint32_t level)
{
  uint8_t axes[3], rank = 0;
  for (uint8_t d = 0; d < image->info.rank; ++d)
    if (image->info.axes[d].kind == DAMACY_NGFF_SPACE)
      axes[rank++] = d;
  double matrix[3][3] = { { 0 } }, scale = 0;
  for (uint8_t i = 0; i < rank; ++i)
    for (uint8_t j = 0; j < rank; ++j) {
      double value = transform->linear[axes[i]][axes[j]] /
                     image->levels[level].scale_to_reference[axes[i]];
      if (!isfinite(value))
        return NAN;
      matrix[i][j] = value;
      scale = fmax(scale, fabs(value));
    }
  if (!scale)
    return 0;
  double gram[3][3] = { { 0 } };
  for (uint8_t i = 0; i < rank; ++i)
    for (uint8_t j = 0; j < rank; ++j)
      for (uint8_t k = 0; k < rank; ++k)
        gram[i][j] += (matrix[i][k] / scale) * (matrix[j][k] / scale);
  for (int sweep = 0; sweep < 24; ++sweep) {
    for (uint8_t p = 0; p < rank; ++p) {
      for (uint8_t q = p + 1; q < rank; ++q) {
        double cross = gram[p][q];
        if (fabs(cross) <= DBL_EPSILON * sqrt(gram[p][p] * gram[q][q]))
          continue;
        double tau = (gram[q][q] - gram[p][p]) / (2 * cross);
        double t = copysign(1, tau) / (fabs(tau) + hypot(1, tau));
        double c = 1 / hypot(1, t), s = t * c;
        gram[p][p] -= t * cross;
        gram[q][q] += t * cross;
        gram[p][q] = gram[q][p] = 0;
        for (uint8_t k = 0; k < rank; ++k) {
          if (k == p || k == q)
            continue;
          double a = gram[k][p], b = gram[k][q];
          gram[k][p] = gram[p][k] = c * a - s * b;
          gram[k][q] = gram[q][k] = s * a + c * b;
        }
      }
    }
  }
  double minimum = gram[0][0];
  for (uint8_t d = 1; d < rank; ++d)
    minimum = fmin(minimum, gram[d][d]);
  return scale * sqrt(fmax(0, minimum));
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
  double spacing = minimum_spacing(image, &query->output_to_reference, 0);
  if (!isfinite(spacing) || spacing <= 0)
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
