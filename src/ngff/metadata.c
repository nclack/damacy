#include "ngff/ngff.h"

#include "zarr/zarr_metadata.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static enum json_err
elements(struct json_node node, struct json_iter* it)
{
  static const struct json_query query = { .kind = QUERY_ITER };
  if (node.type != JSON_ARRAY)
    return JSON_ERR_TYPE;
  return json_iter_init(node.s, &query, 1, it, NULL);
}

static int
hex4(const char* p, uint32_t* out)
{
  *out = 0;
  for (int i = 0; i < 4; ++i) {
    unsigned char c = (unsigned char)p[i];
    uint32_t digit;
    if (c >= '0' && c <= '9')
      digit = c - '0';
    else if (c >= 'a' && c <= 'f')
      digit = c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      digit = c - 'A' + 10;
    else
      return 1;
    *out = (*out << 4) | digit;
  }
  return 0;
}

static enum damacy_status
read_string(struct json_node node, const char** out)
{
  if (node.type != JSON_STRING || !cslice_len(node.s))
    return DAMACY_INVAL;
  char* text = malloc(cslice_len(node.s) + 1);
  if (!text)
    return DAMACY_OOM;
  char* dst = text;
  const char* p = node.s.beg;
  while (p < node.s.end) {
    unsigned char c = (unsigned char)*p++;
    if (c != '\\') {
      *dst++ = (char)c;
      continue;
    }
    if (p == node.s.end)
      goto Invalid;
    c = (unsigned char)*p++;
    if (c == 'u') {
      uint32_t code;
      if (node.s.end - p < 4 || hex4(p, &code))
        goto Invalid;
      p += 4;
      if (code >= 0xd800 && code <= 0xdbff) {
        uint32_t low;
        if (node.s.end - p < 6 || p[0] != '\\' || p[1] != 'u' ||
            hex4(p + 2, &low) || low < 0xdc00 || low > 0xdfff)
          goto Invalid;
        p += 6;
        code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
      } else if (code >= 0xdc00 && code <= 0xdfff) {
        goto Invalid;
      }
      if (!code)
        goto Invalid;
      if (code < 0x80) {
        *dst++ = (char)code;
      } else if (code < 0x800) {
        *dst++ = (char)(0xc0 | (code >> 6));
        *dst++ = (char)(0x80 | (code & 0x3f));
      } else if (code < 0x10000) {
        *dst++ = (char)(0xe0 | (code >> 12));
        *dst++ = (char)(0x80 | ((code >> 6) & 0x3f));
        *dst++ = (char)(0x80 | (code & 0x3f));
      } else {
        *dst++ = (char)(0xf0 | (code >> 18));
        *dst++ = (char)(0x80 | ((code >> 12) & 0x3f));
        *dst++ = (char)(0x80 | ((code >> 6) & 0x3f));
        *dst++ = (char)(0x80 | (code & 0x3f));
      }
    } else {
      const char* escapes = "\"\\/bfnrt";
      const char* values = "\"\\/\b\f\n\r\t";
      const char* found = strchr(escapes, c);
      if (!found)
        goto Invalid;
      *dst++ = values[found - escapes];
    }
  }
  *dst = 0;
  *out = text;
  return DAMACY_OK;
Invalid:
  free(text);
  return DAMACY_INVAL;
}

static enum json_err
member(struct json_node node, const char* key, struct json_node* out)
{
  struct json_object_iter it;
  enum json_err error = json_object_iter_init(node, &it);
  if (error != JSON_OK)
    return error;
  struct json_node name, value;
  int found = 0;
  while ((error = json_object_iter_next(&it, &name, &value)) == JSON_OK) {
    if (!cslice_len(name.s))
      continue;
    const char* text = NULL;
    enum damacy_status status = read_string(name, &text);
    if (status != DAMACY_OK)
      return status == DAMACY_OOM ? JSON_ERR_OOM : JSON_ERR_PARSE;
    int matches = !strcmp(key, text);
    free((void*)text);
    if (matches) {
      if (found)
        return JSON_ERR_PARSE;
      *out = value;
      found = 1;
    }
  }
  return error == JSON_ERR_NOT_FOUND && found ? JSON_OK : error;
}

static int
string_equal(struct json_node node, const char* expected)
{
  const char* text = NULL;
  if (read_string(node, &text) != DAMACY_OK)
    return 0;
  int result = !strcmp(text, expected);
  free((void*)text);
  return result;
}

static int
root_object(struct cslice src, struct json_node* root)
{
  return json_resolve(src, NULL, 0, root, NULL) || root->type != JSON_OBJECT;
}

static enum damacy_status
read_axes(struct json_node node, struct damacy_ngff_info* info)
{
  struct json_iter it;
  if (elements(node, &it))
    return DAMACY_INVAL;
  uint8_t spaces = 0;
  int time_seen = 0, channel_seen = 0;
  struct json_node axis;
  enum json_err error;
  while ((error = json_iter_next(&it, &axis)) == JSON_OK) {
    if (info->rank == 5)
      return DAMACY_RANK;
    struct damacy_ngff_axis* out = &info->axes[info->rank++];
    struct json_node value;
    if (member(axis, "name", &value))
      return DAMACY_INVAL;
    enum damacy_status status = read_string(value, &out->name);
    if (status != DAMACY_OK)
      return status;
    for (uint8_t i = 0; i + 1 < info->rank; ++i)
      if (!strcmp(out->name, info->axes[i].name))
        return DAMACY_INVAL;
    if (member(axis, "type", &value))
      return DAMACY_UNSUPPORTED;
    if (string_equal(value, "space")) {
      out->kind = DAMACY_NGFF_SPACE;
      spaces++;
    } else if (string_equal(value, "time")) {
      if (info->rank != 1 || time_seen++)
        return DAMACY_INVAL;
      out->kind = DAMACY_NGFF_TIME;
    } else if (string_equal(value, "channel")) {
      if (spaces || channel_seen++)
        return DAMACY_INVAL;
      out->kind = DAMACY_NGFF_CHANNEL;
    } else {
      return DAMACY_UNSUPPORTED;
    }
    error = member(axis, "unit", &value);
    if (error == JSON_OK) {
      status = read_string(value, &out->unit);
      if (status != DAMACY_OK)
        return status;
    } else if (error != JSON_ERR_NOT_FOUND) {
      return DAMACY_INVAL;
    }
  }
  if (error != JSON_ERR_NOT_FOUND || spaces < 2 || spaces > 3)
    return DAMACY_INVAL;
  return DAMACY_OK;
}

static enum damacy_status
read_vector(struct json_node node, uint8_t rank, double* values)
{
  struct json_iter it;
  if (elements(node, &it))
    return DAMACY_INVAL;
  struct json_node value;
  for (uint8_t d = 0; d < rank; ++d)
    if (json_iter_next(&it, &value) || json_as_double(value, &values[d]) ||
        !isfinite(values[d]))
      return DAMACY_INVAL;
  return json_iter_next(&it, &value) == JSON_ERR_NOT_FOUND ? DAMACY_OK
                                                           : DAMACY_INVAL;
}

static enum damacy_status
read_transform(struct json_node node,
               uint8_t rank,
               double* scale,
               double* translation)
{
  struct json_iter it;
  if (elements(node, &it))
    return DAMACY_INVAL;
  int count = 0;
  enum json_err error;
  struct json_node transform;
  while ((error = json_iter_next(&it, &transform)) == JSON_OK) {
    struct json_node type, value;
    if (member(transform, "type", &type))
      return DAMACY_INVAL;
    const char* key = count == 0 ? "scale" : "translation";
    if (count > 1 || !string_equal(type, key))
      return DAMACY_UNSUPPORTED;
    if (member(transform, "path", &value) != JSON_ERR_NOT_FOUND)
      return DAMACY_UNSUPPORTED;
    if (member(transform, key, &value))
      return DAMACY_INVAL;
    enum damacy_status status =
      read_vector(value, rank, count ? translation : scale);
    if (status != DAMACY_OK)
      return status;
    ++count;
  }
  if (error != JSON_ERR_NOT_FOUND || !count)
    return DAMACY_INVAL;
  for (uint8_t d = 0; d < rank; ++d)
    if (scale[d] <= 0)
      return DAMACY_INVAL;
  return DAMACY_OK;
}

static int
relative_path(const char* path)
{
  if (!*path || *path == '/' || strchr(path, '\\') || strchr(path, ':'))
    return 0;
  const char* p = path;
  do {
    const char* end = strchr(p, '/');
    size_t len = end ? (size_t)(end - p) : strlen(p);
    if (!len || (len == 1 && p[0] == '.') ||
        (len == 2 && p[0] == '.' && p[1] == '.'))
      return 0;
    p = end ? end + 1 : NULL;
  } while (p);
  return 1;
}

static enum damacy_status
read_level(struct json_node node,
           const char* uri,
           uint8_t rank,
           struct damacy_ngff_level* level)
{
  struct json_node value;
  if (member(node, "path", &value))
    return DAMACY_INVAL;
  const char* path = NULL;
  enum damacy_status status = read_string(value, &path);
  if (status != DAMACY_OK)
    return status;
  if (!relative_path(path)) {
    free((void*)path);
    return DAMACY_INVAL;
  }
  size_t root_len = strlen(uri), path_len = strlen(path);
  if (root_len > SIZE_MAX - path_len - 2) {
    free((void*)path);
    return DAMACY_BUDGET;
  }
  char* joined = malloc(root_len + path_len + 2);
  if (!joined) {
    free((void*)path);
    return DAMACY_OOM;
  }
  memcpy(joined, uri, root_len);
  if (root_len && uri[root_len - 1] != '/')
    joined[root_len++] = '/';
  memcpy(joined + root_len, path, path_len + 1);
  free((void*)path);
  level->uri = joined;
  if (member(node, "coordinateTransformations", &value))
    return DAMACY_INVAL;
  return read_transform(
    value, rank, level->scale_to_reference, level->origin_reference_index);
}

static enum damacy_status
normalize_levels(struct damacy_ngff_image* image)
{
  struct damacy_ngff_level reference = image->levels[0];
  for (uint32_t i = image->info.level_count; i-- > 0;) {
    struct damacy_ngff_level* level = &image->levels[i];
    for (uint8_t d = 0; d < image->info.rank; ++d) {
      double scale = level->scale_to_reference[d];
      double origin = level->origin_reference_index[d];
      double base_scale = reference.scale_to_reference[d];
      double base_origin = reference.origin_reference_index[d];
      if (image->info.axes[d].kind != DAMACY_NGFF_SPACE) {
        if (scale != base_scale || origin != base_origin)
          return DAMACY_UNSUPPORTED;
        level->scale_to_reference[d] = 1;
        level->origin_reference_index[d] = 0;
      } else {
        if (i && scale < image->levels[i - 1].scale_to_reference[d])
          return DAMACY_INVAL;
        double ratio = scale / base_scale;
        double offset = (origin - base_origin) / base_scale + 0.5 * (1 - ratio);
        if (!isfinite(ratio) || !isfinite(offset))
          return DAMACY_INVAL;
        level->scale_to_reference[d] = ratio;
        level->origin_reference_index[d] = offset;
      }
    }
  }
  return DAMACY_OK;
}

enum damacy_status
ngff_parse_group(struct cslice src,
                 const char* uri,
                 uint32_t multiscale_index,
                 uint32_t max_levels,
                 struct damacy_ngff_image** out)
{
  *out = NULL;
  struct json_node root, node, ome, multiscale;
  uint64_t format;
  if (root_object(src, &root) || member(root, "zarr_format", &node) ||
      json_as_uint(node, &format) || format != 3 ||
      member(root, "node_type", &node) || !string_equal(node, "group") ||
      member(root, "attributes", &node) || member(node, "ome", &ome) ||
      member(ome, "version", &node))
    return DAMACY_INVAL;
  if (!string_equal(node, "0.5"))
    return DAMACY_UNSUPPORTED;
  if (member(ome, "multiscales", &node) || node.type != JSON_ARRAY)
    return DAMACY_INVAL;
  const struct json_query select = { .kind = QUERY_INDEX,
                                     .index = multiscale_index };
  if (json_resolve(node.s, &select, 1, &multiscale, NULL))
    return DAMACY_INVAL;
  struct damacy_ngff_image* image = calloc(1, sizeof(*image));
  if (!image)
    return DAMACY_OOM;
  enum damacy_status status = DAMACY_INVAL;
  if (member(multiscale, "axes", &node))
    goto Fail;
  status = read_axes(node, &image->info);
  if (status != DAMACY_OK)
    goto Fail;
  enum json_err error;
  status = DAMACY_INVAL;
  if (member(multiscale, "datasets", &node))
    goto Fail;
  struct json_iter it;
  if (elements(node, &it))
    goto Fail;
  struct json_node dataset;
  while ((error = json_iter_next(&it, &dataset)) == JSON_OK) {
    if (image->info.level_count >= max_levels) {
      status = DAMACY_BUDGET;
      goto Fail;
    }
    size_t count = (size_t)image->info.level_count + 1;
    if (count > SIZE_MAX / sizeof(*image->levels)) {
      status = DAMACY_BUDGET;
      goto Fail;
    }
    void* levels = realloc(image->levels, count * sizeof(*image->levels));
    if (!levels) {
      status = DAMACY_OOM;
      goto Fail;
    }
    image->levels = levels;
    struct damacy_ngff_level* level = &image->levels[image->info.level_count++];
    memset(level, 0, sizeof(*level));
    status = read_level(dataset, uri, image->info.rank, level);
    if (status != DAMACY_OK)
      goto Fail;
    for (uint32_t i = 0; i + 1 < image->info.level_count; ++i)
      if (!strcmp(level->uri, image->levels[i].uri)) {
        status = DAMACY_INVAL;
        goto Fail;
      }
  }
  if (error != JSON_ERR_NOT_FOUND || !image->info.level_count) {
    status = DAMACY_INVAL;
    goto Fail;
  }
  status = normalize_levels(image);
  if (status != DAMACY_OK)
    goto Fail;
  image->info.levels = image->levels;
  *out = image;
  return DAMACY_OK;
Fail:
  damacy_ngff_image_destroy(image);
  return status;
}

enum damacy_status
ngff_parse_array(struct cslice src,
                 struct damacy_ngff_image* image,
                 uint32_t index)
{
  struct json_node root, names, value;
  if (root_object(src, &root))
    return DAMACY_INVAL;
  static const char* keys[] = { "zarr_format",    "node_type",
                                "shape",          "data_type",
                                "chunk_grid",     "chunk_key_encoding",
                                "fill_value",     "codecs",
                                "dimension_names" };
  for (size_t i = 0; i < sizeof(keys) / sizeof(*keys); ++i)
    if (member(root, keys[i], &value))
      return DAMACY_INVAL;
  struct zarr_metadata array;
  if (zarr_metadata_parse(src.beg, cslice_len(src), &array))
    return DAMACY_UNSUPPORTED;
  if (array.rank != image->info.rank)
    return DAMACY_RANK;
  struct json_iter it;
  if (member(root, "dimension_names", &names) || elements(names, &it))
    return DAMACY_INVAL;
  for (uint8_t d = 0; d < array.rank; ++d) {
    const char* name = NULL;
    if (json_iter_next(&it, &value))
      return DAMACY_INVAL;
    enum damacy_status status = read_string(value, &name);
    if (status != DAMACY_OK)
      return status;
    int match = !strcmp(name, image->info.axes[d].name);
    free((void*)name);
    if (!match || !array.shape[d] || array.shape[d] > INT64_C(4503599627370495))
      return DAMACY_INVAL;
    if (index && image->info.axes[d].kind != DAMACY_NGFF_SPACE &&
        array.shape[d] != (uint64_t)image->levels[0].shape[d])
      return DAMACY_UNSUPPORTED;
    if (index && array.shape[d] > (uint64_t)image->levels[index - 1].shape[d])
      return DAMACY_INVAL;
    image->levels[index].shape[d] = (int64_t)array.shape[d];
  }
  if (json_iter_next(&it, &value) != JSON_ERR_NOT_FOUND)
    return DAMACY_INVAL;
  if (!index) {
    image->dtype = array.dtype;
    if (member(root, "data_type", &value))
      return DAMACY_INVAL;
    return read_string(value, &image->info.data_type);
  }
  return array.dtype == image->dtype ? DAMACY_OK : DAMACY_DTYPE;
}

const struct damacy_ngff_info*
damacy_ngff_image_info(const struct damacy_ngff_image* image)
{
  return image ? &image->info : NULL;
}

void
damacy_ngff_image_destroy(struct damacy_ngff_image* image)
{
  if (!image)
    return;
  for (uint8_t d = 0; d < image->info.rank; ++d) {
    free((void*)image->info.axes[d].name);
    free((void*)image->info.axes[d].unit);
  }
  for (uint32_t i = 0; i < image->info.level_count; ++i)
    free((void*)image->levels[i].uri);
  free((void*)image->info.data_type);
  free(image->levels);
  free(image);
}
