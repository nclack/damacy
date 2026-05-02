#include "zarr_metadata.h"

#include "util/json.h"

#include <string.h>

// Read a JSON array of unsigned integers into out[0..max_rank). Sets
// *out_rank to the number of elements. Returns 0 on success.
static int
read_uint_array(struct json_node arr,
                uint64_t* out,
                uint8_t* out_rank,
                uint8_t max_rank)
{
  if (arr.type != JSON_ARRAY)
    return 1;
  static const struct json_seg iter_segs[] = {
    { .kind = SEG_ITER },
    { .kind = SEG_END },
  };
  static const struct json_pred iter_pred = { .segs = iter_segs };
  struct json_iter it;
  if (json_iter_init(arr.s, &iter_pred, &it, NULL))
    return 1;
  size_t n = 0;
  for (;;) {
    struct json_node v;
    enum json_err e = json_iter_next(&it, &v);
    if (e == JSON_ERR_NOT_FOUND)
      break;
    if (e != JSON_OK || n >= max_rank)
      return 1;
    if (json_as_uint(v, &out[n]))
      return 1;
    n++;
  }
  if (n == 0)
    return 1;
  *out_rank = (uint8_t)n;
  return 0;
}

static int
parse_data_type(struct json_node n, enum dtype* out)
{
  if (n.type != JSON_STRING)
    return 1;
  if (n.flag == JSON_NODE_FLAG_STR_ESCAPED)
    return 1; // dtype names are pure ASCII
  return dtype_from_zarr_string(n.s.beg, cslice_len(n.s), out);
}

// Find the inner zstd codec inside the sharding_indexed codec's "codecs"
// array. Skips "bytes" entries. Returns 0 on success.
static int
parse_inner_codec(struct json_node codecs_arr, struct codec_config* out)
{
  *out = (struct codec_config){ 0 };
  out->id = CODEC_NONE;
  if (codecs_arr.type != JSON_ARRAY)
    return 1;

  static const struct json_seg iter_segs[] = {
    { .kind = SEG_ITER },
    { .kind = SEG_END },
  };
  static const struct json_pred iter_pred = { .segs = iter_segs };
  static const struct json_seg name_path[] = {
    { SEG_KEY, .key = "name" },
    { .kind = SEG_END },
  };
  static const struct json_pred name_pred = { .segs = name_path };

  struct json_iter it;
  if (json_iter_init(codecs_arr.s, &iter_pred, &it, NULL))
    return 1;

  for (;;) {
    struct json_node c;
    enum json_err e = json_iter_next(&it, &c);
    if (e == JSON_ERR_NOT_FOUND)
      break;
    if (e != JSON_OK || c.type != JSON_OBJECT)
      return 1;
    struct json_node name;
    if (json_resolve(c.s, &name_pred, &name, NULL) || name.type != JSON_STRING)
      return 1;
    if (json_str_eq(name, "bytes"))
      continue; // metadata-only; not a transformation we model
    if (json_str_eq(name, "zstd")) {
      out->id = CODEC_ZSTD;
      static const struct json_seg level_path[] = {
        { SEG_KEY, .key = "configuration" },
        { SEG_KEY, .key = "level" },
        { .kind = SEG_END },
      };
      static const struct json_pred level_pred = { .segs = level_path };
      struct json_node level;
      if (json_resolve(c.s, &level_pred, &level, NULL) == JSON_OK) {
        int64_t lv = 0;
        if (json_as_int(level, &lv) == 0 && lv >= 0 && lv < 256)
          out->level = (uint8_t)lv;
      }
      return 0;
    }
    return 1; // unknown / unsupported codec for v0
  }
  // No inner codec means raw bytes.
  out->id = CODEC_NONE;
  return 0;
}

int
zarr_metadata_parse(const char* src, size_t src_len, struct zarr_metadata* out)
{
  if (!src || !out)
    return 1;
  memset(out, 0, sizeof(*out));

  struct cslice all = { .beg = src, .end = src + src_len };

  // node_type must be "array"
  {
    static const struct json_seg path[] = {
      { SEG_KEY, .key = "node_type" },
      { .kind = SEG_END },
    };
    static const struct json_pred pred = { .segs = path };
    struct json_node n;
    if (json_resolve(all, &pred, &n, NULL) || !json_str_eq(n, "array"))
      return 1;
  }

  // zarr_format must be 3
  {
    static const struct json_seg path[] = {
      { SEG_KEY, .key = "zarr_format" },
      { .kind = SEG_END },
    };
    static const struct json_pred pred = { .segs = path };
    struct json_node n;
    if (json_resolve(all, &pred, &n, NULL))
      return 1;
    int64_t v = 0;
    if (json_as_int(n, &v) || v != 3)
      return 1;
  }

  // shape
  {
    static const struct json_seg path[] = {
      { SEG_KEY, .key = "shape" },
      { .kind = SEG_END },
    };
    static const struct json_pred pred = { .segs = path };
    struct json_node n;
    if (json_resolve(all, &pred, &n, NULL))
      return 1;
    if (read_uint_array(n, out->shape, &out->rank, DAMACY_MAX_RANK))
      return 1;
  }

  // data_type
  {
    static const struct json_seg path[] = {
      { SEG_KEY, .key = "data_type" },
      { .kind = SEG_END },
    };
    static const struct json_pred pred = { .segs = path };
    struct json_node n;
    if (json_resolve(all, &pred, &n, NULL))
      return 1;
    if (parse_data_type(n, &out->dtype))
      return 1;
  }

  // chunk_grid -> outer (shard) shape
  {
    static const struct json_seg path[] = {
      { SEG_KEY, .key = "chunk_grid" },
      { SEG_KEY, .key = "configuration" },
      { SEG_KEY, .key = "chunk_shape" },
      { .kind = SEG_END },
    };
    static const struct json_pred pred = { .segs = path };
    struct json_node n;
    if (json_resolve(all, &pred, &n, NULL))
      return 1;
    uint8_t rank2 = 0;
    if (read_uint_array(n, out->shard_shape, &rank2, DAMACY_MAX_RANK))
      return 1;
    if (rank2 != out->rank)
      return 1;
  }

  // codecs[] | select(.name == "sharding_indexed") — find the sharding codec
  static const struct json_seg sharding_name_path[] = {
    { SEG_KEY, .key = "name" },
    { .kind = SEG_END },
  };
  static const struct json_seg sharding_path[] = {
    { SEG_KEY, .key = "codecs" },
    { .kind = SEG_ITER },
    { SEG_WHERE,
      .where = { .path = sharding_name_path,
                 .rhs = "sharding_indexed",
                 .rhs_type = JSON_STRING } },
    { .kind = SEG_END },
  };
  static const struct json_pred sharding_pred = { .segs = sharding_path };

  struct json_node sharding;
  if (json_resolve(all, &sharding_pred, &sharding, NULL))
    return 1; // v0 requires sharded layout
  out->sharded = 1;

  // inner chunk shape
  {
    static const struct json_seg path[] = {
      { SEG_KEY, .key = "configuration" },
      { SEG_KEY, .key = "chunk_shape" },
      { .kind = SEG_END },
    };
    static const struct json_pred pred = { .segs = path };
    struct json_node n;
    if (json_resolve(sharding.s, &pred, &n, NULL))
      return 1;
    uint8_t rank2 = 0;
    if (read_uint_array(n, out->inner_chunk_shape, &rank2, DAMACY_MAX_RANK))
      return 1;
    if (rank2 != out->rank)
      return 1;
  }

  // inner codecs
  {
    static const struct json_seg path[] = {
      { SEG_KEY, .key = "configuration" },
      { SEG_KEY, .key = "codecs" },
      { .kind = SEG_END },
    };
    static const struct json_pred pred = { .segs = path };
    struct json_node n;
    if (json_resolve(sharding.s, &pred, &n, NULL))
      return 1;
    if (parse_inner_codec(n, &out->inner_codec))
      return 1;
  }

  // index_location (optional; defaults to "end")
  out->index_location_end = 1;
  {
    static const struct json_seg path[] = {
      { SEG_KEY, .key = "configuration" },
      { SEG_KEY, .key = "index_location" },
      { .kind = SEG_END },
    };
    static const struct json_pred pred = { .segs = path };
    struct json_node n;
    if (json_resolve(sharding.s, &pred, &n, NULL) == JSON_OK &&
        json_str_eq(n, "start"))
      out->index_location_end = 0;
  }

  // Sanity: shard_shape must be a multiple of inner_chunk_shape on each dim.
  for (uint8_t i = 0; i < out->rank; ++i) {
    if (out->inner_chunk_shape[i] == 0 ||
        out->shard_shape[i] % out->inner_chunk_shape[i] != 0)
      return 1;
  }

  return 0;
}
