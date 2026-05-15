#include "zarr/zarr_metadata.h"

#include "util/json.h"
#include "util/prelude.h"

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
  static const struct json_query iter_parts[] = { { .kind = QUERY_ITER } };
  struct json_iter it;
  if (json_iter_init(arr.s, iter_parts, countof(iter_parts), &it, NULL))
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

static int
resolve_blosc_cname(struct json_node codec_obj, struct codec_config* out)
{
  static const struct json_query cname_path[] = {
    { QUERY_KEY, .key = "configuration" },
    { QUERY_KEY, .key = "cname" },
  };
  struct json_node cname;
  if (json_resolve(
        codec_obj.s, cname_path, countof(cname_path), &cname, NULL) ||
      cname.type != JSON_STRING)
    return 1;
  if (json_str_eq(cname, "lz4") || json_str_eq(cname, "lz4hc")) {
    out->id = CODEC_BLOSC_LZ4;
    return 0;
  }
  if (json_str_eq(cname, "zstd")) {
    out->id = CODEC_BLOSC_ZSTD;
    return 0;
  }
  return 1;
}

// Find the inner codec inside the sharding_indexed codec's "codecs"
// array. Skips "bytes" entries. Returns 0 on success.
static int
parse_inner_codec(struct json_node codecs_arr, struct codec_config* out)
{
  *out = (struct codec_config){ 0 };
  out->id = CODEC_NONE;
  if (codecs_arr.type != JSON_ARRAY)
    return 1;

  static const struct json_query iter_parts[] = { { .kind = QUERY_ITER } };
  static const struct json_query name_path[] = { { QUERY_KEY, .key = "name" } };

  struct json_iter it;
  if (json_iter_init(codecs_arr.s, iter_parts, countof(iter_parts), &it, NULL))
    return 1;

  for (;;) {
    struct json_node c;
    enum json_err e = json_iter_next(&it, &c);
    if (e == JSON_ERR_NOT_FOUND)
      break;
    if (e != JSON_OK || c.type != JSON_OBJECT)
      return 1;
    struct json_node name;
    if (json_resolve(c.s, name_path, countof(name_path), &name, NULL) ||
        name.type != JSON_STRING)
      return 1;
    if (json_str_eq(name, "bytes"))
      continue; // metadata-only; not a transformation we model
    if (json_str_eq(name, "zstd")) {
      out->id = CODEC_ZSTD;
      return 0;
    }
    if (json_str_eq(name, "blosc")) {
      return resolve_blosc_cname(c, out);
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
    static const struct json_query path[] = { { QUERY_KEY,
                                                .key = "node_type" } };
    struct json_node n;
    if (json_resolve(all, path, countof(path), &n, NULL) ||
        !json_str_eq(n, "array"))
      return 1;
  }

  // zarr_format must be 3
  {
    static const struct json_query path[] = { { QUERY_KEY,
                                                .key = "zarr_format" } };
    struct json_node n;
    if (json_resolve(all, path, countof(path), &n, NULL))
      return 1;
    int64_t v = 0;
    if (json_as_int(n, &v) || v != 3)
      return 1;
  }

  // shape
  {
    static const struct json_query path[] = { { QUERY_KEY, .key = "shape" } };
    struct json_node n;
    if (json_resolve(all, path, countof(path), &n, NULL))
      return 1;
    if (read_uint_array(n, out->shape, &out->rank, DAMACY_MAX_RANK))
      return 1;
  }

  // data_type
  {
    static const struct json_query path[] = { { QUERY_KEY,
                                                .key = "data_type" } };
    struct json_node n;
    if (json_resolve(all, path, countof(path), &n, NULL))
      return 1;
    if (parse_data_type(n, &out->dtype))
      return 1;
  }

  // chunk_grid -> outer (shard) shape
  {
    static const struct json_query path[] = {
      { QUERY_KEY, .key = "chunk_grid" },
      { QUERY_KEY, .key = "configuration" },
      { QUERY_KEY, .key = "chunk_shape" }
    };
    struct json_node n;
    if (json_resolve(all, path, countof(path), &n, NULL))
      return 1;
    uint8_t rank2 = 0;
    if (read_uint_array(n, out->shard_shape, &rank2, DAMACY_MAX_RANK))
      return 1;
    if (rank2 != out->rank)
      return 1;
  }

  // codecs[] | select(.name == "sharding_indexed") — find the sharding codec
  static const struct json_query sharding_name_path[] = { { QUERY_KEY,
                                                            .key = "name" } };
  static const struct json_query sharding_path[] = {
    { QUERY_KEY, .key = "codecs" },
    { .kind = QUERY_ITER },
    { QUERY_WHERE,
      .where = { .part = sharding_name_path,
                 .n = countof(sharding_name_path),
                 .rhs = "sharding_indexed",
                 .rhs_type = JSON_STRING } }
  };

  struct json_node sharding;
  enum json_err sharding_err =
    json_resolve(all, sharding_path, countof(sharding_path), &sharding, NULL);
  if (sharding_err == JSON_OK) {
    out->sharded = 1;

    // inner chunk shape
    {
      static const struct json_query path[] = {
        { QUERY_KEY, .key = "configuration" },
        { QUERY_KEY, .key = "chunk_shape" }
      };
      struct json_node n;
      if (json_resolve(sharding.s, path, countof(path), &n, NULL))
        return 1;
      uint8_t rank2 = 0;
      if (read_uint_array(n, out->inner_chunk_shape, &rank2, DAMACY_MAX_RANK))
        return 1;
      if (rank2 != out->rank)
        return 1;
    }

    // inner codecs
    {
      static const struct json_query path[] = {
        { QUERY_KEY, .key = "configuration" }, { QUERY_KEY, .key = "codecs" }
      };
      struct json_node n;
      if (json_resolve(sharding.s, path, countof(path), &n, NULL))
        return 1;
      if (parse_inner_codec(n, &out->inner_codec))
        return 1;
    }

    // index_location (optional; defaults to "end")
    out->index_location_end = 1;
    {
      static const struct json_query path[] = {
        { QUERY_KEY, .key = "configuration" },
        { QUERY_KEY, .key = "index_location" }
      };
      struct json_node n;
      if (json_resolve(sharding.s, path, countof(path), &n, NULL) == JSON_OK &&
          json_str_eq(n, "start"))
        out->index_location_end = 0;
    }
  } else if (sharding_err == JSON_ERR_NOT_FOUND) {
    // Non-sharded zarr v3: each chunk is its own file. Treat as
    // "shard of one" — the chunk_grid's chunk_shape IS the inner chunk
    // shape, and the outer (shard) shape equals the inner. The
    // top-level codecs list applies to each chunk directly.
    out->sharded = 0;
    out->index_location_end = 1;
    for (uint8_t i = 0; i < out->rank; ++i)
      out->inner_chunk_shape[i] = out->shard_shape[i];

    static const struct json_query codecs_path[] = { { QUERY_KEY,
                                                       .key = "codecs" } };
    struct json_node codecs;
    if (json_resolve(all, codecs_path, countof(codecs_path), &codecs, NULL))
      return 1;
    if (parse_inner_codec(codecs, &out->inner_codec))
      return 1;
  } else {
    return 1;
  }

  // Sanity: shard_shape must be a multiple of inner_chunk_shape on each dim.
  for (uint8_t i = 0; i < out->rank; ++i) {
    if (out->inner_chunk_shape[i] == 0 ||
        out->shard_shape[i] % out->inner_chunk_shape[i] != 0)
      return 1;
  }

  return 0;
}

int
zarr_metadata_inner_per_shard(const struct zarr_metadata* meta,
                              uint64_t* out_per_dim,
                              uint64_t* out_total)
{
  if (!meta)
    return 1;
  uint64_t total = 1;
  for (uint8_t d = 0; d < meta->rank; ++d) {
    if (meta->inner_chunk_shape[d] == 0 ||
        meta->shard_shape[d] % meta->inner_chunk_shape[d] != 0)
      return 1;
    uint64_t per = meta->shard_shape[d] / meta->inner_chunk_shape[d];
    if (per != 0 && total > UINT64_MAX / per)
      return 1; // overflow
    total *= per;
    if (out_per_dim)
      out_per_dim[d] = per;
  }
  if (out_total)
    *out_total = total;
  return 0;
}
