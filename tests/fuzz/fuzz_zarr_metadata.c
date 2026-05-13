#include "damacy_limits.h"
#include "zarr/zarr_metadata.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define countof(arr) (sizeof(arr) / sizeof((arr)[0]))

struct tape
{
  const uint8_t* p;
  const uint8_t* end;
};

static uint8_t
tape_u8(struct tape* t)
{
  if (t->p >= t->end)
    return 0;
  return *t->p++;
}

// Each menu mixes valid and adjacent-invalid entries so the modulo
// selector lands on both legal and rejected branches.

static const char* const k_node_type[] = {
  "array", "group", "Array", "", "arr",
};

static const char* const k_zarr_format[] = {
  "3", "2", "0", "-1", "3.0", "true",
};

static const char* const k_dtype[] = {
  "uint8",   "uint16",    "uint32", "uint64",  "int8",
  "int16",   "int32",     "int64",  "float16", "float32",
  "float64", "complex64", "bool",   "",        "uint128",
};

static const char* const k_shape[] = {
  "[1024,1024,128]",
  "[64]",
  "[1,2,3,4,5,6,7,8,9,10,11,12,13]",
  "[]",
  "[0]",
  "[0,0,0]",
  "[18446744073709551616]",
  "[-1]",
  "[1.5]",
  "[\"x\"]",
  "{}",
  "null",
};

static const char* const k_outer_shape[] = {
  "[1024,1024,128]", "[64]", "[1,1,1]", "[]", "[0]", "[3,3,3]",
};

static const char* const k_inner_shape[] = {
  "[64,64,16]", "[1,1,1]", "[64]",  "[]",         "[0]",
  "[0,0,0]",    "[3,5,7]", "[3,3]", "[64,64,17]",
};

// Adjacent string literals on separate lines trip
// clang-tidy bugprone-suspicious-missing-comma; keep each on one line.
// clang-format off
static const char* const k_inner_codecs[] = {
  "[{\"name\":\"bytes\"}]",
  "[{\"name\":\"bytes\"},{\"name\":\"zstd\",\"configuration\":{\"level\":3}}]",
  // lz4 / lz4hc intentionally exercise the planner's rejection path:
  // zarr_metadata.c still parses these names and assigns CODEC_BLOSC_LZ4,
  // which the planner then rejects. Keep them in the corpus.
  "[{\"name\":\"bytes\"},{\"name\":\"blosc\",\"configuration\":{\"cname\":\"lz4\"}}]",
  "[{\"name\":\"bytes\"},{\"name\":\"blosc\",\"configuration\":{\"cname\":\"zstd\"}}]",
  "[{\"name\":\"blosc\",\"configuration\":{\"cname\":\"lz4hc\"}}]",
  "[{\"name\":\"snappy\"}]",
  "[{\"name\":\"blosc\",\"configuration\":{\"cname\":\"brotli\"}}]",
  "[{\"name\":\"blosc\"}]",
  "[{\"id\":42}]",
  "[{\"name\":7}]",
  "{\"name\":\"zstd\"}",
  "[]",
};
// clang-format on

static const char* const k_index_location[] = {
  "\"end\"", "\"start\"", "\"middle\"", "null", "0",
};

static const char* const k_extra_outer[] = {
  "",
  ",{\"name\":\"crc32c\"}",
  ",{\"name\":\"transpose\",\"configuration\":{\"order\":[2,1,0]}}",
};

static const char*
pick(struct tape* t, const char* const* menu, size_t n)
{
  return menu[tape_u8(t) % n];
}

static size_t
append(char* dst, size_t off, size_t cap, const char* s)
{
  size_t n = strlen(s);
  if (off == SIZE_MAX || off + n >= cap)
    return SIZE_MAX;
  memcpy(dst + off, s, n);
  return off + n;
}

int
LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
  struct tape t = { .p = data, .end = data + size };
  const char* node_type = pick(&t, k_node_type, countof(k_node_type));
  const char* zarr_fmt = pick(&t, k_zarr_format, countof(k_zarr_format));
  const char* dt = pick(&t, k_dtype, countof(k_dtype));
  const char* shape = pick(&t, k_shape, countof(k_shape));
  const char* outer = pick(&t, k_outer_shape, countof(k_outer_shape));
  const char* inner = pick(&t, k_inner_shape, countof(k_inner_shape));
  const char* codecs = pick(&t, k_inner_codecs, countof(k_inner_codecs));
  const char* idxloc = pick(&t, k_index_location, countof(k_index_location));
  const char* extra = pick(&t, k_extra_outer, countof(k_extra_outer));
  int emit_idxloc = (tape_u8(&t) & 1) != 0;

  enum
  {
    DOC_CAP = 8192
  };
  static char doc[DOC_CAP];
  size_t off = 0;

  off = append(doc, off, DOC_CAP, "{\"node_type\":\"");
  off = append(doc, off, DOC_CAP, node_type);
  off = append(doc, off, DOC_CAP, "\",\"zarr_format\":");
  off = append(doc, off, DOC_CAP, zarr_fmt);
  off = append(doc, off, DOC_CAP, ",\"shape\":");
  off = append(doc, off, DOC_CAP, shape);
  off = append(doc, off, DOC_CAP, ",\"data_type\":\"");
  off = append(doc, off, DOC_CAP, dt);
  off = append(doc,
               off,
               DOC_CAP,
               "\",\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{"
               "\"chunk_shape\":");
  off = append(doc, off, DOC_CAP, outer);
  off = append(doc,
               off,
               DOC_CAP,
               "}},\"codecs\":[{\"name\":\"sharding_indexed\","
               "\"configuration\":{\"chunk_shape\":");
  off = append(doc, off, DOC_CAP, inner);
  off = append(doc, off, DOC_CAP, ",\"codecs\":");
  off = append(doc, off, DOC_CAP, codecs);
  if (emit_idxloc) {
    off = append(doc, off, DOC_CAP, ",\"index_location\":");
    off = append(doc, off, DOC_CAP, idxloc);
  }
  off = append(doc, off, DOC_CAP, "}}");
  off = append(doc, off, DOC_CAP, extra);
  off = append(doc, off, DOC_CAP, "]}");

  if (off == SIZE_MAX)
    return 0;

  struct zarr_metadata meta;
  if (zarr_metadata_parse(doc, off, &meta) == 0) {
    uint64_t per_dim[DAMACY_MAX_RANK];
    uint64_t total;
    (void)zarr_metadata_inner_per_shard(&meta, per_dim, &total);
    (void)zarr_metadata_inner_per_shard(&meta, NULL, &total);
    (void)zarr_metadata_inner_per_shard(&meta, per_dim, NULL);
  }

  if (size > 0) {
    struct zarr_metadata raw;
    (void)zarr_metadata_parse((const char*)data, size, &raw);
  }

  return 0;
}
