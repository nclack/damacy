#include "query_stream.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t
read_le(const unsigned char* bytes, unsigned count)
{
  uint64_t value = 0;
  for (unsigned i = 0; i < count; ++i)
    value |= (uint64_t)bytes[i] << (8 * i);
  return value;
}

void
query_stream_free(struct query_stream* stream)
{
  free(stream->records);
  free(stream->samples_per_source);
  *stream = (struct query_stream){ 0 };
}

int
query_stream_load(const char* path,
                  uint32_t arrays,
                  const int64_t* shapes,
                  size_t shape_stride,
                  const uint8_t* bytes_per_pixel,
                  const int64_t* sample_shape,
                  uint64_t expected_count,
                  struct query_stream* out)
{
  if (!arrays || !expected_count || expected_count > 10000000 ||
      expected_count > SIZE_MAX / sizeof(struct query_record))
    return 1;
  FILE* file = fopen(path, "rb");
  if (!file)
    return 1;
  int error = 1;
  unsigned char header[16];
  if (fread(header, 1, sizeof header, file) != sizeof header ||
      memcmp(header, "DAMACYQ1", 8) || read_le(header + 8, 8) != expected_count)
    goto Done;
  out->records = calloc((size_t)expected_count, sizeof(*out->records));
  out->samples_per_source = calloc(arrays, sizeof(*out->samples_per_source));
  if (!out->records || !out->samples_per_source)
    goto Done;
  uint64_t pixels = 1;
  for (unsigned axis = 0; axis < 3; ++axis) {
    if (sample_shape[axis] <= 0 ||
        pixels > UINT64_MAX / (uint64_t)sample_shape[axis])
      goto Done;
    pixels *= (uint64_t)sample_shape[axis];
  }
  for (uint64_t i = 0; i < expected_count; ++i) {
    unsigned char bytes[16];
    if (fread(bytes, 1, sizeof bytes, file) != sizeof bytes)
      goto Done;
    struct query_record* query = &out->records[i];
    query->array = (uint32_t)read_le(bytes, 4);
    if (query->array >= arrays || !bytes_per_pixel[query->array])
      goto Done;
    for (unsigned axis = 0; axis < 3; ++axis) {
      query->origin[axis] = (uint32_t)read_le(bytes + 4 + 4 * axis, 4);
      int64_t size = shapes[(size_t)query->array * shape_stride + axis];
      if (size < sample_shape[axis] ||
          query->origin[axis] > (uint64_t)(size - sample_shape[axis]))
        goto Done;
    }
    uint64_t itemsize = bytes_per_pixel[query->array];
    if (pixels > (UINT64_MAX - out->source_bytes) / itemsize)
      goto Done;
    out->source_bytes += pixels * itemsize;
    ++out->samples_per_source[query->array];
  }
  if (fgetc(file) != EOF || ferror(file))
    goto Done;
  out->count = expected_count;
  error = 0;
Done:
  fclose(file);
  if (error)
    query_stream_free(out);
  return error;
}
