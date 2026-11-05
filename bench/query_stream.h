#pragma once

#include <stddef.h>
#include <stdint.h>

struct query_record
{
  uint32_t array;
  uint32_t origin[3];
};

struct query_stream
{
  struct query_record* records;
  uint64_t* samples_per_source;
  uint64_t count;
  uint64_t source_bytes;
};

int
query_stream_load(const char* path,
                  uint32_t arrays,
                  const int64_t* shapes,
                  size_t shape_stride,
                  const uint8_t* bytes_per_pixel,
                  const int64_t* sample_shape,
                  uint64_t expected_count,
                  struct query_stream* out);
void
query_stream_free(struct query_stream* stream);
