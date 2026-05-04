#pragma once

#include "util/strbuf.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // JSON writer backed by a caller-owned strbuf. Appends to whatever is
  // already in the buffer; use strbuf_reset beforehand if you want to
  // start fresh. The buffer auto-grows, so there is no truncation mode.
  struct json_writer
  {
    struct strbuf* sb;
    int needs_comma;
    int error;
  };

  void jw_init(struct json_writer* jw, struct strbuf* sb);
  void jw_object_begin(struct json_writer* jw);
  void jw_object_end(struct json_writer* jw);
  void jw_array_begin(struct json_writer* jw);
  void jw_array_end(struct json_writer* jw);
  void jw_key(struct json_writer* jw, const char* key);
  void jw_string(struct json_writer* jw, const char* val);
  void jw_int(struct json_writer* jw, int64_t val);
  void jw_uint(struct json_writer* jw, uint64_t val);
  void jw_float(struct json_writer* jw, double val);
  void jw_null(struct json_writer* jw);
  void jw_bool(struct json_writer* jw, int val);
  // Emit a pre-validated JSON value as-is. Participates in comma gating.
  void jw_raw(struct json_writer* jw, const char* text);
  size_t jw_length(const struct json_writer* jw);
  int jw_error(const struct json_writer* jw);

#ifdef __cplusplus
}
#endif
