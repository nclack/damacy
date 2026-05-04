#include "util/json_writer.h"

#include <stdarg.h>

void
jw_init(struct json_writer* jw, struct strbuf* sb)
{
  jw->sb = sb;
  jw->needs_comma = 0;
  jw->error = 0;
}

static void
jw_put(struct json_writer* jw, const char* fmt, ...)
{
  if (jw->error)
    return;
  va_list ap;
  va_start(ap, fmt);
  int rc = strbuf_vappendf(jw->sb, fmt, ap);
  va_end(ap);
  if (rc)
    jw->error = 1;
}

static void
jw_comma(struct json_writer* jw)
{
  if (jw->needs_comma)
    jw_put(jw, ",");
}

void
jw_object_begin(struct json_writer* jw)
{
  jw_comma(jw);
  jw_put(jw, "{");
  jw->needs_comma = 0;
}

void
jw_object_end(struct json_writer* jw)
{
  jw_put(jw, "}");
  jw->needs_comma = 1;
}

void
jw_array_begin(struct json_writer* jw)
{
  jw_comma(jw);
  jw_put(jw, "[");
  jw->needs_comma = 0;
}

void
jw_array_end(struct json_writer* jw)
{
  jw_put(jw, "]");
  jw->needs_comma = 1;
}

void
jw_key(struct json_writer* jw, const char* key)
{
  jw_comma(jw);
  jw_put(jw, "\"");
  // Keys are assumed to be safe identifiers; no escaping needed
  jw_put(jw, "%s", key);
  jw_put(jw, "\":");
  jw->needs_comma = 0;
}

static void
jw_put_escaped_string(struct json_writer* jw, const char* s)
{
  jw_put(jw, "\"");
  for (; *s && !jw->error; ++s) {
    unsigned char c = (unsigned char)*s;
    switch (c) {
      case '"':
        jw_put(jw, "\\\"");
        break;
      case '\\':
        jw_put(jw, "\\\\");
        break;
      case '\n':
        jw_put(jw, "\\n");
        break;
      case '\r':
        jw_put(jw, "\\r");
        break;
      case '\t':
        jw_put(jw, "\\t");
        break;
      default:
        if (c < 0x20)
          jw_put(jw, "\\u%04x", c);
        else
          jw_put(jw, "%c", c);
        break;
    }
  }
  jw_put(jw, "\"");
}

void
jw_string(struct json_writer* jw, const char* val)
{
  jw_comma(jw);
  jw_put_escaped_string(jw, val);
  jw->needs_comma = 1;
}

void
jw_int(struct json_writer* jw, int64_t val)
{
  jw_comma(jw);
  jw_put(jw, "%lld", (long long)val);
  jw->needs_comma = 1;
}

void
jw_uint(struct json_writer* jw, uint64_t val)
{
  jw_comma(jw);
  jw_put(jw, "%llu", (unsigned long long)val);
  jw->needs_comma = 1;
}

void
jw_float(struct json_writer* jw, double val)
{
  jw_comma(jw);
  // Always include a decimal point so JSON parsers see a float, not an int.
  if (val == (long long)val && val >= -1e15 && val <= 1e15)
    jw_put(jw, "%.1f", val);
  else
    jw_put(jw, "%g", val);
  jw->needs_comma = 1;
}

void
jw_null(struct json_writer* jw)
{
  jw_comma(jw);
  jw_put(jw, "null");
  jw->needs_comma = 1;
}

void
jw_bool(struct json_writer* jw, int val)
{
  jw_comma(jw);
  jw_put(jw, val ? "true" : "false");
  jw->needs_comma = 1;
}

void
jw_raw(struct json_writer* jw, const char* text)
{
  jw_comma(jw);
  jw_put(jw, "%s", text);
  jw->needs_comma = 1;
}

size_t
jw_length(const struct json_writer* jw)
{
  return strbuf_len(jw->sb);
}

int
jw_error(const struct json_writer* jw)
{
  return jw->error;
}
