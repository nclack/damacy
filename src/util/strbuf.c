#include "util/strbuf.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
is_inline(const struct strbuf* sb)
{
  return sb->beg == sb->inline_buf;
}

static void
activate_inline(struct strbuf* sb)
{
  sb->beg = sb->inline_buf;
  sb->end = sb->inline_buf;
  sb->cap_end = sb->inline_buf + STRBUF_INLINE_CAP;
  sb->inline_buf[0] = 0;
}

void
strbuf_free(struct strbuf* sb)
{
  if (!sb)
    return;
  if (sb->beg && !is_inline(sb))
    free(sb->beg);
  sb->beg = NULL;
  sb->end = NULL;
  sb->cap_end = NULL;
}

void
strbuf_reset(struct strbuf* sb)
{
  if (!sb || !sb->beg)
    return;
  sb->end = sb->beg;
  *sb->end = 0;
}

int
strbuf_reserve(struct strbuf* sb, size_t need)
{
  if (!sb)
    return 1;
  if (!sb->beg)
    activate_inline(sb);

  size_t len = (size_t)(sb->end - sb->beg);
  size_t cap = (size_t)(sb->cap_end - sb->beg);
  if (need > SIZE_MAX - 1 - len)
    return 1;
  size_t want = len + need + 1;
  if (want <= cap)
    return 0;

  size_t new_cap = cap ? cap : STRBUF_INLINE_CAP;
  while (new_cap < want) {
    size_t doubled = new_cap * 2;
    if (doubled < new_cap)
      return 1;
    new_cap = doubled;
  }

  char* new_buf;
  if (is_inline(sb)) {
    new_buf = (char*)malloc(new_cap);
    if (!new_buf)
      return 1;
    memcpy(new_buf, sb->beg, len + 1);
  } else {
    new_buf = (char*)realloc(sb->beg, new_cap);
    if (!new_buf)
      return 1;
  }
  sb->beg = new_buf;
  sb->end = new_buf + len;
  sb->cap_end = new_buf + new_cap;
  return 0;
}

int
strbuf_append(struct strbuf* sb, const char* data, size_t n)
{
  if (strbuf_reserve(sb, n))
    return 1;
  memcpy(sb->end, data, n);
  sb->end += n;
  *sb->end = 0;
  return 0;
}

int
strbuf_append_cstr(struct strbuf* sb, const char* s)
{
  return strbuf_append(sb, s, strlen(s));
}

int
strbuf_vappendf(struct strbuf* sb, const char* fmt, va_list ap)
{
  if (!sb)
    return 1;
  if (!sb->beg)
    activate_inline(sb);

  va_list ap2;
  va_copy(ap2, ap);
  size_t avail = (size_t)(sb->cap_end - sb->end);
  int n = vsnprintf(sb->end, avail, fmt, ap);
  if (n < 0) {
    va_end(ap2);
    return 1;
  }
  if ((size_t)n < avail) {
    sb->end += (size_t)n;
    va_end(ap2);
    return 0;
  }

  if (strbuf_reserve(sb, (size_t)n)) {
    va_end(ap2);
    return 1;
  }
  avail = (size_t)(sb->cap_end - sb->end);
  int n2 = vsnprintf(sb->end, avail, fmt, ap2);
  va_end(ap2);
  if (n2 < 0 || (size_t)n2 >= avail)
    return 1;
  sb->end += (size_t)n2;
  return 0;
}

int
strbuf_appendf(struct strbuf* sb, const char* fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  int rc = strbuf_vappendf(sb, fmt, ap);
  va_end(ap);
  return rc;
}

int
strbuf_set(struct strbuf* sb, const char* s)
{
  strbuf_reset(sb);
  return strbuf_append_cstr(sb, s);
}

int
strbuf_join_path(struct strbuf* sb, const char* prefix, const char* suffix)
{
  strbuf_reset(sb);
  if (prefix && prefix[0]) {
    if (strbuf_append_cstr(sb, prefix))
      return 1;
    size_t len = strbuf_len(sb);
    if (len > 0 && strbuf_cstr(sb)[len - 1] != '/') {
      if (strbuf_append(sb, "/", 1))
        return 1;
    }
  }
  return strbuf_append_cstr(sb, suffix);
}

const char*
strbuf_cstr(const struct strbuf* sb)
{
  return (sb && sb->beg) ? sb->beg : "";
}

size_t
strbuf_len(const struct strbuf* sb)
{
  return sb->beg ? (size_t)(sb->end - sb->beg) : 0;
}
