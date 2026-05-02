// Dynamic string buffer with small-string inline storage.
//
// Use for path/key construction and other variable-length text where a
// fixed-size stack buffer would risk silent truncation. Short content
// (<= STRBUF_INLINE_CAP bytes) stays in the inline buffer — zero heap
// allocations. Growth spills to heap, then reuses the allocation across
// subsequent reset/append cycles.
//
// Zero-initialization is valid: `struct strbuf sb = {0}` is safe to pass
// to any function; the inline buffer lazy-activates on first append.
#pragma once

#include <stdarg.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define STRBUF_INLINE_CAP 128

  struct strbuf
  {
    char inline_buf[STRBUF_INLINE_CAP];
    char* beg;
    char* end;
    char* cap_end;
  };

  void strbuf_free(struct strbuf* sb);
  void strbuf_reset(struct strbuf* sb);
  int strbuf_reserve(struct strbuf* sb, size_t need);
  int strbuf_append(struct strbuf* sb, const char* data, size_t n);
  int strbuf_append_cstr(struct strbuf* sb, const char* s);
  int strbuf_appendf(struct strbuf* sb, const char* fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;
  int strbuf_vappendf(struct strbuf* sb, const char* fmt, va_list ap);
  int strbuf_set(struct strbuf* sb, const char* s);
  size_t strbuf_len(const struct strbuf* sb);
  const char* strbuf_cstr(const struct strbuf* sb);

#ifdef __cplusplus
}
#endif
