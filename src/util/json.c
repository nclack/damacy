#include "util/json.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---- cslice helpers --------------------------------------------------------

static int
cs_at_end(struct cslice c)
{
  return c.beg >= c.end;
}

static void
skip_ws(struct cslice* c)
{
  while (!cs_at_end(*c)) {
    char ch = *c->beg;
    if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r')
      break;
    c->beg++;
  }
}

// ---- Lexer -----------------------------------------------------------------
//
// Each lex_* helper consumes the next value/key from c (no leading ws),
// fills the output, and advances c->beg past the consumed bytes.

static int
lex_string_span(struct cslice* c,
                struct cslice* span,
                enum json_node_flag* flag)
{
  if (cs_at_end(*c) || *c->beg != '"')
    return 1;
  c->beg++;
  const char* start = c->beg;
  enum json_node_flag f = JSON_NODE_FLAG_NONE;
  while (!cs_at_end(*c)) {
    char ch = *c->beg;
    if (ch == '"') {
      span->beg = start;
      span->end = c->beg;
      *flag = f;
      c->beg++;
      return 0;
    }
    if (ch == '\\') {
      f = JSON_NODE_FLAG_STR_ESCAPED;
      c->beg++;
      if (cs_at_end(*c))
        return 1;
      char esc = *c->beg;
      if (esc == 'u') {
        if ((size_t)(c->end - c->beg) < 5)
          return 1;
        for (int k = 1; k <= 4; ++k) {
          if (!isxdigit((unsigned char)c->beg[k]))
            return 1;
        }
        c->beg += 5;
      } else {
        switch (esc) {
          case '"':
          case '\\':
          case '/':
          case 'b':
          case 'f':
          case 'n':
          case 'r':
          case 't':
            c->beg++;
            break;
          default:
            return 1;
        }
      }
      continue;
    }
    if ((unsigned char)ch < 0x20)
      return 1;
    c->beg++;
  }
  return 1;
}

static int
lex_number_span(struct cslice* c,
                struct cslice* span,
                enum json_node_flag* flag)
{
  const char* start = c->beg;
  enum json_node_flag f = JSON_NODE_FLAG_NONE;
  if (!cs_at_end(*c) && *c->beg == '-')
    c->beg++;
  if (cs_at_end(*c))
    return 1;
  if (*c->beg == '0') {
    c->beg++;
  } else if (*c->beg >= '1' && *c->beg <= '9') {
    while (!cs_at_end(*c) && *c->beg >= '0' && *c->beg <= '9')
      c->beg++;
  } else {
    return 1;
  }
  if (!cs_at_end(*c) && *c->beg == '.') {
    f = JSON_NODE_FLAG_NUM_FLOAT;
    c->beg++;
    if (cs_at_end(*c) || !(*c->beg >= '0' && *c->beg <= '9'))
      return 1;
    while (!cs_at_end(*c) && *c->beg >= '0' && *c->beg <= '9')
      c->beg++;
  }
  if (!cs_at_end(*c) && (*c->beg == 'e' || *c->beg == 'E')) {
    f = JSON_NODE_FLAG_NUM_FLOAT;
    c->beg++;
    if (!cs_at_end(*c) && (*c->beg == '+' || *c->beg == '-'))
      c->beg++;
    if (cs_at_end(*c) || !(*c->beg >= '0' && *c->beg <= '9'))
      return 1;
    while (!cs_at_end(*c) && *c->beg >= '0' && *c->beg <= '9')
      c->beg++;
  }
  span->beg = start;
  span->end = c->beg;
  *flag = f;
  return 0;
}

static int
lex_keyword(struct cslice* c, const char* kw)
{
  size_t L = strlen(kw);
  if ((size_t)(c->end - c->beg) < L)
    return 1;
  if (memcmp(c->beg, kw, L) != 0)
    return 1;
  c->beg += L;
  return 0;
}

// Skips one container by tracking depth and respecting strings. On entry
// c->beg is at the opening '{' or '['; on success c->beg is just past
// the matching close.
static int
skip_container(struct cslice* c)
{
  if (cs_at_end(*c))
    return 1;
  char open = *c->beg;
  char close;
  if (open == '{')
    close = '}';
  else if (open == '[')
    close = ']';
  else
    return 1;
  int depth = 0;
  while (!cs_at_end(*c)) {
    char ch = *c->beg;
    if (ch == '"') {
      // Lex through the string.
      struct cslice tmp = *c;
      struct cslice span;
      enum json_node_flag f;
      if (lex_string_span(&tmp, &span, &f))
        return 1;
      *c = tmp;
      continue;
    }
    if (ch == open) {
      depth++;
      c->beg++;
      continue;
    }
    if (ch == close) {
      depth--;
      c->beg++;
      if (depth == 0)
        return 0;
      continue;
    }
    c->beg++;
  }
  return 1;
}

// Lex the next value at *c (no leading ws). Fills out and advances c.
static enum json_err
lex_value(struct cslice* c, struct json_node* out)
{
  if (cs_at_end(*c))
    return JSON_ERR_PARSE;
  out->flag = JSON_NODE_FLAG_NONE;
  char ch = *c->beg;
  if (ch == '{' || ch == '[') {
    const char* start = c->beg;
    if (skip_container(c))
      return JSON_ERR_PARSE;
    out->type = (ch == '{') ? JSON_OBJECT : JSON_ARRAY;
    out->s.beg = start;
    out->s.end = c->beg;
    return JSON_OK;
  }
  if (ch == '"') {
    if (lex_string_span(c, &out->s, &out->flag))
      return JSON_ERR_PARSE;
    out->type = JSON_STRING;
    return JSON_OK;
  }
  if (ch == 't') {
    const char* start = c->beg;
    if (lex_keyword(c, "true"))
      return JSON_ERR_PARSE;
    out->type = JSON_BOOL;
    out->s.beg = start;
    out->s.end = c->beg;
    return JSON_OK;
  }
  if (ch == 'f') {
    const char* start = c->beg;
    if (lex_keyword(c, "false"))
      return JSON_ERR_PARSE;
    out->type = JSON_BOOL;
    out->s.beg = start;
    out->s.end = c->beg;
    return JSON_OK;
  }
  if (ch == 'n') {
    const char* start = c->beg;
    if (lex_keyword(c, "null"))
      return JSON_ERR_PARSE;
    out->type = JSON_NULL;
    out->s.beg = start;
    out->s.end = c->beg;
    return JSON_OK;
  }
  if (ch == '-' || (ch >= '0' && ch <= '9')) {
    if (lex_number_span(c, &out->s, &out->flag))
      return JSON_ERR_PARSE;
    out->type = JSON_NUMBER;
    return JSON_OK;
  }
  return JSON_ERR_PARSE;
}

// ---- Object / array stepping ----------------------------------------------

// Find the value associated with a key inside the current object node.
// node.s spans the entire object including braces.
static enum json_err
object_step(struct json_node node, const char* key, struct json_node* out)
{
  if (node.type != JSON_OBJECT)
    return JSON_ERR_TYPE;
  size_t key_len = strlen(key);
  struct cslice c = node.s;
  if (cs_at_end(c) || *c.beg != '{')
    return JSON_ERR_PARSE;
  c.beg++;
  skip_ws(&c);
  if (!cs_at_end(c) && *c.beg == '}')
    return JSON_ERR_NOT_FOUND;
  for (;;) {
    skip_ws(&c);
    if (cs_at_end(c))
      return JSON_ERR_PARSE;
    struct cslice key_span;
    enum json_node_flag key_flag;
    if (lex_string_span(&c, &key_span, &key_flag))
      return JSON_ERR_PARSE;
    skip_ws(&c);
    if (cs_at_end(c) || *c.beg != ':')
      return JSON_ERR_PARSE;
    c.beg++;
    skip_ws(&c);
    int matched = 0;
    if (key_flag != JSON_NODE_FLAG_STR_ESCAPED) {
      size_t span_len = (size_t)(key_span.end - key_span.beg);
      matched =
        (span_len == key_len && memcmp(key_span.beg, key, key_len) == 0);
    }
    if (matched) {
      enum json_err e = lex_value(&c, out);
      if (e != JSON_OK)
        return e;
      return JSON_OK;
    }
    struct json_node skip;
    enum json_err e = lex_value(&c, &skip);
    if (e != JSON_OK)
      return e;
    skip_ws(&c);
    if (cs_at_end(c))
      return JSON_ERR_PARSE;
    if (*c.beg == ',') {
      c.beg++;
      continue;
    }
    if (*c.beg == '}')
      return JSON_ERR_NOT_FOUND;
    return JSON_ERR_PARSE;
  }
}

static enum json_err
array_step(struct json_node node, size_t want, struct json_node* out)
{
  if (node.type != JSON_ARRAY)
    return JSON_ERR_TYPE;
  struct cslice c = node.s;
  if (cs_at_end(c) || *c.beg != '[')
    return JSON_ERR_PARSE;
  c.beg++;
  skip_ws(&c);
  if (!cs_at_end(c) && *c.beg == ']')
    return JSON_ERR_NOT_FOUND;
  size_t i = 0;
  for (;;) {
    skip_ws(&c);
    struct json_node v;
    enum json_err e = lex_value(&c, &v);
    if (e != JSON_OK)
      return e;
    if (i == want) {
      *out = v;
      return JSON_OK;
    }
    i++;
    skip_ws(&c);
    if (cs_at_end(c))
      return JSON_ERR_PARSE;
    if (*c.beg == ',') {
      c.beg++;
      continue;
    }
    if (*c.beg == ']')
      return JSON_ERR_NOT_FOUND;
    return JSON_ERR_PARSE;
  }
}

// ---- WHERE evaluation ------------------------------------------------------

static int
str_eq_unescaped(struct json_node n, const char* rhs)
{
  if (n.type != JSON_STRING || n.flag == JSON_NODE_FLAG_STR_ESCAPED)
    return 0;
  size_t L = strlen(rhs);
  size_t span = cslice_len(n.s);
  return span == L && memcmp(n.s.beg, rhs, L) == 0;
}

static int
num_eq_literal(struct json_node n, const char* rhs)
{
  if (n.type != JSON_NUMBER)
    return 0;
  size_t L = strlen(rhs);
  size_t span = cslice_len(n.s);
  return span == L && memcmp(n.s.beg, rhs, L) == 0;
}

static int
bool_eq_literal(struct json_node n, const char* rhs)
{
  if (n.type != JSON_BOOL)
    return 0;
  size_t L = strlen(rhs);
  size_t span = cslice_len(n.s);
  return span == L && memcmp(n.s.beg, rhs, L) == 0;
}

// Forward decl: WHERE evaluation reuses json_resolve to apply a sub-query.
enum json_err
json_resolve(struct cslice src,
             const struct json_query* parts,
             size_t n_parts,
             struct json_node* out,
             struct json_error* err);

static int
where_holds(struct json_node node, const struct json_query* part)
{
  struct json_node leaf;
  if (json_resolve(node.s, part->where.part, part->where.n, &leaf, NULL) !=
      JSON_OK)
    return 0;
  switch (part->where.rhs_type) {
    case JSON_STRING:
      return str_eq_unescaped(leaf, part->where.rhs);
    case JSON_NUMBER:
      return num_eq_literal(leaf, part->where.rhs);
    case JSON_BOOL:
      return bool_eq_literal(leaf, part->where.rhs);
    default:
      return 0;
  }
}

// ---- Core evaluator --------------------------------------------------------
//
// The evaluator walks segments left-to-right, mutating `cur`. When it
// hits QUERY_ITER, it opens a frame on the iterator stack (or, in single-
// result mode, just descends into the first element). On a WHERE failure
// or NOT_FOUND deeper down, it backtracks to the most recent open frame
// and advances to that array's next element.

static enum json_err
iter_open_array(struct json_node arr, struct cslice* rest, uint8_t* saw_first)
{
  if (arr.type != JSON_ARRAY)
    return JSON_ERR_TYPE;
  struct cslice c = arr.s;
  if (cs_at_end(c) || *c.beg != '[')
    return JSON_ERR_PARSE;
  c.beg++;
  *rest = c;
  *saw_first = 0;
  return JSON_OK;
}

// Read the next element of an open array frame. Returns JSON_OK with the
// value, JSON_ERR_NOT_FOUND when the array is exhausted.
static enum json_err
iter_take(struct cslice* rest, uint8_t* saw_first, struct json_node* out)
{
  skip_ws(rest);
  if (cs_at_end(*rest))
    return JSON_ERR_PARSE;
  if (!*saw_first) {
    if (*rest->beg == ']') {
      rest->beg++;
      return JSON_ERR_NOT_FOUND;
    }
    *saw_first = 1;
  } else {
    if (*rest->beg == ']') {
      rest->beg++;
      return JSON_ERR_NOT_FOUND;
    }
    if (*rest->beg != ',')
      return JSON_ERR_PARSE;
    rest->beg++;
    skip_ws(rest);
  }
  return lex_value(rest, out);
}

// Apply parts[start .. n) to cur. QUERY_ITER pushes a frame onto the
// iterator stack and continues with the first element. `it` is always
// non-NULL: json_resolve owns a stack-local iter, and json_iter_next +
// backtrack_iter pass through the caller's. When the loop runs to
// completion (all parts consumed), the current value is the result.
static enum json_err
apply_from(struct json_node cur,
           const struct json_query* parts,
           size_t n,
           size_t start,
           struct json_iter* it,
           struct json_node* out);

static enum json_err
backtrack_iter(struct json_iter* it, struct json_node* out)
{
  while (it->depth > 0) {
    struct json_iter_frame* f = &it->frames[it->depth - 1];
    struct json_node next;
    enum json_err e = iter_take(&f->rest, &f->saw_first, &next);
    if (e == JSON_ERR_NOT_FOUND) {
      it->depth--;
      continue;
    }
    if (e != JSON_OK)
      return e;
    e = apply_from(next, it->parts, it->n_parts, f->part_index + 1, it, out);
    if (e == JSON_OK)
      return JSON_OK;
    if (e == JSON_ERR_NOT_FOUND)
      continue; // try next element
    return e;
  }
  return JSON_ERR_NOT_FOUND;
}

static enum json_err
apply_from(struct json_node cur,
           const struct json_query* parts,
           size_t n,
           size_t start,
           struct json_iter* it,
           struct json_node* out)
{
  for (size_t i = start; i < n; ++i) {
    // Switch with no default: -Wswitch-enum forces every new kind to land here.
    switch (parts[i].kind) {
      case QUERY_KEY: {
        enum json_err e = object_step(cur, parts[i].key, &cur);
        if (e != JSON_OK)
          return e;
        break;
      }
      case QUERY_INDEX: {
        enum json_err e = array_step(cur, parts[i].index, &cur);
        if (e != JSON_OK)
          return e;
        break;
      }
      case QUERY_ITER: {
        if (it->depth >= JSON_ITER_MAX_FRAMES)
          return JSON_ERR_OOM;
        struct json_iter_frame* f = &it->frames[it->depth];
        f->part_index = i;
        enum json_err e = iter_open_array(cur, &f->rest, &f->saw_first);
        if (e != JSON_OK)
          return e;
        it->depth++;
        // Take first element of this frame.
        struct json_node first;
        e = iter_take(&f->rest, &f->saw_first, &first);
        if (e == JSON_ERR_NOT_FOUND) {
          // Empty array: pop and propagate.
          it->depth--;
          return JSON_ERR_NOT_FOUND;
        }
        if (e != JSON_OK)
          return e;
        cur = first;
        break;
      }
      case QUERY_WHERE:
        if (!where_holds(cur, &parts[i]))
          return JSON_ERR_NOT_FOUND;
        // pass-through; keep cur
        break;
    }
  }
  *out = cur;
  return JSON_OK;
}

// ---- Public API ------------------------------------------------------------

enum json_err
json_resolve(struct cslice src,
             const struct json_query* parts,
             size_t n_parts,
             struct json_node* out,
             struct json_error* err)
{
  if (!out || (n_parts > 0 && !parts)) {
    if (err)
      *err = (struct json_error){ .code = JSON_ERR_INVALID, .offset = 0 };
    return JSON_ERR_INVALID;
  }
  struct cslice c = src;
  skip_ws(&c);
  struct json_node root;
  enum json_err e = lex_value(&c, &root);
  if (e != JSON_OK) {
    if (err)
      *err =
        (struct json_error){ .code = e, .offset = (size_t)(c.beg - src.beg) };
    return e;
  }
  // Try the path; on NOT_FOUND with iter parts, fall back to scanning
  // the iter alternatives via a temporary iterator.
  struct json_iter it;
  it.src = src;
  it.parts = parts;
  it.n_parts = n_parts;
  it.depth = 0;
  it.done = 0;
  e = apply_from(root, parts, n_parts, 0, &it, out);
  if (e == JSON_ERR_NOT_FOUND && it.depth > 0)
    e = backtrack_iter(&it, out);
  if (e != JSON_OK && err)
    *err =
      (struct json_error){ .code = e, .offset = (size_t)(c.beg - src.beg) };
  return e;
}

enum json_err
json_iter_init(struct cslice src,
               const struct json_query* parts,
               size_t n_parts,
               struct json_iter* it,
               struct json_error* err)
{
  if (!it || (n_parts > 0 && !parts)) {
    if (err)
      *err = (struct json_error){ .code = JSON_ERR_INVALID, .offset = 0 };
    return JSON_ERR_INVALID;
  }
  it->src = src;
  it->parts = parts;
  it->n_parts = n_parts;
  it->depth = 0;
  it->done = 0;
  it->last_err = (struct json_error){ JSON_OK, 0 };
  if (err)
    *err = (struct json_error){ JSON_OK, 0 };
  return JSON_OK;
}

enum json_err
json_iter_next(struct json_iter* it, struct json_node* out)
{
  if (!it || !out)
    return JSON_ERR_INVALID;
  if (it->done)
    return JSON_ERR_NOT_FOUND;
  if (it->depth == 0) {
    // First call: lex root and apply from the top.
    struct cslice c = it->src;
    skip_ws(&c);
    struct json_node root;
    enum json_err e = lex_value(&c, &root);
    if (e != JSON_OK) {
      it->done = 1;
      return e;
    }
    e = apply_from(root, it->parts, it->n_parts, 0, it, out);
    if (e == JSON_OK)
      return JSON_OK;
    if (e == JSON_ERR_NOT_FOUND && it->depth > 0)
      return backtrack_iter(it, out);
    it->done = 1;
    return e;
  }
  // Subsequent call: advance the deepest frame.
  enum json_err e = backtrack_iter(it, out);
  if (e != JSON_OK)
    it->done = 1;
  return e;
}

// ---- Primitive conversions -------------------------------------------------

enum json_err
json_as_int(struct json_node n, int64_t* out)
{
  if (!out)
    return JSON_ERR_INVALID;
  if (n.type != JSON_NUMBER)
    return JSON_ERR_TYPE;
  if (n.flag == JSON_NODE_FLAG_NUM_FLOAT)
    return JSON_ERR_RANGE;
  size_t len = cslice_len(n.s);
  char buf[64];
  if (len == 0 || len >= sizeof(buf))
    return JSON_ERR_RANGE;
  memcpy(buf, n.s.beg, len);
  buf[len] = 0;
  errno = 0;
  char* end = NULL;
  long long v = strtoll(buf, &end, 10);
  if (errno != 0 || end != buf + len)
    return JSON_ERR_RANGE;
  *out = (int64_t)v;
  return JSON_OK;
}

enum json_err
json_as_uint(struct json_node n, uint64_t* out)
{
  if (!out)
    return JSON_ERR_INVALID;
  if (n.type != JSON_NUMBER)
    return JSON_ERR_TYPE;
  if (n.flag == JSON_NODE_FLAG_NUM_FLOAT)
    return JSON_ERR_RANGE;
  size_t len = cslice_len(n.s);
  if (len > 0 && n.s.beg[0] == '-')
    return JSON_ERR_RANGE;
  char buf[64];
  if (len == 0 || len >= sizeof(buf))
    return JSON_ERR_RANGE;
  memcpy(buf, n.s.beg, len);
  buf[len] = 0;
  errno = 0;
  char* end = NULL;
  unsigned long long v = strtoull(buf, &end, 10);
  if (errno != 0 || end != buf + len)
    return JSON_ERR_RANGE;
  *out = (uint64_t)v;
  return JSON_OK;
}

enum json_err
json_as_double(struct json_node n, double* out)
{
  if (!out)
    return JSON_ERR_INVALID;
  if (n.type != JSON_NUMBER)
    return JSON_ERR_TYPE;
  size_t len = cslice_len(n.s);
  char buf[64];
  if (len == 0 || len >= sizeof(buf))
    return JSON_ERR_RANGE;
  memcpy(buf, n.s.beg, len);
  buf[len] = 0;
  errno = 0;
  char* end = NULL;
  double v = strtod(buf, &end);
  if (errno != 0 || end != buf + len)
    return JSON_ERR_RANGE;
  *out = v;
  return JSON_OK;
}

enum json_err
json_as_bool(struct json_node n, int* out)
{
  if (!out)
    return JSON_ERR_INVALID;
  if (n.type != JSON_BOOL)
    return JSON_ERR_TYPE;
  size_t len = cslice_len(n.s);
  if (len == 4 && memcmp(n.s.beg, "true", 4) == 0) {
    *out = 1;
    return JSON_OK;
  }
  if (len == 5 && memcmp(n.s.beg, "false", 5) == 0) {
    *out = 0;
    return JSON_OK;
  }
  // Defense-in-depth: lex_value only emits JSON_BOOL nodes whose span
  // is exactly "true" or "false", so this path is unreachable on a
  // well-lexed node. Kept (rather than __builtin_unreachable) so an
  // invariant break shows up as a clean error code, not UB.
  return JSON_ERR_RANGE;
}

int
json_str_eq(struct json_node n, const char* lit)
{
  if (n.type != JSON_STRING || n.flag == JSON_NODE_FLAG_STR_ESCAPED || !lit)
    return 0;
  size_t L = strlen(lit);
  size_t span = cslice_len(n.s);
  return span == L && memcmp(n.s.beg, lit, L) == 0;
}
