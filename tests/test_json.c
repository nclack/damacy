#include "expect.h"
#include "util/json.h"
#include "util/prelude.h"

#include <string.h>

#define SRC(s) ((struct cslice){ .beg = (s), .end = (s) + sizeof(s) - 1 })

static int
test_lex_int(void)
{
  struct json_node n;
  EXPECT(json_resolve(SRC("42"), NULL, 0, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_NUMBER);
  EXPECT(n.flag == JSON_NODE_FLAG_NONE);
  int64_t v;
  EXPECT(json_as_int(n, &v) == JSON_OK);
  EXPECT(v == 42);
  return 0;
}

static int
test_lex_float(void)
{
  struct json_node n;
  EXPECT(json_resolve(SRC("1.5e2"), NULL, 0, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_NUMBER);
  EXPECT(n.flag == JSON_NODE_FLAG_NUM_FLOAT);
  EXPECT(json_as_int(n, &(int64_t){ 0 }) == JSON_ERR_RANGE);
  double d;
  EXPECT(json_as_double(n, &d) == JSON_OK);
  EXPECT(d == 150.0);
  return 0;
}

static int
test_lex_string_plain(void)
{
  struct json_node n;
  EXPECT(json_resolve(SRC("\"hello\""), NULL, 0, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_STRING);
  EXPECT(n.flag == JSON_NODE_FLAG_NONE);
  EXPECT(json_str_eq(n, "hello"));
  EXPECT(!json_str_eq(n, "world"));
  return 0;
}

static int
test_lex_string_escaped(void)
{
  struct json_node n;
  // The unescaped logical content is "a\"b" but the raw bytes include
  // the backslash. json_str_eq must reject the escaped node even if the
  // logical content matches.
  EXPECT(json_resolve(SRC("\"a\\\"b\""), NULL, 0, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_STRING);
  EXPECT(n.flag == JSON_NODE_FLAG_STR_ESCAPED);
  EXPECT(!json_str_eq(n, "a\"b"));
  return 0;
}

static int
test_lex_bool_null(void)
{
  struct json_node n;
  EXPECT(json_resolve(SRC("true"), NULL, 0, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_BOOL);
  int b;
  EXPECT(json_as_bool(n, &b) == JSON_OK && b == 1);
  EXPECT(json_resolve(SRC("false"), NULL, 0, &n, NULL) == JSON_OK);
  EXPECT(json_as_bool(n, &b) == JSON_OK && b == 0);
  EXPECT(json_resolve(SRC("null"), NULL, 0, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_NULL);
  return 0;
}

static int
test_object_key_descent(void)
{
  static const struct json_query path[] = { { .kind = QUERY_KEY, .key = "a" },
                                            { .kind = QUERY_KEY, .key = "b" } };
  struct json_node n;
  EXPECT(json_resolve(
           SRC("{\"a\":{\"b\":1}}"), path, countof(path), &n, NULL) == JSON_OK);
  int64_t v;
  EXPECT(json_as_int(n, &v) == JSON_OK && v == 1);
  return 0;
}

static int
test_object_key_skip(void)
{
  // Inner objects, arrays, escaped strings on the way must all be skipped.
  static const struct json_query path[] = { { .kind = QUERY_KEY,
                                              .key = "wanted" } };
  struct json_node n;
  static const char src[] =
    "{\"a\":{\"x\":[1,2,3]},\"b\":\"\\\"q\\\"\",\"wanted\":7}";
  EXPECT(json_resolve(SRC(src), path, countof(path), &n, NULL) == JSON_OK);
  int64_t v;
  EXPECT(json_as_int(n, &v) == JSON_OK && v == 7);
  return 0;
}

static int
test_array_index(void)
{
  static const struct json_query path[] = { { .kind = QUERY_INDEX,
                                              .index = 1 } };
  struct json_node n;
  EXPECT(json_resolve(SRC("[10,20,30]"), path, countof(path), &n, NULL) ==
         JSON_OK);
  int64_t v;
  EXPECT(json_as_int(n, &v) == JSON_OK && v == 20);
  return 0;
}

static int
test_iter_basic(void)
{
  static const struct json_query parts[] = { { .kind = QUERY_ITER } };
  struct json_iter it;
  EXPECT(json_iter_init(SRC("[1,2,3]"), parts, countof(parts), &it, NULL) ==
         JSON_OK);
  int64_t expected[] = { 1, 2, 3 };
  for (int i = 0; i < 3; ++i) {
    struct json_node n;
    EXPECT(json_iter_next(&it, &n) == JSON_OK);
    int64_t v;
    EXPECT(json_as_int(n, &v) == JSON_OK && v == expected[i]);
  }
  struct json_node n;
  EXPECT(json_iter_next(&it, &n) == JSON_ERR_NOT_FOUND);
  return 0;
}

static int
test_where_resolve_first(void)
{
  // codecs[] | select(.name == "sharding_indexed")
  static const struct json_query sub[] = { { .kind = QUERY_KEY,
                                             .key = "name" } };
  static const struct json_query path[] = {
    { .kind = QUERY_KEY, .key = "codecs" },
    { .kind = QUERY_ITER },
    { .kind = QUERY_WHERE,
      .where = { .part = sub,
                 .n = countof(sub),
                 .rhs = "sharding_indexed",
                 .rhs_type = JSON_STRING } }
  };
  static const char src[] = "{\"codecs\":[{\"name\":\"bytes\"},"
                            "{\"name\":\"sharding_indexed\",\"id\":42}]}";
  struct json_node n;
  EXPECT(json_resolve(SRC(src), path, countof(path), &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_OBJECT);
  // Drill into the matched element to confirm we got the right codec.
  static const struct json_query id_path[] = { { .kind = QUERY_KEY,
                                                 .key = "id" } };
  struct json_node id;
  EXPECT(json_resolve(n.s, id_path, countof(id_path), &id, NULL) == JSON_OK);
  int64_t v;
  EXPECT(json_as_int(id, &v) == JSON_OK && v == 42);
  return 0;
}

static int
test_where_iter_filtered(void)
{
  // codecs[] | select(.name == "match") — expect exactly two emissions.
  static const struct json_query sub[] = { { .kind = QUERY_KEY,
                                             .key = "name" } };
  static const struct json_query path[] = { { .kind = QUERY_ITER },
                                            { .kind = QUERY_WHERE,
                                              .where = { .part = sub,
                                                         .n = countof(sub),
                                                         .rhs = "match",
                                                         .rhs_type =
                                                           JSON_STRING } } };
  static const char src[] = "[{\"name\":\"match\",\"id\":1},"
                            "{\"name\":\"skip\"},"
                            "{\"name\":\"match\",\"id\":2}]";
  struct json_iter it;
  EXPECT(json_iter_init(SRC(src), path, countof(path), &it, NULL) == JSON_OK);

  struct json_node n;
  EXPECT(json_iter_next(&it, &n) == JSON_OK);
  static const struct json_query id_path[] = { { .kind = QUERY_KEY,
                                                 .key = "id" } };
  struct json_node id;
  EXPECT(json_resolve(n.s, id_path, countof(id_path), &id, NULL) == JSON_OK);
  int64_t v;
  EXPECT(json_as_int(id, &v) == JSON_OK && v == 1);

  EXPECT(json_iter_next(&it, &n) == JSON_OK);
  EXPECT(json_resolve(n.s, id_path, countof(id_path), &id, NULL) == JSON_OK);
  EXPECT(json_as_int(id, &v) == JSON_OK && v == 2);

  EXPECT(json_iter_next(&it, &n) == JSON_ERR_NOT_FOUND);
  return 0;
}

static int
test_negative_not_found(void)
{
  static const struct json_query path[] = { { .kind = QUERY_KEY,
                                              .key = "missing" } };
  struct json_node n;
  EXPECT(json_resolve(SRC("{\"a\":1}"), path, countof(path), &n, NULL) ==
         JSON_ERR_NOT_FOUND);
  return 0;
}

static int
test_negative_type(void)
{
  static const struct json_query path[] = { { .kind = QUERY_KEY, .key = "a" } };
  struct json_node n;
  // Root is a number, not an object: QUERY_KEY should fail with TYPE.
  EXPECT(json_resolve(SRC("42"), path, countof(path), &n, NULL) ==
         JSON_ERR_TYPE);
  return 0;
}

static int
test_negative_parse(void)
{
  // The lexer is intentionally lazy — bytes inside a skipped container
  // are not validated until something tries to read them. So the parse
  // error must come from a malformation that derails brace-balancing:
  // here, an unterminated array.
  struct json_node n;
  struct json_error err;
  EXPECT(json_resolve(SRC("[1,2"), NULL, 0, &n, &err) == JSON_ERR_PARSE);
  EXPECT(err.code == JSON_ERR_PARSE);

  // And: descending into a malformed value DOES fail, because the value
  // lexer rejects '}' where a value should be.
  static const struct json_query path[] = { { .kind = QUERY_KEY, .key = "a" } };
  EXPECT(json_resolve(SRC("{\"a\":}"), path, countof(path), &n, NULL) ==
         JSON_ERR_PARSE);
  return 0;
}

static int
test_primitive_conversions(void)
{
  struct json_node n;
  // Wrong-type checks
  EXPECT(json_resolve(SRC("\"x\""), NULL, 0, &n, NULL) == JSON_OK);
  int64_t i;
  EXPECT(json_as_int(n, &i) == JSON_ERR_TYPE);
  // uint with negative literal
  EXPECT(json_resolve(SRC("-3"), NULL, 0, &n, NULL) == JSON_OK);
  uint64_t u;
  EXPECT(json_as_uint(n, &u) == JSON_ERR_RANGE);
  // double on int
  EXPECT(json_resolve(SRC("17"), NULL, 0, &n, NULL) == JSON_OK);
  double d;
  EXPECT(json_as_double(n, &d) == JSON_OK && d == 17.0);
  return 0;
}

static int
test_null_args(void)
{
  static const char src_buf[] = "1";
  static const struct cslice src = { .beg = src_buf,
                                     .end = src_buf + sizeof(src_buf) - 1 };
  struct json_node n;
  struct json_iter it;
  struct json_error err;

  // (parts=NULL, n_parts=0) is the empty query — returns the root.
  EXPECT(json_resolve(src, NULL, 0, &n, NULL) == JSON_OK);

  // (parts=NULL, n_parts>0) is invalid; out=NULL is invalid.
  EXPECT(json_resolve(src, NULL, 1, &n, NULL) == JSON_ERR_INVALID);
  err.code = JSON_OK;
  EXPECT(json_resolve(src, NULL, 1, &n, &err) == JSON_ERR_INVALID);
  EXPECT(err.code == JSON_ERR_INVALID);
  EXPECT(json_resolve(src, NULL, 0, NULL, NULL) == JSON_ERR_INVALID);

  // json_iter_init: same shape — empty query is valid; bad arity is not.
  EXPECT(json_iter_init(src, NULL, 0, &it, NULL) == JSON_OK);
  EXPECT(json_iter_init(src, NULL, 1, &it, NULL) == JSON_ERR_INVALID);
  err.code = JSON_OK;
  EXPECT(json_iter_init(src, NULL, 1, &it, &err) == JSON_ERR_INVALID);
  EXPECT(err.code == JSON_ERR_INVALID);
  EXPECT(json_iter_init(src, NULL, 0, NULL, NULL) == JSON_ERR_INVALID);

  // json_iter_next: it / out.
  EXPECT(json_iter_next(NULL, &n) == JSON_ERR_INVALID);
  EXPECT(json_iter_init(src, NULL, 0, &it, NULL) == JSON_OK);
  EXPECT(json_iter_next(&it, NULL) == JSON_ERR_INVALID);

  // json_as_*: out=NULL.
  EXPECT(json_resolve(src, NULL, 0, &n, NULL) == JSON_OK);
  EXPECT(json_as_int(n, NULL) == JSON_ERR_INVALID);
  EXPECT(json_as_uint(n, NULL) == JSON_ERR_INVALID);
  EXPECT(json_as_double(n, NULL) == JSON_ERR_INVALID);
  EXPECT(json_as_bool(n, NULL) == JSON_ERR_INVALID);
  return 0;
}

static int
test_err_offset_set(void)
{
  // On parse failure, json_resolve must populate err with the parse
  // offset. The exact offset depends on lexer internals — assert only
  // that something past the start was reached.
  struct json_node n;
  struct json_error err = { JSON_OK, 0 };
  EXPECT(json_resolve(SRC("[1,2"), NULL, 0, &n, &err) == JSON_ERR_PARSE);
  EXPECT(err.code == JSON_ERR_PARSE);
  EXPECT(err.offset > 0);

  // On iter_init success, err is cleared to JSON_OK.
  err = (struct json_error){ JSON_ERR_PARSE, 99 };
  struct json_iter it;
  EXPECT(json_iter_init(SRC("[1]"), NULL, 0, &it, &err) == JSON_OK);
  EXPECT(err.code == JSON_OK);
  EXPECT(err.offset == 0);
  return 0;
}

int
main(void)
{
  RUN(test_lex_int);
  RUN(test_lex_float);
  RUN(test_lex_string_plain);
  RUN(test_lex_string_escaped);
  RUN(test_lex_bool_null);
  RUN(test_object_key_descent);
  RUN(test_object_key_skip);
  RUN(test_array_index);
  RUN(test_iter_basic);
  RUN(test_where_resolve_first);
  RUN(test_where_iter_filtered);
  RUN(test_negative_not_found);
  RUN(test_negative_type);
  RUN(test_negative_parse);
  RUN(test_primitive_conversions);
  RUN(test_null_args);
  RUN(test_err_offset_set);
  log_info("all tests passed");
  return 0;
}
