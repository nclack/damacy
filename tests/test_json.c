#include "util/json.h"

#include <stdio.h>
#include <string.h>

#define SRC(s) ((struct cslice){ .beg = (s), .end = (s) + sizeof(s) - 1 })
#define EXPECT(cond)                                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(                                                                 \
        stderr, "  expect failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static int
test_lex_int(void)
{
  static const struct json_seg segs[] = { { .kind = SEG_END } };
  static const struct json_pred pred = { .segs = segs };
  struct json_node n;
  EXPECT(json_resolve(SRC("42"), &pred, &n, NULL) == JSON_OK);
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
  static const struct json_seg segs[] = { { .kind = SEG_END } };
  static const struct json_pred pred = { .segs = segs };
  struct json_node n;
  EXPECT(json_resolve(SRC("1.5e2"), &pred, &n, NULL) == JSON_OK);
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
  static const struct json_seg segs[] = { { .kind = SEG_END } };
  static const struct json_pred pred = { .segs = segs };
  struct json_node n;
  EXPECT(json_resolve(SRC("\"hello\""), &pred, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_STRING);
  EXPECT(n.flag == JSON_NODE_FLAG_NONE);
  EXPECT(json_str_eq(n, "hello"));
  EXPECT(!json_str_eq(n, "world"));
  return 0;
}

static int
test_lex_string_escaped(void)
{
  static const struct json_seg segs[] = { { .kind = SEG_END } };
  static const struct json_pred pred = { .segs = segs };
  struct json_node n;
  // The unescaped logical content is "a\"b" but the raw bytes include
  // the backslash. json_str_eq must reject the escaped node even if the
  // logical content matches.
  EXPECT(json_resolve(SRC("\"a\\\"b\""), &pred, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_STRING);
  EXPECT(n.flag == JSON_NODE_FLAG_STR_ESCAPED);
  EXPECT(!json_str_eq(n, "a\"b"));
  return 0;
}

static int
test_lex_bool_null(void)
{
  static const struct json_seg segs[] = { { .kind = SEG_END } };
  static const struct json_pred pred = { .segs = segs };
  struct json_node n;
  EXPECT(json_resolve(SRC("true"), &pred, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_BOOL);
  int b;
  EXPECT(json_as_bool(n, &b) == JSON_OK && b == 1);
  EXPECT(json_resolve(SRC("false"), &pred, &n, NULL) == JSON_OK);
  EXPECT(json_as_bool(n, &b) == JSON_OK && b == 0);
  EXPECT(json_resolve(SRC("null"), &pred, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_NULL);
  return 0;
}

static int
test_object_key_descent(void)
{
  static const struct json_seg path[] = {
    { .kind = SEG_KEY, .key = "a" },
    { .kind = SEG_KEY, .key = "b" },
    { .kind = SEG_END },
  };
  static const struct json_pred pred = { .segs = path };
  struct json_node n;
  EXPECT(json_resolve(SRC("{\"a\":{\"b\":1}}"), &pred, &n, NULL) == JSON_OK);
  int64_t v;
  EXPECT(json_as_int(n, &v) == JSON_OK && v == 1);
  return 0;
}

static int
test_object_key_skip(void)
{
  // Inner objects, arrays, escaped strings on the way must all be skipped.
  static const struct json_seg path[] = {
    { .kind = SEG_KEY, .key = "wanted" },
    { .kind = SEG_END },
  };
  static const struct json_pred pred = { .segs = path };
  struct json_node n;
  static const char src[] =
    "{\"a\":{\"x\":[1,2,3]},\"b\":\"\\\"q\\\"\",\"wanted\":7}";
  EXPECT(json_resolve(SRC(src), &pred, &n, NULL) == JSON_OK);
  int64_t v;
  EXPECT(json_as_int(n, &v) == JSON_OK && v == 7);
  return 0;
}

static int
test_array_index(void)
{
  static const struct json_seg path[] = {
    { .kind = SEG_INDEX, .index = 1 },
    { .kind = SEG_END },
  };
  static const struct json_pred pred = { .segs = path };
  struct json_node n;
  EXPECT(json_resolve(SRC("[10,20,30]"), &pred, &n, NULL) == JSON_OK);
  int64_t v;
  EXPECT(json_as_int(n, &v) == JSON_OK && v == 20);
  return 0;
}

static int
test_iter_basic(void)
{
  static const struct json_seg segs[] = {
    { .kind = SEG_ITER },
    { .kind = SEG_END },
  };
  static const struct json_pred pred = { .segs = segs };
  struct json_iter it;
  EXPECT(json_iter_init(SRC("[1,2,3]"), &pred, &it, NULL) == JSON_OK);
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
  static const struct json_seg sub[] = {
    { .kind = SEG_KEY, .key = "name" },
    { .kind = SEG_END },
  };
  static const struct json_seg path[] = {
    { .kind = SEG_KEY, .key = "codecs" },
    { .kind = SEG_ITER },
    { .kind = SEG_WHERE,
      .where = { .path = sub,
                 .rhs = "sharding_indexed",
                 .rhs_type = JSON_STRING } },
    { .kind = SEG_END },
  };
  static const struct json_pred pred = { .segs = path };
  static const char src[] = "{\"codecs\":[{\"name\":\"bytes\"},"
                            "{\"name\":\"sharding_indexed\",\"id\":42}]}";
  struct json_node n;
  EXPECT(json_resolve(SRC(src), &pred, &n, NULL) == JSON_OK);
  EXPECT(n.type == JSON_OBJECT);
  // Drill into the matched element to confirm we got the right codec.
  static const struct json_seg id_path[] = {
    { .kind = SEG_KEY, .key = "id" },
    { .kind = SEG_END },
  };
  static const struct json_pred id_pred = { .segs = id_path };
  struct json_node id;
  EXPECT(json_resolve(n.s, &id_pred, &id, NULL) == JSON_OK);
  int64_t v;
  EXPECT(json_as_int(id, &v) == JSON_OK && v == 42);
  return 0;
}

static int
test_where_iter_filtered(void)
{
  // codecs[] | select(.name == "match") — expect exactly two emissions.
  static const struct json_seg sub[] = {
    { .kind = SEG_KEY, .key = "name" },
    { .kind = SEG_END },
  };
  static const struct json_seg path[] = {
    { .kind = SEG_ITER },
    { .kind = SEG_WHERE,
      .where = { .path = sub, .rhs = "match", .rhs_type = JSON_STRING } },
    { .kind = SEG_END },
  };
  static const struct json_pred pred = { .segs = path };
  static const char src[] = "[{\"name\":\"match\",\"id\":1},"
                            "{\"name\":\"skip\"},"
                            "{\"name\":\"match\",\"id\":2}]";
  struct json_iter it;
  EXPECT(json_iter_init(SRC(src), &pred, &it, NULL) == JSON_OK);

  struct json_node n;
  EXPECT(json_iter_next(&it, &n) == JSON_OK);
  static const struct json_seg id_path[] = {
    { .kind = SEG_KEY, .key = "id" },
    { .kind = SEG_END },
  };
  static const struct json_pred id_pred = { .segs = id_path };
  struct json_node id;
  EXPECT(json_resolve(n.s, &id_pred, &id, NULL) == JSON_OK);
  int64_t v;
  EXPECT(json_as_int(id, &v) == JSON_OK && v == 1);

  EXPECT(json_iter_next(&it, &n) == JSON_OK);
  EXPECT(json_resolve(n.s, &id_pred, &id, NULL) == JSON_OK);
  EXPECT(json_as_int(id, &v) == JSON_OK && v == 2);

  EXPECT(json_iter_next(&it, &n) == JSON_ERR_NOT_FOUND);
  return 0;
}

static int
test_negative_not_found(void)
{
  static const struct json_seg path[] = {
    { .kind = SEG_KEY, .key = "missing" },
    { .kind = SEG_END },
  };
  static const struct json_pred pred = { .segs = path };
  struct json_node n;
  EXPECT(json_resolve(SRC("{\"a\":1}"), &pred, &n, NULL) == JSON_ERR_NOT_FOUND);
  return 0;
}

static int
test_negative_type(void)
{
  static const struct json_seg path[] = {
    { .kind = SEG_KEY, .key = "a" },
    { .kind = SEG_END },
  };
  static const struct json_pred pred = { .segs = path };
  struct json_node n;
  // Root is a number, not an object: SEG_KEY should fail with TYPE.
  EXPECT(json_resolve(SRC("42"), &pred, &n, NULL) == JSON_ERR_TYPE);
  return 0;
}

static int
test_negative_parse(void)
{
  // The lexer is intentionally lazy — bytes inside a skipped container
  // are not validated until something tries to read them. So the parse
  // error must come from a malformation that derails brace-balancing:
  // here, an unterminated array.
  static const struct json_seg segs[] = { { .kind = SEG_END } };
  static const struct json_pred pred = { .segs = segs };
  struct json_node n;
  struct json_error err;
  EXPECT(json_resolve(SRC("[1,2"), &pred, &n, &err) == JSON_ERR_PARSE);
  EXPECT(err.code == JSON_ERR_PARSE);

  // And: descending into a malformed value DOES fail, because the value
  // lexer rejects '}' where a value should be.
  static const struct json_seg path[] = {
    { .kind = SEG_KEY, .key = "a" },
    { .kind = SEG_END },
  };
  static const struct json_pred pred2 = { .segs = path };
  EXPECT(json_resolve(SRC("{\"a\":}"), &pred2, &n, NULL) == JSON_ERR_PARSE);
  return 0;
}

static int
test_primitive_conversions(void)
{
  static const struct json_seg segs[] = { { .kind = SEG_END } };
  static const struct json_pred pred = { .segs = segs };
  struct json_node n;
  // Wrong-type checks
  EXPECT(json_resolve(SRC("\"x\""), &pred, &n, NULL) == JSON_OK);
  int64_t i;
  EXPECT(json_as_int(n, &i) == JSON_ERR_TYPE);
  // uint with negative literal
  EXPECT(json_resolve(SRC("-3"), &pred, &n, NULL) == JSON_OK);
  uint64_t u;
  EXPECT(json_as_uint(n, &u) == JSON_ERR_RANGE);
  // double on int
  EXPECT(json_resolve(SRC("17"), &pred, &n, NULL) == JSON_OK);
  double d;
  EXPECT(json_as_double(n, &d) == JSON_OK && d == 17.0);
  return 0;
}

struct test_entry
{
  const char* name;
  int (*fn)(void);
};

#define TEST(name) { #name, name }

int
main(void)
{
  static const struct test_entry tests[] = {
    TEST(test_lex_int),
    TEST(test_lex_float),
    TEST(test_lex_string_plain),
    TEST(test_lex_string_escaped),
    TEST(test_lex_bool_null),
    TEST(test_object_key_descent),
    TEST(test_object_key_skip),
    TEST(test_array_index),
    TEST(test_iter_basic),
    TEST(test_where_resolve_first),
    TEST(test_where_iter_filtered),
    TEST(test_negative_not_found),
    TEST(test_negative_type),
    TEST(test_negative_parse),
    TEST(test_primitive_conversions),
  };
  int failed = 0;
  for (size_t i = 0; i < sizeof tests / sizeof tests[0]; ++i) {
    int rc = tests[i].fn();
    if (rc) {
      printf("FAIL %s\n", tests[i].name);
      failed++;
    } else {
      printf("OK   %s\n", tests[i].name);
    }
  }
  if (failed)
    printf("%d test(s) failed\n", failed);
  return failed ? 1 : 0;
}
