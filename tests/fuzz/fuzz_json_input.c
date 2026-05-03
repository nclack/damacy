// Input-bytes fuzzer for util/json.
//
// libFuzzer feeds us a (data, size) buffer; we point a cslice at it and
// drive both the eager (json_resolve) and lazy-iterator (json_iter_next)
// code paths. The lazy parser only lexes what a query touches, so a
// passive harness would leave most of the parser uncovered. To force a
// full walk we run a QUERY_ITER over the root and call every json_as_*
// converter on each emitted node — this drags the lexer through every
// child value of the input.
//
// Sanitizers (ASan + UBSan) catch OOB reads on truncated input and
// signed-overflow / shift UB inside json_as_int and friends.
#include "util/json.h"

#include <stddef.h>
#include <stdint.h>

#define countof(arr) (sizeof(arr) / sizeof((arr)[0]))

static void
walk_node(struct json_node n)
{
  // Run every converter regardless of node type. The error-returning
  // paths matter as much as the success paths — they hit the integer
  // and float lexers which are easy places to have UB.
  int64_t i;
  uint64_t u;
  double d;
  int b;
  (void)json_as_int(n, &i);
  (void)json_as_uint(n, &u);
  (void)json_as_double(n, &d);
  (void)json_as_bool(n, &b);
  (void)json_str_eq(n, "");
  (void)json_str_eq(n, "x");
}

int
LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
  const struct cslice src = { .beg = (const char*)data,
                              .end = (const char*)data + size };

  // We always pass a real json_error pointer through to the parser so
  // the *err = ... fill-in branches are exercised. Zero-init so MSan
  // doesn't flag the parser's internal copy of an uninitialized
  // struct on success paths (where the parser legitimately doesn't
  // write *err). We don't assert on err.offset — its exact value
  // depends on lexer internals — but ASan/UBSan see the arithmetic at
  // json.c:559-561 and 574-576.
  struct json_error err = { JSON_OK, 0 };

  // 1. Resolve at the root with an empty path. Exercises top-level
  //    value lexing for every JSON type.
  {
    struct json_node n;
    if (json_resolve(src, NULL, 0, &n, &err) == JSON_OK)
      walk_node(n);
  }

  // 2. Iterate over root as if it were an array. If src is an object or
  //    scalar the call returns an error and we move on; if it's an
  //    array, every element is lexed and converted.
  {
    static const struct json_query parts[] = { { .kind = QUERY_ITER } };
    struct json_iter it;
    if (json_iter_init(src, parts, countof(parts), &it, &err) == JSON_OK) {
      struct json_node n;
      // Cap iterations defensively; libFuzzer's input is bounded but a
      // pathological array of zero-width tokens shouldn't dominate runtime.
      for (int k = 0; k < 4096; ++k) {
        if (json_iter_next(&it, &n) != JSON_OK)
          break;
        walk_node(n);
      }
      // Re-call after exhaustion so the it->done branch is exercised.
      (void)json_iter_next(&it, &n);
    }
  }

  // 3. Look up a couple of common keys. Exercises the object lexer's
  //    skip-past-value logic for non-matching keys, which is where
  //    lazy parsers historically miscount braces and quotes.
  {
    static const char* const keys[] = { "name",   "shape", "codecs",
                                        "chunks", "id",    "" };
    for (size_t k = 0; k < sizeof(keys) / sizeof(keys[0]); ++k) {
      const struct json_query path[] = { { .kind = QUERY_KEY,
                                           .key = keys[k] } };
      struct json_node n;
      if (json_resolve(src, path, countof(path), &n, &err) == JSON_OK)
        walk_node(n);
    }
  }

  return 0;
}
