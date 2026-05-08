# Using `util/json.h`

This is a manual for callers of the JSON reader. The header at `src/util/json.h`
is the authoritative reference; this document teaches you to use it.

## Mental model

There are three main ideas driving the design:

**Zero alloc, zero copy.** Every result you get back is a `cslice` (constant
slice) pointing into your original source buffer. The reader never copies bytes.
There's no "json document" object to free.

**Lazy.** Nothing is parsed until a query asks for it. The bytes belonging to
subtrees you never touch are never lexed. This makes the reader fault tolerant,
and should mean the reader is fast on huge documents you only sample sparsely.

**Query-driven.** You don't walk a tree by hand. You build a *query* — an array
of parts plus its length — and either resolve it to a single value
(`json_resolve`) or open it as an iterator (`json_iter_init` / `json_iter_next`).

The point is to read specific fields out of large documents without paying for
what you don't need.

## Building blocks

### Input

The input is given as a constant slice or `cslice` which is just `{const char*
beg, const char* end}` — a half-open byte range. Build one from any contiguous
JSON text:

```c
const char* text = "{\"hello\": 1}";
struct cslice src = { .beg = text, .end = text + strlen(text) };
```

`src` must outlive every node derived from it. The reader does not own
the bytes.

### Result: `struct json_node`

Every successful call hands back:

```c
struct json_node {
  enum json_type      type;  // JSON_NULL/BOOL/NUMBER/STRING/ARRAY/OBJECT
  enum json_node_flag flag;  // JSON_NODE_FLAG_NONE / STR_ESCAPED / NUM_FLOAT
  struct cslice       s;     // span into src
};
```

The span is the *bytes* of the value:

- Numbers: the digits/exponent (e.g., `1.5e2`).
- Strings: bytes *between* the quotes (so `"hello"` gives a 5-byte span).
- Arrays/objects: from the opening `[`/`{` through the matching close.

To get a typed value out of a node, use the converters (below).

### Query: an array of `json_query` parts

A query is a plain C array of `struct json_query` parts. Four part kinds:

| Kind          | Meaning                                              | Field            |
|---------------|------------------------------------------------------|------------------|
| `QUERY_KEY`   | descend into `object[key]`                           | `.key` (NUL-term)|
| `QUERY_INDEX` | array element at index                               | `.index` (size_t)|
| `QUERY_ITER`  | iterate every array element                          | —                |
| `QUERY_WHERE` | filter: keep current iff sub-query matches a literal | `.where { part, n, rhs, rhs_type }` |

Queries are usually defined at compile-time as constants:

```c
static const struct json_query path[] = {
  { .kind = QUERY_KEY,   .key = "shape" },
  { .kind = QUERY_INDEX, .index = 0 },
};
json_resolve(src, path, countof(path), &n, NULL);
```

`countof(arr)` here is the usual `(sizeof(arr) / sizeof((arr)[0]))` macro for
getting the array length.

## The simplest call

An empty query (`NULL, 0`) returns the root value:

```c
const char text[] = "42";
struct cslice src = { .beg = text, .end = text + sizeof(text) - 1 };

struct json_node n;
if (json_resolve(src, NULL, 0, &n, NULL) != JSON_OK) { /* handle */ }
// n.type == JSON_NUMBER, n.s spans "42"

int64_t v;
json_as_int(n, &v);  // v == 42
```

The last argument to `json_resolve` is used to return parse-error position
info. Pass `NULL` when you don't need that, or pass `&err` to receive a `struct
json_error { code, offset }`.

## Walking objects and arrays

Reach into `obj.shape[0]`:

```c
const char text[] = "{\"shape\":[1024,1024,128]}";
struct cslice src = { .beg = text, .end = text + sizeof text - 1 };

static const struct json_query path[] = {
  { .kind = QUERY_KEY,   .key = "shape" },
  { .kind = QUERY_INDEX, .index = 0 },
};

struct json_node n;
json_resolve(src, path, countof(path), &n, NULL);
int64_t v;
json_as_int(n, &v);  // v == 1024
```

Object keys match by *raw bytes* against the on-disk JSON. A key written with
escape sequences (e.g., `"s"` in the source) won't match against `"s"` —
the reader does no decode. In practice: write your `.key` strings to match the
bytes you expect on disk.

Object descent skips lazily. In the example above, `1024,1024,128` aren't lexed
as numbers until `QUERY_INDEX 0` actually demands an array element.

## Iteration

To visit every element of an array, use `json_iter_init` + `json_iter_next`:

```c
static const struct json_query parts[] = {
  { .kind = QUERY_KEY,  .key = "shape" },
  { .kind = QUERY_ITER },
};

struct json_iter it;
json_iter_init(src, parts, countof(parts), &it, NULL);

struct json_node n;
while (json_iter_next(&it, &n) == JSON_OK) {
  int64_t v;
  json_as_int(n, &v);
  // ... do something with v
}
// iter_next eventually returns JSON_ERR_NOT_FOUND; that's the loop terminator.
```

`struct json_iter` carries a small fixed frame stack — it never
allocates. There's no `json_iter_free`; the iterator dies with the
storage you put it in.

`QUERY_ITER` can appear multiple times for nested iteration. The cap is
`JSON_ITER_MAX_FRAMES` (4) simultaneously open iterations along a
single path; exceeding it returns `JSON_ERR_OOM`.

## Filtering with `QUERY_WHERE`

Filter array elements by a sub-query equality. The canonical case is
"find the codec named X":

```c
// codecs[] | select(.name == "sharding_indexed")
static const struct json_query sub[] = {
  { .kind = QUERY_KEY, .key = "name" },
};
static const struct json_query path[] = {
  { .kind = QUERY_KEY,  .key = "codecs" },
  { .kind = QUERY_ITER },
  { .kind = QUERY_WHERE,
    .where = { .part     = sub,
               .n        = countof(sub),
               .rhs      = "sharding_indexed",
               .rhs_type = JSON_STRING } },
};
```

`QUERY_WHERE` is *pass-through*: when the sub-query against the current node
matches the literal, the current node continues down the path; otherwise the
iterator advances to the next array element. `where.part` and `where.n` are
the same `(parts, n)` shape as the public API — one carries the sub-query
array, the other its length. `rhs_type` must be `JSON_STRING`, `JSON_NUMBER`,
or `JSON_BOOL` — those are the literal types the comparator understands.
Strings compare unescaped; numbers compare numerically; bools compare by
literal.

## Reading typed values

Given a `json_node`, the converters do the typed read:

```c
int64_t  i;  uint64_t u;  double d;  int b;
json_as_int   (n, &i);  // JSON_NUMBER, not a float literal
json_as_uint  (n, &u);  // JSON_NUMBER, non-negative, fits 64 bits
json_as_double(n, &d);  // JSON_NUMBER, any
json_as_bool  (n, &b);  // JSON_BOOL, b set to 0 or 1
```

Each returns `JSON_OK` on success, `JSON_ERR_TYPE` if the node isn't the
expected type, `JSON_ERR_RANGE` if the literal won't fit (e.g., `json_as_int` on
`1.5e2`, or `json_as_uint` on `-3`).

For string equality, `json_str_eq` does a byte-exact compare of a `JSON_STRING`
node against a NUL-terminated literal:

```c
if (json_str_eq(n, "sharding_indexed")) { ... }
```

It returns `0` whenever the node has `JSON_NODE_FLAG_STR_ESCAPED` — escaped
strings never match, even if their decoded form would. See the next section.

## Strings and escapes

A string node's match (the `s` field) spans the bytes *between* the quotes. The
`flag` tells you whether the source contained any backslash escapes:

```c
// "hello"   →  s is 5 bytes ("hello"), flag == JSON_NODE_FLAG_NONE
// "a\"b"    →  s is 4 bytes (a, \, ", b), flag == JSON_NODE_FLAG_STR_ESCAPED
```

For escape-free strings, `s.beg/s.end` is the value you want directly — no
decode required. For escaped strings, the raw bytes still include the backslash
sequences; you must decode them yourself if you need the logical content.
`json_str_eq` is the fast path for the common escape-free case and refuses
(returns 0) on escaped nodes by design.

This same byte-exact rule applies to `QUERY_KEY` lookup: an object key spelled
with escapes won't match a `.key` written without them.

## Error handling

Every public function returns an `enum json_err`:

| Code              | Meaning                                                    |
|-------------------|------------------------------------------------------------|
| `JSON_OK`         | success                                                    |
| `JSON_ERR_INVALID`| NULL pointer or malformed call                             |
| `JSON_ERR_PARSE`  | input bytes are malformed JSON                             |
| `JSON_ERR_TYPE`   | node isn't the expected type for this operation            |
| `JSON_ERR_RANGE`  | numeric literal won't fit the requested type               |
| `JSON_ERR_NOT_FOUND` | object key missing or iteration exhausted               |
| `JSON_ERR_OOM`    | iter frame stack would overflow (>4 nested `QUERY_ITER`)   |

For parse errors, optionally pass a `struct json_error*` to
`json_resolve` / `json_iter_init` to learn the byte offset where the
lexer gave up:

```c
struct json_error err = { JSON_OK, 0 };
if (json_resolve(src, parts, countof(parts), &n, &err) == JSON_ERR_PARSE) {
  fprintf(stderr, "parse error at offset %zu\n", err.offset);
}
```

`JSON_ERR_NOT_FOUND` is normal control flow — it's how you detect
"key missing" or "iteration exhausted." `JSON_ERR_PARSE` indicates a
malformed input.

## Potential pitfalls

- **Lifetimes.** Every node points into `src`. If `src` is freed, your
  nodes dangle. The reader does no copying — that's the point.

- **Lazy parsing means errors come late.** A malformed value buried in
  a subtree you don't query is silently ignored. If you need full
  validation, this library might not be right for you.

- **String matching is byte-exact.** Both `QUERY_KEY` lookup and
  `json_str_eq` compare raw bytes. Escaped strings won't match against
  decoded literals.

For larger worked examples — including `QUERY_WHERE` against zarr
metadata — see `tests/test_json.c`.
