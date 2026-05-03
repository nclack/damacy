// Query fuzzer for util/json.
//
// The fuzz input is consumed as a tape: the first bytes encode a
// json_query (segments, keys, indices, optionally a QUERY_WHERE with a
// sub-query). The remaining bytes are evaluated against a fixed JSON
// corpus, not the fuzz input itself — this fuzzer stresses the query
// evaluator (segment dispatch, iter-frame stack, QUERY_WHERE recursion)
// rather than the lexer.
//
// Differential oracle: when the decoded query contains no QUERY_ITER
// anywhere (including inside QUERY_WHERE.path), the result of
// json_resolve must match the first emission of json_iter_next on the
// same query. Both code paths exist; neither references the other; a
// disagreement is a real bug. Multi-valued queries skip the oracle —
// json_resolve is documented to return the first emission, but verifying
// that for QUERY_ITER paths re-implements the iterator inside the harness
// and isn't worth the complexity.
//
// Note on QUERY_WHERE: a QUERY_WHERE.path with QUERY_ITER inside does NOT
// make the outer query multi-valued — where_holds() evaluates the
// sub-path through a fresh json_resolve (json.c:387), so its emissions
// never reach the parent. We skip the oracle in that case anyway,
// conservatively, because reasoning about the sub-path's own
// backtracking under both top-level paths simultaneously is more work
// than it's worth. Tightening to "outer query has no QUERY_ITER" would
// expand oracle coverage but needs more analysis of where_holds.
//
// Bounds (deliberately tight, both for fuzzer throughput and to stay
// inside library limits):
//   MAX_SEGS_PER_LEVEL = 6   one above JSON_ITER_MAX_FRAMES so the
//                            evaluator's stack-overflow check (json.c
//                            line 511, JSON_ERR_OOM) is reachable
//   MAX_DEPTH          = 2   sub-queries inside QUERY_WHERE.path
//   MAX_STR            = 8   key / rhs literal length
#include "util/json.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum
{
  MAX_SEGS_PER_LEVEL = 6,
  MAX_DEPTH = 2,
  MAX_STR = 8,
  ARENA_CAP = 4096,
};

struct tape
{
  const uint8_t* p;
  const uint8_t* end;
};

static uint8_t
tape_u8(struct tape* t)
{
  if (t->p >= t->end)
    return 0;
  return *t->p++;
}

struct arena
{
  uint8_t buf[ARENA_CAP];
  size_t off;
};

static void*
arena_alloc(struct arena* a, size_t n, size_t align)
{
  size_t aligned = (a->off + align - 1) & ~(align - 1);
  if (aligned + n > ARENA_CAP)
    return NULL;
  a->off = aligned + n;
  return a->buf + aligned;
}

// Pull up to MAX_STR bytes off the tape, copy into the arena, NUL-
// terminate. Returns NULL on arena exhaustion (decoder bails).
static const char*
decode_str(struct tape* t, struct arena* a)
{
  uint8_t len = tape_u8(t) % (MAX_STR + 1);
  char* s = arena_alloc(a, (size_t)len + 1, 1);
  if (!s)
    return NULL;
  for (uint8_t i = 0; i < len; ++i)
    s[i] = (char)tape_u8(t);
  s[len] = '\0';
  return s;
}

// Forward decl — QUERY_WHERE recurses through this.
static struct json_query*
decode_parts(struct tape* t,
             struct arena* a,
             int depth,
             size_t* n_out,
             int* ok);

static int
decode_part(struct json_query* out, struct tape* t, struct arena* a, int depth)
{
  // Map the kind byte through an explicit table (do NOT rely on enum
  // numeric ordering — the order in json.h is incidental). At MAX_DEPTH
  // we drop QUERY_WHERE so recursion terminates. Modulo bias is fine; the
  // fuzzer doesn't need a uniform distribution.
  static const enum json_query_kind kind_menu[] = {
    QUERY_KEY, QUERY_INDEX, QUERY_ITER, QUERY_WHERE
  };
  uint8_t k = tape_u8(t);
  size_t menu_n = (depth >= MAX_DEPTH) ? 3 : 4;
  enum json_query_kind kind = kind_menu[k % menu_n];

  out->kind = kind;
  switch (kind) {
    case QUERY_KEY: {
      const char* s = decode_str(t, a);
      if (!s)
        return 0;
      out->key = s;
      return 1;
    }
    case QUERY_INDEX:
      // Keep small — large indices waste cycles on pure miss paths.
      out->index = (size_t)(tape_u8(t) & 0x0f);
      return 1;
    case QUERY_ITER:
      return 1;
    case QUERY_WHERE: {
      int sub_ok = 0;
      size_t sub_n = 0;
      struct json_query* sub = decode_parts(t, a, depth + 1, &sub_n, &sub_ok);
      if (!sub_ok)
        return 0;
      const char* rhs = decode_str(t, a);
      if (!rhs)
        return 0;
      out->where.part = sub;
      out->where.n = sub_n;
      out->where.rhs = rhs;
      // Map the rhs_type byte onto STRING/NUMBER/BOOL — the only types
      // the evaluator accepts on the rhs side.
      static const enum json_type rhs_types[] = { JSON_STRING,
                                                  JSON_NUMBER,
                                                  JSON_BOOL };
      out->where.rhs_type = rhs_types[tape_u8(t) % 3];
      return 1;
    }
  }
  // Unreachable given the modulo above; keep the compiler happy.
  return 0;
}

static struct json_query*
decode_parts(struct tape* t, struct arena* a, int depth, size_t* n_out, int* ok)
{
  uint8_t n = tape_u8(t) % (MAX_SEGS_PER_LEVEL + 1);
  *n_out = n;
  if (n == 0) {
    *ok = 1;
    return NULL;
  }
  struct json_query* parts = arena_alloc(
    a, sizeof(struct json_query) * (size_t)n, _Alignof(struct json_query));
  if (!parts) {
    *ok = 0;
    return NULL;
  }
  for (uint8_t i = 0; i < n; ++i) {
    if (!decode_part(&parts[i], t, a, depth)) {
      *ok = 0;
      return NULL;
    }
  }
  *ok = 1;
  return parts;
}

// Walk the part tree looking for any QUERY_ITER. If none exist, the
// query is single-valued and json_resolve must match the first
// json_iter_next emission.
static int
contains_iter(const struct json_query* parts, size_t n)
{
  for (size_t i = 0; i < n; ++i) {
    const struct json_query* p = &parts[i];
    if (p->kind == QUERY_ITER)
      return 1;
    if (p->kind == QUERY_WHERE && p->where.part &&
        contains_iter(p->where.part, p->where.n))
      return 1;
  }
  return 0;
}

// Fixed JSON corpus to evaluate fuzzed queries against. Pulled from
// real shapes the parser sees in production (zarr v3 metadata) plus a
// couple of degenerate cases. Each is the subject of every fuzzed query
// — running N corpora multiplies the effective test count per fuzz
// iteration but stays cheap because all are small.
// clang-format off
static const char CORPUS_zarr[] = "{\"codecs\":[{\"name\":\"bytes\"},{\"name\":\"sharding_indexed\",\"id\":42}],\"shape\":[1024,1024,128],\"chunks\":[64,64,16]}";
// clang-format on
static const char* const CORPUS[] = {
  CORPUS_zarr,
  "[1,2,3,4,5]",
  "{\"a\":{\"b\":{\"c\":{\"d\":1}}}}",
  "{\"name\":\"x\",\"name\":\"y\"}",
  "[]",
  "{}",
  "null",
  "\"\"",
  "{\"a\":[true,false,null,1.5e2,-3,\"hi\"]}",
  // 6 levels of array nesting — gives the evaluator somewhere to push
  // 5+ frames so the JSON_ITER_MAX_FRAMES stack-overflow check fires
  // when the fuzzer happens to pick 5+ QUERY_ITER segments in a row.
  "[[[[[[1]]]]]]",
};

static int
node_eq(struct json_node a, struct json_node b)
{
  if (a.type != b.type || a.flag != b.flag)
    return 0;
  size_t la = (size_t)(a.s.end - a.s.beg);
  size_t lb = (size_t)(b.s.end - b.s.beg);
  if (la != lb)
    return 0;
  return la == 0 || memcmp(a.s.beg, b.s.beg, la) == 0;
}

int
LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
  struct tape t = { .p = data, .end = data + size };
  struct arena a = { .off = 0 };

  int ok = 0;
  size_t n_parts = 0;
  struct json_query* parts = decode_parts(&t, &a, 0, &n_parts, &ok);
  if (!ok)
    return 0;
  const int single_valued = !contains_iter(parts, n_parts);

  // Thread real err pointers through so the parser's *err = ...
  // branches are exercised. Zero-init so MSan doesn't flag reads of
  // uninitialized memory on success paths where the parser doesn't
  // write *err. We don't compare err contents — exact offsets are an
  // internal detail — but ASan/UBSan see the writes.
  struct json_error rerr = { JSON_OK, 0 };
  struct json_error ierr = { JSON_OK, 0 };

  for (size_t ci = 0; ci < sizeof(CORPUS) / sizeof(CORPUS[0]); ++ci) {
    const char* s = CORPUS[ci];
    const struct cslice src = { .beg = s, .end = s + strlen(s) };

    struct json_node rn;
    enum json_err rrc = json_resolve(src, parts, n_parts, &rn, &rerr);

    struct json_iter it;
    enum json_err irc = json_iter_init(src, parts, n_parts, &it, &ierr);
    struct json_node in;
    enum json_err nrc = JSON_ERR_NOT_FOUND;
    if (irc == JSON_OK)
      nrc = json_iter_next(&it, &in);
    // Effective return: init failure, else next() result.
    enum json_err iter_rc = (irc == JSON_OK) ? nrc : irc;

    if (single_valued) {
      // Oracle: single-valued queries must agree byte-for-byte between
      // the eager and lazy paths. Trip the address sanitizer if not.
      if (rrc != iter_rc) {
        abort();
      }
      if (rrc == JSON_OK && !node_eq(rn, in)) {
        abort();
      }
    }
  }
  return 0;
}
