// Zero-allocation, lazy JSON reader.
//
// All operations work directly on a cslice that points at the JSON
// source. Nothing is parsed up front; the bytes belonging to subtrees
// the caller never asks for are never lexed.
//
// A query is an array of json_query parts plus its length (`n_parts`).
// Callers typically derive n_parts from a static array via their own
// `sizeof(arr)/sizeof((arr)[0])` macro (commonly named `countof`).
// The same query drives both:
//   json_resolve   — return the first surviving result.
//   json_iter_init — open a stream of results (next + next + ... until
//                    JSON_ERR_NOT_FOUND).
//
// An empty query (parts=NULL, n_parts=0) returns the root value.
//
// Lifetime: every returned json_node holds a cslice into the original
// src buffer; src must outlive any node derived from it.
#pragma once

#include "util/slice.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum json_type
  {
    JSON_INVALID = 0,
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT,
  };

  enum json_err
  {
    JSON_OK = 0,
    JSON_ERR_INVALID,   // null/bad arguments
    JSON_ERR_PARSE,     // src is malformed JSON
    JSON_ERR_TYPE,      // node is not the expected JSON type
    JSON_ERR_RANGE,     // numeric literal does not fit the requested type
    JSON_ERR_NOT_FOUND, // object key missing or iteration exhausted
    JSON_ERR_OOM,       // would-be allocation refused (e.g., iter stack)
  };

  // Per-node tag. At most one applies, so it lives as a single value, not
  // a bitmask: STR_ESCAPED only on strings, NUM_FLOAT only on numbers.
  enum json_node_flag
  {
    JSON_NODE_FLAG_NONE = 0,
    JSON_NODE_FLAG_STR_ESCAPED,
    JSON_NODE_FLAG_NUM_FLOAT,
  };

  struct json_node
  {
    enum json_type type;
    enum json_node_flag flag;
    struct cslice s; // span into src (between quotes for strings)
  };

  // Position info for parse failures. offset is the byte offset into the
  // src cslice where the lexer gave up. Filled in by json_resolve and
  // json_iter_init when the caller passes a non-NULL err.
  struct json_error
  {
    enum json_err code;
    size_t offset;
  };

  // ---- Query language --------------------------------------------------

  enum json_query_kind
  {
    QUERY_KEY = 0, // .key                — single-valued
    QUERY_INDEX,   // .[i]                — single-valued
    QUERY_ITER,    // .[]                 — yields each element
    QUERY_WHERE,   // select(part == rhs) — pass-through if predicate holds
  };

  struct json_query
  {
    enum json_query_kind kind;
    union
    {
      const char* key; // QUERY_KEY
      size_t index;    // QUERY_INDEX
      struct
      {
        const struct json_query* part; // sub-query against current value
        size_t n;                      // length of `part` array
        const char* rhs;               // NUL-terminated literal to compare
        enum json_type rhs_type;       // STRING, NUMBER, BOOL
      } where;
    };
  };

  // ---- Resolution ------------------------------------------------------

  // Evaluate the query against src and return the first surviving
  // result. Multi-valued queries (containing QUERY_ITER) simply return
  // their first emitted value. `n_parts == 0` returns the root.
  enum json_err json_resolve(struct cslice src,
                             const struct json_query* parts,
                             size_t n_parts,
                             struct json_node* out,
                             struct json_error* err);

  // Iterator state. Treat as opaque; only the size matters at the call
  // site. The frame stack is bounded so iteration never allocates.
  enum
  {
    JSON_ITER_MAX_FRAMES = 4
  };

  struct json_iter_frame
  {
    size_t part_index;  // index of the QUERY_ITER part that opened this frame
    struct cslice rest; // remaining bytes inside the open array
    uint8_t saw_first;  // have we emitted the first element yet?
  };

  struct json_iter
  {
    struct cslice src;
    const struct json_query* parts;
    size_t n_parts;
    struct json_error last_err;
    uint8_t depth;
    uint8_t done;
    struct json_iter_frame frames[JSON_ITER_MAX_FRAMES];
  };

  // Open an iteration over the query's results. err is optional.
  enum json_err json_iter_init(struct cslice src,
                               const struct json_query* parts,
                               size_t n_parts,
                               struct json_iter* it,
                               struct json_error* err);

  // Produce the next surviving result. Returns JSON_OK on a hit,
  // JSON_ERR_NOT_FOUND when the stream is exhausted, or a parse error
  // code if src is malformed past the previous emission.
  enum json_err json_iter_next(struct json_iter* it, struct json_node* out);

  // ---- Primitive conversions ------------------------------------------
  // JSON_OK on success, JSON_ERR_TYPE on wrong node type, JSON_ERR_RANGE
  // when the literal does not parse cleanly into the requested form.
  enum json_err json_as_int(struct json_node n, int64_t* out);
  enum json_err json_as_uint(struct json_node n, uint64_t* out);
  enum json_err json_as_double(struct json_node n, double* out);
  enum json_err json_as_bool(struct json_node n, int* out);

  // Byte-exact compare of a JSON_STRING node against a NUL-terminated
  // literal. Returns 1 on match, 0 otherwise. Strings containing escape
  // sequences never match: their on-disk bytes include the backslash
  // sequences, so callers wanting unescaped equality must do their own
  // decode.
  int json_str_eq(struct json_node n, const char* lit);

#ifdef __cplusplus
}
#endif
