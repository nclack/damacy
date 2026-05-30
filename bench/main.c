// damacy_bench — drives the public damacy API end-to-end against a
// scenario.json. Reads the scenario from argv[1], writes results JSON to
// stdout, sends diagnostics to stderr.
//
//   damacy_bench scenario.json > results.json
//
// All path/timestamp orchestration belongs in bench/run.py; this binary
// is the timing core only.
#include "damacy.h"

#include "util/json.h"
#include "util/json_writer.h"
#include "util/slice.h"
#include "util/strbuf.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define countof(a) (sizeof(a) / sizeof((a)[0]))

// ---- tiny prelude -----------------------------------------------------------

static double
now_seconds(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int
slurp_file(const char* path, char** out, size_t* out_len)
{
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "bench: cannot open %s: %s\n", path, strerror(errno));
    return 1;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return 1;
  }
  long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return 1;
  }
  rewind(f);
  char* buf = (char*)malloc((size_t)sz + 1);
  if (!buf) {
    fclose(f);
    return 1;
  }
  size_t got = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  if (got != (size_t)sz) {
    free(buf);
    return 1;
  }
  buf[sz] = '\0';
  *out = buf;
  *out_len = (size_t)sz;
  return 0;
}

// xorshift64* — small, deterministic; fine for "pick AABBs and zarr ids".
struct rng
{
  uint64_t s;
};
static uint64_t
rng_next(struct rng* r)
{
  uint64_t x = r->s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  r->s = x;
  return x * 2685821657736338717ull;
}
static uint64_t
rng_range(struct rng* r, uint64_t n)
{
  return n ? rng_next(r) % n : 0;
}

// ---- scenario ---------------------------------------------------------------

struct scenario
{
  // dataset
  char store_root[256];
  uint32_t n_zarrs;
  char uri_fmt[128];
  uint8_t rank;
  int64_t zarr_shape[DAMACY_MAX_RANK];
  enum damacy_dtype dtype;

  // sampling
  int64_t sample_shape[DAMACY_MAX_RANK];
  uint32_t n_batches;
  uint32_t n_warmup_batches;
  uint32_t samples_per_batch;
  uint64_t sampling_seed;

  // pipeline
  uint32_t lookahead_samples;
  uint32_t n_io_threads;
  uint32_t n_prefetch_io_threads;
  uint64_t max_gpu_memory_bytes;
  uint32_t max_chunk_uncompressed_bytes;
  uint64_t max_read_op_bytes;
  uint32_t n_array_meta_cache;
  uint32_t n_shard_index_cache;
  uint32_t n_chunk_layout_cache;
  uint8_t host_buffer_waves;
  uint8_t bypass_decode;

  double consumer_hold_ms;

  // raw JSON src for echo (kept alive by main)
  struct cslice src;
};

static int
read_string_into(struct cslice src,
                 const struct json_query* parts,
                 size_t n_parts,
                 char* out,
                 size_t out_cap)
{
  struct json_node n;
  if (json_resolve(src, parts, n_parts, &n, NULL) || n.type != JSON_STRING)
    return 1;
  size_t len = cslice_len(n.s);
  if (len + 1 > out_cap)
    return 1;
  memcpy(out, n.s.beg, len);
  out[len] = '\0';
  return 0;
}

static int
read_uint(struct cslice src,
          const struct json_query* parts,
          size_t n_parts,
          uint64_t* out)
{
  struct json_node n;
  if (json_resolve(src, parts, n_parts, &n, NULL))
    return 1;
  return json_as_uint(n, out);
}

// Optional uint with a default.
static void
read_uint_opt(struct cslice src,
              const struct json_query* parts,
              size_t n_parts,
              uint64_t* out,
              uint64_t fallback)
{
  if (read_uint(src, parts, n_parts, out))
    *out = fallback;
}

// Optional bool with a default. Accepts `true`/`false` or 0/1.
static void
read_bool_opt(struct cslice src,
              const struct json_query* parts,
              size_t n_parts,
              int* out,
              int fallback)
{
  struct json_node n;
  if (json_resolve(src, parts, n_parts, &n, NULL)) {
    *out = fallback;
    return;
  }
  int b = 0;
  if (json_as_bool(n, &b) == JSON_OK) {
    *out = b;
    return;
  }
  uint64_t v = 0;
  if (json_as_uint(n, &v) == JSON_OK) {
    *out = v ? 1 : 0;
    return;
  }
  *out = fallback;
}

static void
read_double_opt(struct cslice src,
                const struct json_query* parts,
                size_t n_parts,
                double* out,
                double fallback)
{
  struct json_node n;
  if (json_resolve(src, parts, n_parts, &n, NULL) ||
      json_as_double(n, out) != JSON_OK) {
    *out = fallback;
  }
}

static void
sleep_ms(double ms)
{
  if (ms <= 0.0)
    return;
  long sec = (long)(ms / 1e3);
  long nsec = (long)((ms - (double)sec * 1e3) * 1e6);
  struct timespec req = { .tv_sec = sec, .tv_nsec = nsec };
  struct timespec rem;
  while (nanosleep(&req, &rem) == -1 && errno == EINTR)
    req = rem;
}

// Read an array of int64 into `out[0..max)`. Sets *out_n.
static int
read_int_array(struct cslice src,
               const struct json_query* parts,
               size_t n_parts,
               int64_t* out,
               uint8_t max,
               uint8_t* out_n)
{
  struct json_node arr;
  if (json_resolve(src, parts, n_parts, &arr, NULL) || arr.type != JSON_ARRAY)
    return 1;
  static const struct json_query iter[] = { { .kind = QUERY_ITER } };
  struct json_iter it;
  if (json_iter_init(arr.s, iter, countof(iter), &it, NULL))
    return 1;
  uint8_t n = 0;
  for (;;) {
    struct json_node v;
    enum json_err e = json_iter_next(&it, &v);
    if (e == JSON_ERR_NOT_FOUND)
      break;
    if (e != JSON_OK || n >= max)
      return 1;
    int64_t iv;
    if (json_as_int(v, &iv))
      return 1;
    out[n++] = iv;
  }
  *out_n = n;
  return 0;
}

static int
parse_dtype(struct cslice src,
            const struct json_query* parts,
            size_t n_parts,
            enum damacy_dtype* out)
{
  struct json_node n;
  if (json_resolve(src, parts, n_parts, &n, NULL) || n.type != JSON_STRING)
    return 1;
  if (json_str_eq(n, "f32")) {
    *out = DAMACY_F32;
    return 0;
  }
  if (json_str_eq(n, "bf16")) {
    *out = DAMACY_BF16;
    return 0;
  }
  return 1;
}

// Bytes per element for the dtype enum (mirrors damacy_dtype_bpe).
static uint64_t
dtype_bpe(enum damacy_dtype d)
{
  switch (d) {
    case DAMACY_BF16:
      return 2;
    case DAMACY_F32:
      return 4;
  }
  return 0;
}

static int
parse_scenario(struct cslice src, struct scenario* sc)
{
  memset(sc, 0, sizeof(*sc));
  sc->src = src;
  sc->n_array_meta_cache = 4096;
  sc->n_shard_index_cache = 16384;
  sc->n_chunk_layout_cache = 4096;

  // dataset
  {
    static const struct json_query p[] = { { QUERY_KEY, .key = "dataset" },
                                           { QUERY_KEY, .key = "store_root" } };
    if (read_string_into(
          src, p, countof(p), sc->store_root, sizeof(sc->store_root)))
      return 1;
  }
  {
    static const struct json_query p[] = { { QUERY_KEY, .key = "dataset" },
                                           { QUERY_KEY, .key = "n_zarrs" } };
    uint64_t v;
    if (read_uint(src, p, countof(p), &v) || v == 0 || v > UINT32_MAX)
      return 1;
    sc->n_zarrs = (uint32_t)v;
  }
  {
    static const struct json_query p[] = { { QUERY_KEY, .key = "dataset" },
                                           { QUERY_KEY, .key = "uri_fmt" } };
    if (read_string_into(src, p, countof(p), sc->uri_fmt, sizeof(sc->uri_fmt)))
      return 1;
  }
  {
    static const struct json_query p[] = { { QUERY_KEY, .key = "dataset" },
                                           { QUERY_KEY, .key = "zarr_shape" } };
    if (read_int_array(
          src, p, countof(p), sc->zarr_shape, DAMACY_MAX_RANK, &sc->rank))
      return 1;
  }
  {
    // Destination dtype lives on `pipeline.dtype` now; per-zarr source
    // dtypes are written by gen_dataset.py and the loader picks them up
    // from each zarr's metadata.
    static const struct json_query p[] = { { QUERY_KEY, .key = "pipeline" },
                                           { QUERY_KEY, .key = "dtype" } };
    if (parse_dtype(src, p, countof(p), &sc->dtype))
      return 1;
  }

  // sampling
  {
    static const struct json_query p[] = {
      { QUERY_KEY, .key = "sampling" }, { QUERY_KEY, .key = "sample_shape" }
    };
    uint8_t n = 0;
    if (read_int_array(
          src, p, countof(p), sc->sample_shape, DAMACY_MAX_RANK, &n))
      return 1;
    if (n != sc->rank)
      return 1;
  }
  {
    uint64_t v;
    static const struct json_query p_b[] = {
      { QUERY_KEY, .key = "sampling" }, { QUERY_KEY, .key = "n_batches" }
    };
    static const struct json_query p_w[] = {
      { QUERY_KEY, .key = "sampling" }, { QUERY_KEY, .key = "n_warmup_batches" }
    };
    static const struct json_query p_bs[] = { { QUERY_KEY, .key = "sampling" },
                                              { QUERY_KEY,
                                                .key = "samples_per_batch" } };
    static const struct json_query p_s[] = { { QUERY_KEY, .key = "sampling" },
                                             { QUERY_KEY, .key = "seed" } };
    if (read_uint(src, p_b, countof(p_b), &v))
      return 1;
    sc->n_batches = (uint32_t)v;
    read_uint_opt(src, p_w, countof(p_w), &v, 0);
    sc->n_warmup_batches = (uint32_t)v;
    if (read_uint(src, p_bs, countof(p_bs), &v))
      return 1;
    sc->samples_per_batch = (uint32_t)v;
    read_uint_opt(src, p_s, countof(p_s), &sc->sampling_seed, 1234);
  }

  // pipeline
  {
    uint64_t v;
    static const struct json_query p_la[] = { { QUERY_KEY, .key = "pipeline" },
                                              { QUERY_KEY,
                                                .key = "lookahead_samples" } };
    static const struct json_query p_io[] = {
      { QUERY_KEY, .key = "pipeline" }, { QUERY_KEY, .key = "n_io_threads" }
    };
    static const struct json_query p_pio[] = {
      { QUERY_KEY, .key = "pipeline" },
      { QUERY_KEY, .key = "n_prefetch_io_threads" }
    };
    static const struct json_query p_g[] = { { QUERY_KEY, .key = "pipeline" },
                                             { QUERY_KEY,
                                               .key = "max_gpu_memory_mb" } };
    static const struct json_query p_c[] = {
      { QUERY_KEY, .key = "pipeline" },
      { QUERY_KEY, .key = "max_chunk_uncompressed_mb" }
    };
    static const struct json_query p_zm[] = { { QUERY_KEY, .key = "pipeline" },
                                              { QUERY_KEY,
                                                .key = "n_array_meta_cache" } };
    static const struct json_query p_sm[] = {
      { QUERY_KEY, .key = "pipeline" },
      { QUERY_KEY, .key = "n_shard_index_cache" }
    };
    static const struct json_query p_cl[] = {
      { QUERY_KEY, .key = "pipeline" },
      { QUERY_KEY, .key = "n_chunk_layout_cache" }
    };
    if (read_uint(src, p_la, countof(p_la), &v))
      return 1;
    sc->lookahead_samples = (uint32_t)v;
    if (read_uint(src, p_io, countof(p_io), &v))
      return 1;
    sc->n_io_threads = (uint32_t)v;
    read_uint_opt(
      src, p_pio, countof(p_pio), &v, DAMACY_DEFAULT_PREFETCH_IO_THREADS);
    sc->n_prefetch_io_threads = (uint32_t)v;
    read_uint_opt(src, p_g, countof(p_g), &v, 0);
    sc->max_gpu_memory_bytes = v << 20;
    read_uint_opt(src, p_c, countof(p_c), &v, 0);
    sc->max_chunk_uncompressed_bytes =
      v ? (uint32_t)(v << 20) : DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES;
    static const struct json_query p_ro[] = {
      { QUERY_KEY, .key = "pipeline" }, { QUERY_KEY, .key = "max_read_op_kb" }
    };
    read_uint_opt(src, p_ro, countof(p_ro), &v, 0);
    sc->max_read_op_bytes = v ? v << 10 : DAMACY_DEFAULT_READ_OP_MAX_BYTES;
    read_uint_opt(src, p_zm, countof(p_zm), &v, 4096);
    sc->n_array_meta_cache = (uint32_t)v;
    read_uint_opt(src, p_sm, countof(p_sm), &v, 16384);
    sc->n_shard_index_cache = (uint32_t)v;
    read_uint_opt(src, p_cl, countof(p_cl), &v, 4096);
    sc->n_chunk_layout_cache = (uint32_t)v;
    static const struct json_query p_hw[] = { { QUERY_KEY, .key = "pipeline" },
                                              { QUERY_KEY,
                                                .key = "host_buffer_waves" } };
    read_uint_opt(
      src, p_hw, countof(p_hw), &v, DAMACY_DEFAULT_HOST_BUFFER_WAVES);
    sc->host_buffer_waves = (uint8_t)v;
    static const struct json_query p_bd[] = {
      { QUERY_KEY, .key = "pipeline" }, { QUERY_KEY, .key = "bypass_decode" }
    };
    int bd = 0;
    read_bool_opt(src, p_bd, countof(p_bd), &bd, 0);
    sc->bypass_decode = (uint8_t)bd;
  }

  {
    static const struct json_query p_h[] = { { QUERY_KEY, .key = "consumer" },
                                             { QUERY_KEY, .key = "hold_ms" } };
    read_double_opt(src, p_h, countof(p_h), &sc->consumer_hold_ms, 0.0);
    if (sc->consumer_hold_ms < 0.0)
      sc->consumer_hold_ms = 0.0;
  }

  // sanity: all axes can fit a sample
  if (sc->lookahead_samples < sc->samples_per_batch) {
    fprintf(stderr,
            "scenario: lookahead_samples=%u must be at least "
            "samples_per_batch=%u\n",
            sc->lookahead_samples,
            sc->samples_per_batch);
    return 1;
  }
  for (uint8_t d = 0; d < sc->rank; ++d) {
    if (sc->sample_shape[d] > sc->zarr_shape[d]) {
      fprintf(stderr,
              "scenario: sample_shape[%u]=%lld > zarr_shape[%u]=%lld\n",
              d,
              (long long)sc->sample_shape[d],
              d,
              (long long)sc->zarr_shape[d]);
      return 1;
    }
  }
  return 0;
}

// ---- sample generation ------------------------------------------------------

// One owned URI string per zarr, fixed-size for the bench harness.
#define BENCH_MAX_URI 256
struct uri_table
{
  char* mem; // n_zarrs * BENCH_MAX_URI
  uint32_t n;
};

static int
uri_table_init(struct uri_table* t, const char* fmt, uint32_t n_zarrs)
{
  t->mem = (char*)calloc((size_t)n_zarrs, BENCH_MAX_URI);
  t->n = n_zarrs;
  if (!t->mem)
    return 1;
  for (uint32_t i = 0; i < n_zarrs; ++i) {
    int w = snprintf(&t->mem[i * BENCH_MAX_URI], BENCH_MAX_URI, fmt, i);
    if (w < 0 || w >= BENCH_MAX_URI)
      return 1;
  }
  return 0;
}
static void
uri_table_free(struct uri_table* t)
{
  if (!t)
    return;
  free(t->mem);
  t->mem = NULL;
}
static const char*
uri_table_get(const struct uri_table* t, uint32_t i)
{
  return &t->mem[(i % t->n) * BENCH_MAX_URI];
}

static void
fill_random_sample(const struct scenario* sc,
                   const struct uri_table* uris,
                   struct rng* rng,
                   struct damacy_sample* s)
{
  uint32_t z = (uint32_t)rng_range(rng, sc->n_zarrs);
  s->uri = uri_table_get(uris, z);
  s->aabb.rank = sc->rank;
  for (uint8_t d = 0; d < sc->rank; ++d) {
    int64_t span = sc->zarr_shape[d] - sc->sample_shape[d] + 1;
    int64_t beg = (int64_t)rng_range(rng, (uint64_t)span);
    s->aabb.dims[d].beg = beg;
    s->aabb.dims[d].end = beg + sc->sample_shape[d];
  }
}

// ---- run --------------------------------------------------------------------

struct run_metrics
{
  double init_ms;
  double ttfb_ms; // first push -> first OK pop
  double wall_ms; // steady-state only (after warmup, after stats reset)
  double consumer_block_ms_total;
  double consumer_push_ms_total; // time inside damacy_push between release and
                                 // next OK pop
  double consumer_pop_wait_ms_total; // time inside the final damacy_pop that
                                     // returns OK
  uint64_t pushed;
  uint64_t popped;
  struct damacy_stats stats;
};

// Push exactly n_target_batches * samples_per_batch samples and pop
// n_target_batches batches. When hold_ms > 0, sleeps after a successful
// pop and before release to simulate the consumer holding the batch.
// block_out accumulates wait time between releasing batch i-1 (or drive
// entry) and popping batch i — the pipeline-stall signal. push_out and
// pop_wait_out split that wait: time inside damacy_push (cumulative
// across calls in the cycle) vs the final damacy_pop that returned OK.
static int
drive(struct damacy* d,
      const struct scenario* sc,
      const struct uri_table* uris,
      struct rng* rng,
      uint32_t n_target_batches,
      double hold_ms,
      uint64_t* pushed,
      uint64_t* popped,
      double* first_pop_t,
      double* block_out,
      double* push_out,
      double* pop_wait_out)
{
  const uint32_t pool_cap = sc->lookahead_samples;
  const uint64_t samples_target =
    (uint64_t)n_target_batches * (uint64_t)sc->samples_per_batch;
  struct damacy_sample* pool =
    (struct damacy_sample*)calloc(pool_cap, sizeof(*pool));
  if (!pool)
    return 1;

  uint32_t in_pool = 0;
  uint32_t cursor = 0;

  uint64_t pushed_local = 0;
  uint64_t popped_local = 0;
  double t_wait_start = now_seconds();
  double push_acc_ms = 0.0;
  while (popped_local < n_target_batches) {
    if (pushed_local < samples_target) {
      if (cursor == in_pool) {
        cursor = 0;
        uint64_t remaining = samples_target - pushed_local;
        in_pool = remaining < pool_cap ? (uint32_t)remaining : pool_cap;
        for (uint32_t i = 0; i < in_pool; ++i)
          fill_random_sample(sc, uris, rng, &pool[i]);
      }
      struct damacy_sample_slice slice = { .beg = pool + cursor,
                                           .end = pool + in_pool };
      double tpush0 = now_seconds();
      struct damacy_push_result pr = damacy_push(d, slice);
      push_acc_ms += (now_seconds() - tpush0) * 1e3;
      uint32_t consumed = (uint32_t)(pr.unconsumed.beg - slice.beg);
      cursor += consumed;
      pushed_local += consumed;
      if (pr.status != DAMACY_OK && pr.status != DAMACY_AGAIN) {
        fprintf(stderr, "damacy_push: %s\n", damacy_status_str(pr.status));
        free(pool);
        return 1;
      }
    }

    struct damacy_batch* b = NULL;
    double tpop0 = now_seconds();
    enum damacy_status ps = damacy_pop(d, &b);
    double tpop1 = now_seconds();
    if (ps == DAMACY_OK) {
      if (first_pop_t && popped_local == 0)
        *first_pop_t = tpop1;
      if (block_out)
        *block_out += (tpop1 - t_wait_start) * 1e3;
      if (push_out)
        *push_out += push_acc_ms;
      if (pop_wait_out)
        *pop_wait_out += (tpop1 - tpop0) * 1e3;
      ++popped_local;
      sleep_ms(hold_ms);
      damacy_release(d, b);
      t_wait_start = now_seconds();
      push_acc_ms = 0.0;
    } else if (ps != DAMACY_AGAIN) {
      fprintf(stderr, "damacy_pop: %s\n", damacy_status_str(ps));
      free(pool);
      return 1;
    }
  }

  *pushed += pushed_local;
  *popped += popped_local;
  free(pool);
  return 0;
}

// ---- results JSON -----------------------------------------------------------

static void
emit_metric(struct json_writer* jw,
            const struct damacy_metric* m,
            const char* unit)
{
  jw_object_begin(jw);
  jw_key(jw, "name");
  jw_string(jw, m->name ? m->name : "");
  jw_key(jw, "unit");
  jw_string(jw, unit);
  jw_key(jw, "count");
  jw_uint(jw, m->count);
  jw_key(jw, "ms_total");
  jw_float(jw, (double)m->ms);
  jw_key(jw, "ms_avg");
  jw_float(jw, m->count ? (double)m->ms / (double)m->count : 0.0);
  jw_key(jw, "ms_best");
  jw_float(jw, m->count ? (double)m->best_ms : 0.0);
  jw_key(jw, "input_bytes");
  jw_float(jw, m->input_bytes);
  jw_key(jw, "output_bytes");
  jw_float(jw, m->output_bytes);
  jw_object_end(jw);
}

static void
emit_results(const struct scenario* sc, const struct run_metrics* rm, FILE* out)
{
  struct strbuf sb = { 0 };
  struct json_writer jw;
  jw_init(&jw, &sb);

  jw_object_begin(&jw);

  // Echo scenario verbatim.
  jw_key(&jw, "scenario");
  // src is the raw JSON document; emit it as a pre-validated value.
  // We don't bother trimming whitespace.
  {
    // jw_raw expects a NUL-terminated cstring; src.beg is from a slurped
    // file we keep alive and NUL-terminated.
    jw_raw(&jw, sc->src.beg);
  }

  // Timings.
  jw_key(&jw, "timings_ms");
  jw_object_begin(&jw);
  jw_key(&jw, "init");
  jw_float(&jw, rm->init_ms);
  jw_key(&jw, "time_to_first_batch");
  jw_float(&jw, rm->ttfb_ms);
  jw_key(&jw, "wall");
  jw_float(&jw, rm->wall_ms);
  jw_key(&jw, "consumer_block");
  jw_float(&jw, rm->consumer_block_ms_total);
  jw_key(&jw, "consumer_push");
  jw_float(&jw, rm->consumer_push_ms_total);
  jw_key(&jw, "consumer_pop_wait");
  jw_float(&jw, rm->consumer_pop_wait_ms_total);
  jw_object_end(&jw);

  // Per-stage rows with a unit field.
  jw_key(&jw, "stages");
  jw_array_begin(&jw);
  emit_metric(&jw, &rm->stats.plan, "batch");
  emit_metric(&jw, &rm->stats.io, "wave");
  emit_metric(&jw, &rm->stats.h2d, "wave");
  emit_metric(&jw, &rm->stats.decode, "wave");
  emit_metric(&jw, &rm->stats.post_decode, "wave");
  emit_metric(&jw, &rm->stats.decode_gap, "wave");
  emit_metric(&jw, &rm->stats.assemble, "wave");
  emit_metric(&jw, &rm->stats.bind_wait, "wave");
  emit_metric(&jw, &rm->stats.pop_wait, "poll");
  emit_metric(&jw, &rm->stats.flush_wait, "call");
  jw_array_end(&jw);

  // Counters.
  jw_key(&jw, "counters");
  jw_object_begin(&jw);
  jw_key(&jw, "samples_pushed");
  jw_uint(&jw, rm->pushed);
  jw_key(&jw, "batches_emitted");
  jw_uint(&jw, rm->stats.batches_emitted);
  jw_key(&jw, "batches_truncated");
  jw_uint(&jw, rm->stats.batches_truncated);
  jw_key(&jw, "waves_emitted");
  jw_uint(&jw, rm->stats.waves_emitted);
  jw_key(&jw, "worker_steps");
  jw_uint(&jw, rm->stats.worker_steps);
  jw_key(&jw, "chunks_dispatched");
  jw_uint(&jw, rm->stats.chunks_dispatched);
  jw_key(&jw, "chunks_planned");
  jw_uint(&jw, rm->stats.chunks_planned);
  jw_key(&jw, "chunks_to_load");
  jw_uint(&jw, rm->stats.chunks_to_load);
  jw_key(&jw, "reads_issued");
  jw_uint(&jw, rm->stats.reads_issued);
  jw_key(&jw, "distinct_zarrs");
  jw_uint(&jw, rm->stats.array_meta.misses);
  jw_key(&jw, "distinct_shards");
  jw_uint(&jw, rm->stats.shard_index.misses);
  jw_key(&jw, "array_meta_hits");
  jw_uint(&jw, rm->stats.array_meta.hits);
  jw_key(&jw, "array_meta_misses");
  jw_uint(&jw, rm->stats.array_meta.misses);
  jw_key(&jw, "shard_index_hits");
  jw_uint(&jw, rm->stats.shard_index.hits);
  jw_key(&jw, "shard_index_misses");
  jw_uint(&jw, rm->stats.shard_index.misses);
  jw_key(&jw, "chunk_layout_hits");
  jw_uint(&jw, rm->stats.chunk_layout.hits);
  jw_key(&jw, "chunk_layout_misses");
  jw_uint(&jw, rm->stats.chunk_layout.misses);
  jw_key(&jw, "gpu_bytes_committed");
  jw_uint(&jw, rm->stats.gpu_bytes_committed);
  jw_object_end(&jw);

  // Derived numbers.
  uint64_t bpe = dtype_bpe(sc->dtype);
  uint64_t sample_volume = bpe;
  for (uint8_t d = 0; d < sc->rank; ++d)
    sample_volume *= (uint64_t)sc->sample_shape[d];
  double sample_bytes_total = (double)rm->stats.batches_emitted *
                              (double)sc->samples_per_batch *
                              (double)sample_volume;
  double wall_s = rm->wall_ms / 1e3;
  double throughput_mb_s =
    wall_s > 0.0 ? (sample_bytes_total / 1e6) / wall_s : 0.0;
  double sum_stage_ms = (double)rm->stats.plan.ms + (double)rm->stats.io.ms +
                        (double)rm->stats.h2d.ms + (double)rm->stats.decode.ms +
                        (double)rm->stats.post_decode.ms +
                        (double)rm->stats.assemble.ms;
  double stage_concurrency =
    rm->wall_ms > 0.0 ? sum_stage_ms / rm->wall_ms : 0.0;
  double chunks_per_batch =
    rm->stats.batches_emitted
      ? (double)rm->stats.chunks_dispatched / (double)rm->stats.batches_emitted
      : 0.0;
  double chunks_per_wave =
    rm->stats.waves_emitted
      ? (double)rm->stats.chunks_dispatched / (double)rm->stats.waves_emitted
      : 0.0;

  jw_key(&jw, "derived");
  jw_object_begin(&jw);
  jw_key(&jw, "bytes_per_sample");
  jw_uint(&jw, sample_volume);
  jw_key(&jw, "throughput_mb_s");
  jw_float(&jw, throughput_mb_s);
  jw_key(&jw, "stage_concurrency");
  jw_float(&jw, stage_concurrency);
  jw_key(&jw, "chunks_per_batch");
  jw_float(&jw, chunks_per_batch);
  jw_key(&jw, "chunks_per_wave");
  jw_float(&jw, chunks_per_wave);
  jw_object_end(&jw);

  jw_object_end(&jw);

  if (jw_error(&jw)) {
    fprintf(stderr, "bench: json writer error\n");
  } else {
    fwrite(strbuf_cstr(&sb), 1, strbuf_len(&sb), out);
    fputc('\n', out);
  }
  strbuf_free(&sb);
}

// ---- main -------------------------------------------------------------------

int
main(int argc, char** argv)
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s scenario.json\n", argv[0]);
    return 1;
  }
  char* json_buf = NULL;
  size_t json_len = 0;
  if (slurp_file(argv[1], &json_buf, &json_len) != 0)
    return 1;
  struct cslice src = { .beg = json_buf, .end = json_buf + json_len };

  struct scenario sc;
  if (parse_scenario(src, &sc) != 0) {
    fprintf(stderr, "bench: failed to parse %s\n", argv[1]);
    free(json_buf);
    return 1;
  }

  // damacy now wants absolute uris; prepend the scenario's store_root.
  char abs_fmt[BENCH_MAX_URI];
  if (snprintf(abs_fmt, sizeof abs_fmt, "%s/%s", sc.store_root, sc.uri_fmt) >=
      (int)sizeof abs_fmt) {
    fprintf(stderr, "bench: store_root + uri_fmt exceeds BENCH_MAX_URI\n");
    free(json_buf);
    return 1;
  }
  struct uri_table uris = { 0 };
  if (uri_table_init(&uris, abs_fmt, sc.n_zarrs) != 0) {
    fprintf(stderr, "bench: uri table alloc failed\n");
    free(json_buf);
    return 1;
  }

  fprintf(stderr,
          "scenario: store_root=%s n_zarrs=%u rank=%u batches=%u (warmup=%u)\n",
          sc.store_root,
          sc.n_zarrs,
          sc.rank,
          sc.n_batches,
          sc.n_warmup_batches);

  struct damacy_config cfg = {
    .samples_per_batch = sc.samples_per_batch,
    .lookahead_samples = sc.lookahead_samples,
    .dtype = sc.dtype,
    .sample_rank = sc.rank,
    .device = -1,
    .tuning = {
      .n_io_threads = sc.n_io_threads,
      .n_array_meta_cache = sc.n_array_meta_cache,
      .n_shard_index_cache = sc.n_shard_index_cache,
      .n_chunk_layout_cache = sc.n_chunk_layout_cache,
      .max_chunk_uncompressed_bytes = sc.max_chunk_uncompressed_bytes,
      .max_read_op_bytes = sc.max_read_op_bytes,
      .max_gpu_memory_bytes = sc.max_gpu_memory_bytes,
      .host_buffer_waves = sc.host_buffer_waves,
      .max_chunks_per_wave = DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE,
      .max_substreams_per_chunk = DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK,
      .n_prefetch_io_threads = sc.n_prefetch_io_threads,
    },
    .debug = { .bypass_decode = sc.bypass_decode },
  };
  for (uint8_t d = 0; d < sc.rank; ++d)
    cfg.sample_shape[d] = sc.sample_shape[d];

  struct rng rng = { .s = sc.sampling_seed ? sc.sampling_seed : 0xdeadbeefULL };

  // damacy_create requires a CUcontext current. Retain dev 0's primary.
  if (cuInit(0) != CUDA_SUCCESS) {
    fprintf(stderr, "cuInit failed\n");
    uri_table_free(&uris);
    free(json_buf);
    return 1;
  }
  CUdevice cu_dev = 0;
  CUcontext cu_ctx = NULL;
  if (cuDeviceGet(&cu_dev, 0) != CUDA_SUCCESS ||
      cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev) != CUDA_SUCCESS ||
      cuCtxSetCurrent(cu_ctx) != CUDA_SUCCESS) {
    fprintf(stderr, "primary ctx setup failed\n");
    uri_table_free(&uris);
    free(json_buf);
    return 1;
  }

  double t_init_a = now_seconds();
  struct damacy* d = NULL;
  enum damacy_status cs = damacy_create(&cfg, &d);
  double t_init_b = now_seconds();
  if (cs != DAMACY_OK) {
    fprintf(stderr, "damacy_create: %s\n", damacy_status_str(cs));
    uri_table_free(&uris);
    free(json_buf);
    return 1;
  }

  struct run_metrics rm = { 0 };
  rm.init_ms = (t_init_b - t_init_a) * 1e3;

  // ttfb covers cold caches: from very first push attempt to first OK pop.
  double t_first_push = now_seconds();
  double t_first_pop = 0.0;

  // Warmup: warms caches, codec init, kernel JIT. Drain in-flight then
  // reset stats so steady-state metrics are clean.
  if (sc.n_warmup_batches > 0) {
    if (drive(d,
              &sc,
              &uris,
              &rng,
              sc.n_warmup_batches,
              sc.consumer_hold_ms,
              &rm.pushed,
              &rm.popped,
              &t_first_pop,
              NULL,
              NULL,
              NULL)) {
      damacy_destroy(d);
      uri_table_free(&uris);
      free(json_buf);
      return 1;
    }
    enum damacy_status fs = damacy_flush(d);
    if (fs != DAMACY_OK && fs != DAMACY_AGAIN)
      fprintf(stderr, "damacy_flush(warmup): %s\n", damacy_status_str(fs));
    struct damacy_batch* b = NULL;
    while (damacy_pop(d, &b) == DAMACY_OK) {
      ++rm.popped;
      damacy_release(d, b);
    }
    damacy_stats_reset(d);
  }

  rm.ttfb_ms = (t_first_pop > 0.0 ? (t_first_pop - t_first_push) : 0.0) * 1e3;

  // Steady-state run. drive() pushes exactly n_batches * samples_per_batch
  // samples and pops exactly n_batches batches, so no trailing flush is
  // needed.
  uint64_t pushed_steady = 0, popped_steady = 0;
  double t_steady_a = now_seconds();
  double t_first_pop_steady = 0.0;
  if (sc.n_warmup_batches == 0) {
    if (drive(d,
              &sc,
              &uris,
              &rng,
              sc.n_batches,
              sc.consumer_hold_ms,
              &pushed_steady,
              &popped_steady,
              &t_first_pop,
              &rm.consumer_block_ms_total,
              &rm.consumer_push_ms_total,
              &rm.consumer_pop_wait_ms_total)) {
      damacy_destroy(d);
      uri_table_free(&uris);
      free(json_buf);
      return 1;
    }
    rm.ttfb_ms = (t_first_pop - t_first_push) * 1e3;
  } else {
    if (drive(d,
              &sc,
              &uris,
              &rng,
              sc.n_batches,
              sc.consumer_hold_ms,
              &pushed_steady,
              &popped_steady,
              &t_first_pop_steady,
              &rm.consumer_block_ms_total,
              &rm.consumer_push_ms_total,
              &rm.consumer_pop_wait_ms_total)) {
      damacy_destroy(d);
      uri_table_free(&uris);
      free(json_buf);
      return 1;
    }
  }
  double t_steady_b = now_seconds();
  rm.wall_ms = (t_steady_b - t_steady_a) * 1e3;

  rm.pushed += pushed_steady;
  rm.popped += popped_steady;

  damacy_stats_get(d, &rm.stats);
  damacy_destroy(d);
  uri_table_free(&uris);

  emit_results(&sc, &rm, stdout);
  free(json_buf);
  return 0;
}
