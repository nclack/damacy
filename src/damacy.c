// damacy: streaming loader.
//
// Step 5: wave scheduler + double buffering.
//
// Two batch slots (B=2) and two wave slots (W=2) in flight. Each
// `damacy_pop` advances the wave state machine, kicks new work into
// any FREE wave slot, and either returns a READY batch or
// poll-sleeps until one becomes available. Waves do not cross batch
// boundaries (each wave belongs to exactly one batch slot); read-op
// coalescing lands in step 7.
//
// Threading: planner / scheduler / CUDA launches all run on the user
// thread inside damacy_push / damacy_pop / damacy_flush. The only
// background threads are the n_io_threads io_queue workers; each job
// is a single `pread` against an FD looked up in store_fs's per-key
// FD cache (`store_fs.c::fs_get_file`).

#include "damacy.h"

#include "assemble/assemble.h"
#include "decoder/bitshuffle.h"
#include "decoder/blosc1.h"
#include "decoder/blosc1_host.h"
#include "decoder/decoder_lz4.h"
#include "decoder/decoder_memcpy.h"
#include "decoder/decoder_zstd.h"
#include "decoder/shuffle.h"
#include "decoder/status_reduce.h"
#include "dtype/dtype.h"
#include "log/log.h"
#include "planner/planner.h"
#include "platform/platform.h"
#include "store/store.h"
#include "threadpool/threadpool.h"
#include "util/prelude.h"
#include "util/strbuf.h"
#include "zarr/zarr_meta_cache.h"
#include "zarr/zarr_metadata.h"
#include "zarr/zarr_shard_cache.h"

#include <cuda.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Per-batch hard cap. Bounds the planner output and the assemble
// metadata buffer. Wave caps + chunk-uncompressed cap live in
// damacy_limits.h since they're shared with kernel modules.
#define DAMACY_MAX_CHUNKS_PER_BATCH 16384u

// Worst-case substream count per wave for blosc1-zstd: 1 substream per
// blosc-block. blosc1-lz4 splits each block into `typesize` substreams,
// so its per-wave cap scales with the runtime max_bytes_per_element knob
// (resolve_max_bpe(cfg)) — see lz4_subs_per_wave().
#define DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE                                    \
  (DAMACY_MAX_CHUNKS_PER_WAVE * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK)
// Memcpy ops cap: every chunk could conceivably be MEMCPYED.
#define DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE DAMACY_MAX_CHUNKS_PER_WAVE
#define DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE DAMACY_MAX_CHUNKS_PER_WAVE

// Worst-case LZ4 substream count for one wave at the given typesize cap.
static inline size_t
lz4_subs_per_wave(uint8_t max_bpe)
{
  return (size_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE * (size_t)max_bpe;
}

// Poll interval inside damacy_pop's wait-loop. ~50 µs is short enough
// that the boundary between wave stages doesn't add visible latency,
// long enough that we don't burn a core spinning.
#define DAMACY_POP_POLL_NS 50000

// CUDA driver-API result check. Goes through log_error so it lands in
// the same channel as the project's CHECK macros.
#define CR(label, expr)                                                        \
  do {                                                                         \
    CUresult _r = (expr);                                                      \
    if (_r != CUDA_SUCCESS) {                                                  \
      const char* _msg = NULL;                                                 \
      cuGetErrorString(_r, &_msg);                                             \
      log_error("cu: %s -> %s", #expr, _msg ? _msg : "?");                     \
      goto label;                                                              \
    }                                                                          \
  } while (0)

// Driver API uses CUdeviceptr (uint64_t) for device pointers; we keep
// our struct fields typed (void*, struct assemble_chunk*, ...) and cast
// at the alloc/free/memcpy boundary. Inverse direction
// (CUdeviceptr → typed) needs the inverse cast inline.
#define CUDPTR(p) ((CUdeviceptr)(uintptr_t)(p))

static uint64_t
monotonic_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// Initialize a damacy_metric to a "no observations yet" state with a
// stable name pointer.
static void
metric_init(struct damacy_metric* m, const char* name)
{
  m->name = name;
  m->ms = 0.f;
  m->best_ms = 1e30f;
  m->input_bytes = 0;
  m->output_bytes = 0;
  m->count = 0;
}

// Record one observation (elapsed ms + per-stage byte totals) into a
// cumulative damacy_metric. For stall-style metrics (no throughput),
// pass 0 for both byte counters.
static void
metric_record(struct damacy_metric* m, float ms, uint64_t bin, uint64_t bout)
{
  m->ms += ms;
  if (ms < m->best_ms)
    m->best_ms = ms;
  m->input_bytes += (double)bin;
  m->output_bytes += (double)bout;
  m->count += 1;
}

// Initialize all metrics in `s` to the no-observation state.
static void
stats_init(struct damacy_stats* s)
{
  memset(s, 0, sizeof(*s));
  metric_init(&s->plan, "plan");
  metric_init(&s->io, "io");
  metric_init(&s->h2d, "h2d");
  metric_init(&s->decompress, "decompress");
  metric_init(&s->decompress_parse, "decompress.parse");
  metric_init(&s->decompress_zstd, "decompress.zstd");
  metric_init(&s->decompress_lz4, "decompress.lz4");
  metric_init(&s->decompress_post, "decompress.post");
  metric_init(&s->assemble, "assemble");
  metric_init(&s->pop_wait_io, "pop_wait_io");
  metric_init(&s->pop_wait_compute, "pop_wait_compute");
  metric_init(&s->flush_wait, "flush_wait");
}

struct damacy_batch
{
  struct damacy* d;
  uint16_t slot_idx;
  uint64_t batch_id;
};

struct damacy_sample_slot
{
  char* uri;
  struct damacy_aabb aabb;
};

struct damacy_lookahead
{
  struct damacy_sample_slot* slots;
  uint32_t cap;
  uint32_t head;
  uint32_t tail;
  uint32_t size;
};

enum batch_slot_state
{
  BATCH_FREE = 0,
  BATCH_FILLING, // planner has emitted; chunks may or may not be dispatched
  BATCH_READY,   // chunks_remaining == 0; awaiting pop
  BATCH_HELD,    // user holds the handle
};

struct damacy_batch_slot
{
  enum batch_slot_state state;
  uint64_t batch_id;
  uint32_t n_samples; // shape[0]: number of complete samples
  void* dev_ptr;      // device output tensor (allocated lazily)

  // Plan for this batch (filled by planner_plan, peeled by waves).
  struct read_op* read_ops;         // size DAMACY_MAX_CHUNKS_PER_BATCH
  struct chunk_plan* chunk_plans;   // size DAMACY_MAX_CHUNKS_PER_BATCH
  struct sample_plan* sample_plans; // size cfg.batch_size
  void* d_sample_plans;             // device mirror, uploaded once per batch
  uint32_t n_sample_plans;          // == n_samples on success
  uint32_t n_chunks;
  uint32_t n_chunks_dispatched; // 0 .. n_chunks; chunks given to a wave
  int32_t chunks_remaining;     // n_chunks - chunks completed via waves
};

struct damacy_batch_pool
{
  struct damacy_batch_slot slots[2];
  uint64_t n_bytes;                     // size of one slot's output
  uint8_t rank;                         // includes leading N axis
  int64_t shape[DAMACY_MAX_RANK + 1];   // [batch_size, ...sample_axes]
  int64_t strides[DAMACY_MAX_RANK + 1]; // row-major elements
  int allocated;                        // shape established + dev_ptrs alloc'd
};

enum wave_state
{
  WAVE_FREE = 0,
  WAVE_IO,
  WAVE_H2D,
  WAVE_ASSEMBLE, // covers decompress + assemble (one stream, one event)
};

struct damacy_wave
{
  enum wave_state state;
  uint16_t batch_pool_slot;    // valid when state != WAVE_FREE
  uint32_t batch_chunk_offset; // index into slot->chunk_plans of first chunk
  uint32_t n_chunks;           // chunks in this wave (1..max_chunks_per_wave)

  // Cursors into the wave's slabs (set at IO submit; read at later stages).
  uint64_t host_used_bytes;

  // Wave-owned device + pinned-host buffers.
  void* host_slab;        // pinned, cfg.host_buffer_bytes / 2
  void* dev_compressed;   // mirrors host_slab on device
  void* dev_decompressed; // arena, cfg.device_buffer_bytes / 2
  uint64_t host_slab_cap;
  uint64_t dev_decompressed_cap;

  // blosc1 host-parse state, all pinned. parse fills these; the fanout
  // / op records H2D onto stream_h2d for the codec + post-decode stages.
  struct blosc1_host_chunk* h_chunks;
  struct blosc1_host_scratch scratch;
  struct blosc1_host_fanout h_zstd_fan;
  struct blosc1_host_fanout h_lz4_fan;
  struct gpu_memcpy_op* h_memcpy_ops;
  struct gpu_shuffle_op* h_unshuffle_ops;
  struct gpu_shuffle_op* h_bitunshuffle_ops;
  struct blosc1_totals* h_blosc1_totals;

  // status_reduce atomicAdds into n_codec_errors; finalize_wave reads.
  struct blosc1_totals* d_blosc1_totals;

  // Device SOA mirrors of the host fanout (H2D'd in kick_h2d).
  struct nvcomp_fanout zstd_fan;
  struct nvcomp_fanout lz4_fan;

  struct gpu_memcpy_op* d_memcpy_ops;
  struct gpu_shuffle_op* d_unshuffle_ops;
  struct gpu_shuffle_op* d_bitunshuffle_ops;

  float parse_ms; // host wall-clock around blosc1_host_parse
  // Same extent as dev_decompressed; staging area used by the
  // (bit)unshuffle kernels so the per-block transpose isn't bounded by
  // the 64 KB shared-memory cap.
  void* dev_unshuffle_scratch;

  // Assemble per-wave-chunk metadata staging (host + device). One
  // record per chunk in the wave, containing the chunk's arena offset
  // and (sample_idx, chunk_d). Per-sample constants live in the batch
  // slot's d_sample_plans, indexed by sample_idx_in_batch.
  struct assemble_chunk* h_assemble_chunks;
  struct assemble_chunk* d_assemble_chunks;
  uint32_t assemble_max_blocks_per_chunk;
  uint8_t assemble_rank;

  // store_read[] scratch.
  struct store_read* store_reads;

  struct store_event io_event;

  // Per-stage timing events (driver API; cuEventElapsedTime requires
  // timing-enabled events). end events double as sync points: ev.h2d_end
  // is the cuStreamWaitEvent target on stream_compute, ev.asm_end is
  // the wave's "done" event polled in advance_waves.
  struct wave_events
  {
    CUevent h2d_start;
    CUevent bulk_h2d_end; // bulk slab H2D done; used for stats.h2d
    CUevent h2d_end;      // + fanout/op H2Ds + d_blosc1_totals zero done
    CUevent decomp_start;
    CUevent zstd_done;
    CUevent lz4_done;
    CUevent post_start; // stream_compute resumes after waiting on codec streams
    CUevent decomp_end;
    CUevent asm_start;
    CUevent asm_end;
  } ev;

  // Host-side IO timing: submit→completion wall-clock.
  uint64_t io_t_start_ns;
  uint64_t io_t_end_ns;

  // Per-stage byte totals (filled at peel + compute time, drained at
  // finalize into the cumulative damacy_stats fields).
  uint64_t io_bytes;
  uint64_t decomp_in_bytes;
  uint64_t decomp_out_bytes;
  uint64_t assemble_out_bytes;

  // Per-wave decoders: nvCOMP scratch (temp workspace + actual-size and
  // status output arrays) is not safe to share across in-flight waves.
  struct decoder_zstd* zstd_decoder;
  struct decoder_lz4* lz4_decoder;
};

struct damacy
{
  struct damacy_config cfg;
  enum damacy_status failed_status;
  uint64_t next_batch_id;
  uint64_t page_alignment;
  int cuda_device;
  int retained_primary_device; // -1 = captured caller's ctx; else release at
                               // destroy

  // GPU memory budgeting: total bytes committed at create-time (waves +
  // per-batch metadata). Lazy batch-output tensors are summed against
  // gpu_bytes_budget at batch_pool_allocate. 0 budget = no cap.
  uint64_t gpu_bytes_committed;
  uint64_t gpu_bytes_budget;

  struct store* store;
  struct zarr_meta_cache* meta_cache;
  struct zarr_shard_cache* shard_cache;
  struct planner* planner;
  CUstream stream_h2d;
  CUstream stream_compute;
  CUstream stream_zstd; // nvcomp Zstd batch
  CUstream stream_lz4;  // nvcomp LZ4 batch

  struct threadpool* compute_pool; // host blosc1 parse

  struct damacy_lookahead lookahead;
  struct damacy_batch_pool batch_pool;
  struct damacy_wave waves[2];

  // Sample working set used while planning one batch.
  struct damacy_sample_slot* batch_samples;
  struct damacy_sample* batch_stage;

  struct damacy_batch handle;
  struct damacy_stats stats;
};

// --- dtype helpers --------------------------------------------------------

static uint32_t
damacy_dtype_bpe(enum damacy_dtype dt)
{
  switch (dt) {
    case DAMACY_BF16:
      return 2;
    case DAMACY_F32:
      return 4;
  }
  return 0;
}

// Cast support matrix: zarr source dtype → cfg destination dtype. Sources
// are restricted to the common image types; integer→float and float→float
// casts are valid for both bf16 and f32 destinations. Returns 1 if the
// (src, dst) pair has a cast path the assemble kernel will accept.
static int
cast_path_supported(enum damacy_dtype dst, enum dtype src)
{
  switch (dst) {
    case DAMACY_F32:
    case DAMACY_BF16:
      switch (src) {
        case dtype_u8:
        case dtype_u16:
        case dtype_i16:
        case dtype_u32:
        case dtype_i32:
        case dtype_f16:
        case dtype_f32:
          return 1;
        default:
          return 0;
      }
  }
  return 0;
}

// --- config validation ----------------------------------------------------

static enum damacy_status
validate_config(const struct damacy_config* cfg)
{
  CHECK_SILENT(Invalid, cfg);
  CHECK_SILENT(Invalid, cfg->batch_size > 0);
  CHECK_SILENT(Invalid, cfg->lookahead_batches >= 2);
  CHECK_SILENT(Invalid, cfg->n_io_threads > 0);
  CHECK_SILENT(Invalid, cfg->host_buffer_bytes > 0);
  CHECK_SILENT(Invalid, cfg->device_buffer_bytes > 0);
  CHECK_SILENT(Invalid, cfg->n_zarrs_meta_cache > 0);
  CHECK_SILENT(Invalid, cfg->n_shards_meta_cache > 0);
  CHECK_SILENT(Invalid, damacy_dtype_bpe(cfg->dtype) > 0);
  // 0 means "use the compile-time ceiling"; reject values that exceed it
  // since they'd run past the per-wave LZ4 fanout SOA and nvcomp scratch
  // sizing.
  CHECK_SILENT(Invalid,
               cfg->max_bytes_per_element <= DAMACY_BLOSC_MAX_TYPESIZE);
  // Runtime per-chunk cap is bounded by the kernel-array ceiling.
  CHECK_SILENT(Invalid,
               cfg->max_chunk_uncompressed_bytes <=
                 DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES);
  return DAMACY_OK;
Invalid:
  return DAMACY_INVAL;
}

// Resolve the effective max-bytes-per-element from the config. Centralised
// because both wave_init and damacy_create's pipeline-wide caps need it,
// and mapping 0 → ceiling lives in one place.
static uint8_t
resolve_max_bpe(const struct damacy_config* cfg)
{
  return cfg->max_bytes_per_element ? cfg->max_bytes_per_element
                                    : (uint8_t)DAMACY_BLOSC_MAX_TYPESIZE;
}

// Resolve the effective per-chunk uncompressed byte cap. 0 maps to the
// 512 KB default; clamps to the compile-time ceiling for safety.
static uint64_t
resolve_max_chunk_uncompressed(const struct damacy_config* cfg)
{
  uint64_t v = cfg->max_chunk_uncompressed_bytes;
  if (v == 0)
    v = DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES;
  if (v > DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES)
    v = DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES;
  return v;
}

// --- GPU memory budget ---------------------------------------------------

// Categorised breakdown of expected device-resident bytes for the wave
// pool plus per-batch metadata. Lazy batch-output tensors are accounted
// separately at batch_pool_allocate time (size depends on first AABB).
struct gpu_budget
{
  uint64_t dev_compressed;        // 2× host_per_wave (H2D mirror)
  uint64_t dev_decompressed;      // 2× dev_per_wave
  uint64_t dev_unshuffle_scratch; // 2× dev_per_wave
  uint64_t blosc1_meta;           // 2× per-wave parse + assemble metadata
  uint64_t fanout_soa;            // 2× per-wave nvcomp fanout SOA + op arrays
  uint64_t nvcomp_temp;           // 2× (zstd_temp + lz4_temp + actual+status)
  uint64_t batch_metadata;        // 2× cfg.batch_size × sizeof(sample_plan)
  uint64_t total;
};

// Sum of struct-array allocs wave_init makes for blosc1 parse + assemble
// metadata, scaled to one wave. With the parse moved to the host the only
// device-resident blosc1 metadata is the totals struct used by status_reduce.
static uint64_t
wave_blosc1_meta_bytes(void)
{
  const uint64_t cap = (uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE;
  return cap * sizeof(struct assemble_chunk) + sizeof(struct blosc1_totals);
}

// One wave's nvcomp fanout SOA + memcpy/shuffle op arrays.
static uint64_t
wave_fanout_soa_bytes(uint8_t max_bpe)
{
  const uint64_t zsubs = (uint64_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;
  const uint64_t lsubs = (uint64_t)lz4_subs_per_wave(max_bpe);
  // Each codec fanout: 2× (void*) + 2× (size_t).
  uint64_t soa = zsubs * (2 * sizeof(void*) + 2 * sizeof(size_t)) +
                 lsubs * (2 * sizeof(void*) + 2 * sizeof(size_t));
  soa += (uint64_t)DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
         sizeof(struct gpu_memcpy_op);
  soa += 2ull * (uint64_t)DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
         sizeof(struct gpu_shuffle_op);
  return soa;
}

// Predicted nvcomp scratch (temp workspace + actual-size + status arrays)
// for one wave, mirroring wave_init's capacity math.
static enum damacy_status
wave_nvcomp_bytes(const struct damacy_config* cfg, uint64_t* out)
{
  const uint64_t dev_per_wave = cfg->device_buffer_bytes / 2;
  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  const uint8_t max_bpe = resolve_max_bpe(cfg);
  const uint64_t zsubs = (uint64_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;
  const uint64_t lsubs = (uint64_t)lz4_subs_per_wave(max_bpe);
  const uint64_t cap_worst =
    (uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE * runtime_chunk_cap;
  const uint64_t total_uncompressed =
    dev_per_wave < cap_worst ? dev_per_wave : cap_worst;
  uint64_t zstd_per = runtime_chunk_cap;
  if (zstd_per > total_uncompressed && total_uncompressed > 0)
    zstd_per = total_uncompressed;
  uint64_t lz4_per = zstd_per / max_bpe;
  if (lz4_per == 0)
    lz4_per = 1;
  size_t zstd_temp = 0, lz4_temp = 0;
  if (decoder_zstd_query_temp_bytes(
        zsubs, (size_t)zstd_per, (size_t)total_uncompressed, &zstd_temp))
    return DAMACY_CUDA;
  if (decoder_lz4_query_temp_bytes(
        lsubs, (size_t)lz4_per, (size_t)total_uncompressed, &lz4_temp))
    return DAMACY_CUDA;
  // Per-codec actual-size (size_t) and status (int) output arrays.
  *out = (uint64_t)zstd_temp + zsubs * sizeof(size_t) + zsubs * sizeof(int) +
         (uint64_t)lz4_temp + lsubs * sizeof(size_t) + lsubs * sizeof(int);
  return DAMACY_OK;
}

static enum damacy_status
gpu_budget_compute(const struct damacy_config* cfg, struct gpu_budget* out)
{
  const uint64_t host_per_wave = cfg->host_buffer_bytes / 2;
  const uint64_t dev_per_wave = cfg->device_buffer_bytes / 2;
  const uint8_t max_bpe = resolve_max_bpe(cfg);

  uint64_t per_wave_nvcomp = 0;
  enum damacy_status s = wave_nvcomp_bytes(cfg, &per_wave_nvcomp);
  if (s != DAMACY_OK)
    return s;

  out->dev_compressed = 2ull * host_per_wave;
  out->dev_decompressed = 2ull * dev_per_wave;
  out->dev_unshuffle_scratch = 2ull * dev_per_wave;
  out->blosc1_meta = 2ull * wave_blosc1_meta_bytes();
  out->fanout_soa = 2ull * wave_fanout_soa_bytes(max_bpe);
  out->nvcomp_temp = 2ull * per_wave_nvcomp;
  out->batch_metadata =
    2ull * (uint64_t)cfg->batch_size * sizeof(struct sample_plan);
  out->total = out->dev_compressed + out->dev_decompressed +
               out->dev_unshuffle_scratch + out->blosc1_meta + out->fanout_soa +
               out->nvcomp_temp + out->batch_metadata;
  return DAMACY_OK;
}

// --- lookahead ring -------------------------------------------------------

static void
sample_slot_clear(struct damacy_sample_slot* slot)
{
  if (!slot)
    return;
  free(slot->uri);
  slot->uri = NULL;
  memset(&slot->aabb, 0, sizeof(slot->aabb));
}

static int
lookahead_init(struct damacy_lookahead* la, uint32_t cap)
{
  la->slots =
    (struct damacy_sample_slot*)calloc(cap, sizeof(struct damacy_sample_slot));
  CHECK(Error, la->slots);
  la->cap = cap;
  la->head = 0;
  la->tail = 0;
  la->size = 0;
  return 0;
Error:
  return 1;
}

static void
lookahead_destroy(struct damacy_lookahead* la)
{
  if (!la || !la->slots)
    return;
  for (uint32_t i = 0; i < la->cap; ++i)
    sample_slot_clear(&la->slots[i]);
  free(la->slots);
  la->slots = NULL;
}

static int
lookahead_push(struct damacy_lookahead* la, const struct damacy_sample* sample)
{
  if (la->size == la->cap)
    return 1;
  struct damacy_sample_slot* slot = &la->slots[la->tail];
  slot->uri = strdup(sample->uri);
  if (!slot->uri)
    return 1;
  slot->aabb = sample->aabb;
  la->tail = (la->tail + 1) % la->cap;
  la->size++;
  return 0;
}

static void
lookahead_drain(struct damacy_lookahead* la,
                struct damacy_sample_slot* out,
                uint32_t n)
{
  for (uint32_t i = 0; i < n; ++i) {
    out[i] = la->slots[la->head];
    la->slots[la->head] = (struct damacy_sample_slot){ 0 };
    la->head = (la->head + 1) % la->cap;
    la->size--;
  }
}

// --- batch pool -----------------------------------------------------------

static int
batch_slot_init(struct damacy_batch_slot* slot, uint32_t batch_size_cap)
{
  slot->read_ops = (struct read_op*)calloc(DAMACY_MAX_CHUNKS_PER_BATCH,
                                           sizeof(struct read_op));
  CHECK(Error, slot->read_ops);
  slot->chunk_plans = (struct chunk_plan*)calloc(DAMACY_MAX_CHUNKS_PER_BATCH,
                                                 sizeof(struct chunk_plan));
  CHECK(Error, slot->chunk_plans);
  slot->sample_plans =
    (struct sample_plan*)calloc(batch_size_cap, sizeof(struct sample_plan));
  CHECK(Error, slot->sample_plans);
  CUdeviceptr dptr = 0;
  if (cuMemAlloc(&dptr, (size_t)batch_size_cap * sizeof(struct sample_plan)) !=
      CUDA_SUCCESS)
    goto Error;
  slot->d_sample_plans = (void*)(uintptr_t)dptr;
  return 0;
Error:
  return 1;
}

static void
batch_slot_destroy(struct damacy_batch_slot* slot)
{
  if (!slot)
    return;
  // dev_ptr is owned by the batch_pool's lazy allocation; freed there.
  free(slot->read_ops);
  free(slot->chunk_plans);
  free(slot->sample_plans);
  if (slot->d_sample_plans)
    cuMemFree(CUDPTR(slot->d_sample_plans));
  memset(slot, 0, sizeof(*slot));
}

// Tear down both batch slots' device tensors and per-slot heap. Safe on
// a zero-initialized pool.
static void
batch_pool_destroy(struct damacy_batch_pool* pool)
{
  if (!pool)
    return;
  for (int s = 0; s < 2; ++s) {
    if (pool->slots[s].dev_ptr)
      cuMemFree(CUDPTR(pool->slots[s].dev_ptr));
    batch_slot_destroy(&pool->slots[s]);
  }
  memset(pool, 0, sizeof(*pool));
}

// Establish the shared output shape from the first sample's AABB and
// allocate device tensors for both batch slots. Idempotent.
static enum damacy_status
batch_pool_allocate(struct damacy* self, const struct damacy_aabb* sample_aabb)
{
  struct damacy_batch_pool* pool = &self->batch_pool;
  if (pool->allocated)
    return DAMACY_OK;

  uint8_t spatial_rank = sample_aabb->rank;
  uint8_t full_rank = (uint8_t)(spatial_rank + 1);
  CHECK_SILENT(Rank, full_rank <= DAMACY_MAX_RANK + 1);

  uint32_t bpe = damacy_dtype_bpe(self->cfg.dtype);
  int64_t spatial_volume = 1;
  pool->shape[0] = (int64_t)self->cfg.batch_size;
  for (uint8_t d = 0; d < spatial_rank; ++d) {
    int64_t extent = sample_aabb->dims[d].end - sample_aabb->dims[d].beg;
    CHECK_SILENT(Invalid, extent > 0);
    pool->shape[d + 1] = extent;
    spatial_volume *= extent;
  }
  pool->rank = full_rank;
  pool->strides[full_rank - 1] = 1;
  for (int d = (int)full_rank - 2; d >= 0; --d)
    pool->strides[d] = pool->strides[d + 1] * pool->shape[d + 1];
  pool->n_bytes =
    (uint64_t)self->cfg.batch_size * (uint64_t)spatial_volume * (uint64_t)bpe;

  // Lazy batch-output check against the GPU budget. Two slot tensors land
  // here; the wave/batch-metadata totals were already committed at create.
  if (self->gpu_bytes_budget > 0) {
    uint64_t need = 2ull * pool->n_bytes;
    if (self->gpu_bytes_committed + need > self->gpu_bytes_budget) {
      log_error("damacy: batch-output pool would exceed GPU budget "
                "(committed=%llu add=%llu cap=%llu n_bytes=%llu)",
                (unsigned long long)self->gpu_bytes_committed,
                (unsigned long long)need,
                (unsigned long long)self->gpu_bytes_budget,
                (unsigned long long)pool->n_bytes);
      return DAMACY_OOM;
    }
  }

  for (int s = 0; s < 2; ++s) {
    CUdeviceptr dptr = 0;
    CR(CudaFail, cuMemAlloc(&dptr, pool->n_bytes));
    pool->slots[s].dev_ptr = (void*)(uintptr_t)dptr;
  }
  self->gpu_bytes_committed += 2ull * pool->n_bytes;
  pool->allocated = 1;
  return DAMACY_OK;

Rank:
  return DAMACY_RANK;
Invalid:
  return DAMACY_INVAL;
CudaFail:
  return DAMACY_CUDA;
}

static int
sample_shape_matches_pool(const struct damacy* self,
                          const struct damacy_aabb* aabb)
{
  uint8_t spatial_rank = (uint8_t)(self->batch_pool.rank - 1);
  if (aabb->rank != spatial_rank)
    return 0;
  for (uint8_t d = 0; d < spatial_rank; ++d) {
    int64_t extent = aabb->dims[d].end - aabb->dims[d].beg;
    if (extent != self->batch_pool.shape[d + 1])
      return 0;
  }
  return 1;
}

static int
find_free_batch_slot(const struct damacy* self)
{
  for (int s = 0; s < 2; ++s)
    if (self->batch_pool.slots[s].state == BATCH_FREE)
      return s;
  return -1;
}

// Find the oldest READY batch slot (lowest batch_id). Returns -1 if none.
static int
find_oldest_ready_slot(const struct damacy* self)
{
  int best = -1;
  uint64_t best_id = UINT64_MAX;
  for (int s = 0; s < 2; ++s) {
    if (self->batch_pool.slots[s].state == BATCH_READY &&
        self->batch_pool.slots[s].batch_id < best_id) {
      best = s;
      best_id = self->batch_pool.slots[s].batch_id;
    }
  }
  return best;
}

// Same shape, but for FILLING slots. Used by damacy_flush to detect
// whether more work needs to be drained.
static int
find_oldest_filling_slot(const struct damacy* self)
{
  int best = -1;
  uint64_t best_id = UINT64_MAX;
  for (int s = 0; s < 2; ++s) {
    if (self->batch_pool.slots[s].state == BATCH_FILLING &&
        self->batch_pool.slots[s].batch_id < best_id) {
      best = s;
      best_id = self->batch_pool.slots[s].batch_id;
    }
  }
  return best;
}

// Any batch slot still in a non-terminal active state.
static int
any_batch_in_flight(const struct damacy* self)
{
  for (int s = 0; s < 2; ++s) {
    enum batch_slot_state st = self->batch_pool.slots[s].state;
    if (st == BATCH_FILLING || st == BATCH_READY || st == BATCH_HELD)
      return 1;
  }
  return 0;
}

// --- wave -----------------------------------------------------------------

static int
wave_init(struct damacy_wave* wave,
          uint64_t host_slab_bytes,
          uint64_t dev_decompressed_bytes,
          uint8_t max_bpe,
          uint64_t max_chunk_uncompressed_bytes)
{
  wave->state = WAVE_FREE;
  wave->host_slab_cap = host_slab_bytes;
  wave->dev_decompressed_cap = dev_decompressed_bytes;

  CR(Error, cuMemAllocHost(&wave->host_slab, host_slab_bytes));
  CUdeviceptr dptr = 0;
  CR(Error, cuMemAlloc(&dptr, host_slab_bytes));
  wave->dev_compressed = (void*)(uintptr_t)dptr;
  CR(Error, cuMemAlloc(&dptr, dev_decompressed_bytes));
  wave->dev_decompressed = (void*)(uintptr_t)dptr;

  uint32_t cap = DAMACY_MAX_CHUNKS_PER_WAVE;
  CR(Error,
     cuMemAllocHost((void**)&wave->h_chunks,
                    (size_t)cap * sizeof(struct blosc1_host_chunk)));
  CR(Error,
     cuMemAllocHost((void**)&wave->scratch.hdrs,
                    (size_t)cap * sizeof(struct blosc1_chunk_hdr)));
  CR(Error,
     cuMemAllocHost((void**)&wave->scratch.counts,
                    (size_t)cap * sizeof(struct blosc1_chunk_counts)));
  CR(Error,
     cuMemAllocHost((void**)&wave->scratch.offsets,
                    (size_t)cap * sizeof(struct blosc1_chunk_offsets)));
  // Pure host scratch, but pinned for consistency with the rest of the
  // scratch arrays. cap * MAX_BLOCKS uint32_t each (~64 KB per array
  // at the current caps).
  CR(Error,
     cuMemAllocHost((void**)&wave->scratch.bstarts,
                    (size_t)cap * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK *
                      sizeof(uint32_t)));
  CR(Error,
     cuMemAllocHost((void**)&wave->scratch.block_ends,
                    (size_t)cap * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK *
                      sizeof(uint32_t)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_blosc1_totals,
                    sizeof(struct blosc1_totals)));
  wave->h_assemble_chunks =
    (struct assemble_chunk*)calloc(cap, sizeof(struct assemble_chunk));
  CHECK(Error, wave->h_assemble_chunks);
  wave->store_reads =
    (struct store_read*)calloc(cap, sizeof(struct store_read));
  CHECK(Error, wave->store_reads);

  CR(Error, cuMemAlloc(&dptr, (size_t)cap * sizeof(struct assemble_chunk)));
  wave->d_assemble_chunks = (struct assemble_chunk*)(uintptr_t)dptr;

  CR(Error, cuMemAlloc(&dptr, sizeof(struct blosc1_totals)));
  wave->d_blosc1_totals = (struct blosc1_totals*)(uintptr_t)dptr;

  const size_t zsubs = DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE;
  const size_t lsubs = lz4_subs_per_wave(max_bpe);
  CR(
    Error,
    cuMemAllocHost((void**)&wave->h_zstd_fan.comp_ptrs, zsubs * sizeof(void*)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_zstd_fan.comp_sizes,
                    zsubs * sizeof(size_t)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_zstd_fan.decomp_ptrs,
                    zsubs * sizeof(void*)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_zstd_fan.decomp_buf_sizes,
                    zsubs * sizeof(size_t)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_lz4_fan.comp_ptrs, lsubs * sizeof(void*)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_lz4_fan.comp_sizes,
                    lsubs * sizeof(size_t)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_lz4_fan.decomp_ptrs,
                    lsubs * sizeof(void*)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_lz4_fan.decomp_buf_sizes,
                    lsubs * sizeof(size_t)));
  CR(Error, cuMemAlloc(&dptr, zsubs * sizeof(void*)));
  wave->zstd_fan.d_comp_ptrs = (const void**)(uintptr_t)dptr;
  CR(Error, cuMemAlloc(&dptr, zsubs * sizeof(size_t)));
  wave->zstd_fan.d_comp_sizes = (size_t*)(uintptr_t)dptr;
  CR(Error, cuMemAlloc(&dptr, zsubs * sizeof(void*)));
  wave->zstd_fan.d_decomp_ptrs = (void**)(uintptr_t)dptr;
  CR(Error, cuMemAlloc(&dptr, zsubs * sizeof(size_t)));
  wave->zstd_fan.d_decomp_buf_sizes = (size_t*)(uintptr_t)dptr;
  CR(Error, cuMemAlloc(&dptr, lsubs * sizeof(void*)));
  wave->lz4_fan.d_comp_ptrs = (const void**)(uintptr_t)dptr;
  CR(Error, cuMemAlloc(&dptr, lsubs * sizeof(size_t)));
  wave->lz4_fan.d_comp_sizes = (size_t*)(uintptr_t)dptr;
  CR(Error, cuMemAlloc(&dptr, lsubs * sizeof(void*)));
  wave->lz4_fan.d_decomp_ptrs = (void**)(uintptr_t)dptr;
  CR(Error, cuMemAlloc(&dptr, lsubs * sizeof(size_t)));
  wave->lz4_fan.d_decomp_buf_sizes = (size_t*)(uintptr_t)dptr;

  CR(Error,
     cuMemAllocHost((void**)&wave->h_memcpy_ops,
                    DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                      sizeof(struct gpu_memcpy_op)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_unshuffle_ops,
                    DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                      sizeof(struct gpu_shuffle_op)));
  CR(Error,
     cuMemAllocHost((void**)&wave->h_bitunshuffle_ops,
                    DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                      sizeof(struct gpu_shuffle_op)));
  CR(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE *
                  sizeof(struct gpu_memcpy_op)));
  wave->d_memcpy_ops = (struct gpu_memcpy_op*)(uintptr_t)dptr;
  CR(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                  sizeof(struct gpu_shuffle_op)));
  wave->d_unshuffle_ops = (struct gpu_shuffle_op*)(uintptr_t)dptr;
  CR(Error,
     cuMemAlloc(&dptr,
                DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE *
                  sizeof(struct gpu_shuffle_op)));
  wave->d_bitunshuffle_ops = (struct gpu_shuffle_op*)(uintptr_t)dptr;
  CR(Error, cuMemAlloc(&dptr, dev_decompressed_bytes));
  wave->dev_unshuffle_scratch = (void*)(uintptr_t)dptr;

  CR(Error, cuEventCreate(&wave->ev.h2d_start, CU_EVENT_DEFAULT));
  CR(Error, cuEventCreate(&wave->ev.bulk_h2d_end, CU_EVENT_DEFAULT));
  CR(Error, cuEventCreate(&wave->ev.h2d_end, CU_EVENT_DEFAULT));
  CR(Error, cuEventCreate(&wave->ev.decomp_start, CU_EVENT_DEFAULT));
  CR(Error, cuEventCreate(&wave->ev.zstd_done, CU_EVENT_DEFAULT));
  CR(Error, cuEventCreate(&wave->ev.lz4_done, CU_EVENT_DEFAULT));
  CR(Error, cuEventCreate(&wave->ev.post_start, CU_EVENT_DEFAULT));
  CR(Error, cuEventCreate(&wave->ev.decomp_end, CU_EVENT_DEFAULT));
  CR(Error, cuEventCreate(&wave->ev.asm_start, CU_EVENT_DEFAULT));
  CR(Error, cuEventCreate(&wave->ev.asm_end, CU_EVENT_DEFAULT));

  // nvcomp temp scratch is sized off min(runtime per-chunk cap × wave
  // chunks, runtime per-wave decompress budget). The runtime cap (set
  // by cfg.max_chunk_uncompressed_bytes; default 512 KB) is the lever
  // that lets users keep nvcomp scratch small on tight GPU budgets while
  // still letting the compile-time ceiling stretch to 2 MB for users on
  // bigger devices.
  const size_t cap_chunks = DAMACY_MAX_CHUNKS_PER_WAVE;
  const size_t runtime_chunk_cap = (size_t)max_chunk_uncompressed_bytes;
  const size_t cap_worst = cap_chunks * runtime_chunk_cap;
  const size_t wave_total_uncompressed =
    dev_decompressed_bytes < cap_worst ? dev_decompressed_bytes : cap_worst;
  // Zstd substream == one blosc-block, ≤ chunk_uncompressed_cap.
  // LZ4 substream == blosc-block / typesize, so the per-substream cap
  // tightens by max_bpe — directly shrinks nvcomp's LZ4 temp scratch.
  size_t zstd_per_substream_cap = runtime_chunk_cap;
  if (zstd_per_substream_cap > wave_total_uncompressed &&
      wave_total_uncompressed > 0)
    zstd_per_substream_cap = wave_total_uncompressed;
  size_t lz4_per_substream_cap = zstd_per_substream_cap / (size_t)max_bpe;
  if (lz4_per_substream_cap == 0)
    lz4_per_substream_cap = 1;
  wave->zstd_decoder = decoder_zstd_create(DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE,
                                           zstd_per_substream_cap,
                                           wave_total_uncompressed);
  CHECK(Error, wave->zstd_decoder);
  wave->lz4_decoder =
    decoder_lz4_create(lsubs, lz4_per_substream_cap, wave_total_uncompressed);
  CHECK(Error, wave->lz4_decoder);
  return 0;
Error:
  return 1;
}

static void
wave_destroy(struct damacy_wave* wave)
{
  if (!wave)
    return;
  decoder_zstd_destroy(wave->zstd_decoder);
  decoder_lz4_destroy(wave->lz4_decoder);
  // cuMemFreeHost(NULL) is invalid; guard each.
  void* const host_ptrs[] = {
    wave->host_slab,
    wave->h_chunks,
    wave->scratch.hdrs,
    wave->scratch.counts,
    wave->scratch.offsets,
    wave->scratch.bstarts,
    wave->scratch.block_ends,
    wave->h_blosc1_totals,
    (void*)wave->h_zstd_fan.comp_ptrs,
    wave->h_zstd_fan.comp_sizes,
    wave->h_zstd_fan.decomp_ptrs,
    wave->h_zstd_fan.decomp_buf_sizes,
    (void*)wave->h_lz4_fan.comp_ptrs,
    wave->h_lz4_fan.comp_sizes,
    wave->h_lz4_fan.decomp_ptrs,
    wave->h_lz4_fan.decomp_buf_sizes,
    wave->h_memcpy_ops,
    wave->h_unshuffle_ops,
    wave->h_bitunshuffle_ops,
  };
  for (size_t i = 0; i < countof(host_ptrs); ++i)
    if (host_ptrs[i])
      cuMemFreeHost(host_ptrs[i]);
  // Bulk device-pointer free. cuMemFree(0) returns CUDA_ERROR_INVALID_VALUE,
  // so we do guard, but the loop pattern keeps this to a single branch.
  void* const dev_ptrs[] = {
    wave->dev_compressed,
    wave->dev_decompressed,
    wave->d_assemble_chunks,
    wave->d_blosc1_totals,
    (void*)wave->zstd_fan.d_comp_ptrs,
    wave->zstd_fan.d_comp_sizes,
    wave->zstd_fan.d_decomp_ptrs,
    wave->zstd_fan.d_decomp_buf_sizes,
    (void*)wave->lz4_fan.d_comp_ptrs,
    wave->lz4_fan.d_comp_sizes,
    wave->lz4_fan.d_decomp_ptrs,
    wave->lz4_fan.d_decomp_buf_sizes,
    wave->d_memcpy_ops,
    wave->d_unshuffle_ops,
    wave->d_bitunshuffle_ops,
    wave->dev_unshuffle_scratch,
  };
  for (size_t i = 0; i < countof(dev_ptrs); ++i)
    if (dev_ptrs[i])
      cuMemFree(CUDPTR(dev_ptrs[i]));
  // cuEventDestroy is not no-op on NULL; guard each.
  CUevent* const events[] = { &wave->ev.h2d_start,  &wave->ev.bulk_h2d_end,
                              &wave->ev.h2d_end,    &wave->ev.decomp_start,
                              &wave->ev.zstd_done,  &wave->ev.lz4_done,
                              &wave->ev.post_start, &wave->ev.decomp_end,
                              &wave->ev.asm_start,  &wave->ev.asm_end };
  for (size_t i = 0; i < countof(events); ++i)
    if (*events[i])
      cuEventDestroy_v2(*events[i]);
  free(wave->h_assemble_chunks);
  free(wave->store_reads);
  memset(wave, 0, sizeof(*wave));
}

static int
find_free_wave(const struct damacy* self)
{
  for (int w = 0; w < 2; ++w)
    if (self->waves[w].state == WAVE_FREE)
      return w;
  return -1;
}

static int
any_wave_in_flight(const struct damacy* self)
{
  for (int w = 0; w < 2; ++w)
    if (self->waves[w].state != WAVE_FREE)
      return 1;
  return 0;
}

// --- planning -------------------------------------------------------------

// Validate one sample against the meta cache + cfg. Returns the appropriate
// damacy_status; on OK, the sample is pushed into the lookahead.
static enum damacy_status
push_one(struct damacy* self, const struct damacy_sample* sample)
{
  if (!sample->uri)
    return DAMACY_INVAL;
  if (sample->aabb.rank == 0 || sample->aabb.rank > DAMACY_MAX_RANK)
    return DAMACY_RANK;

  const struct zarr_metadata* meta = NULL;
  enum damacy_status ms =
    zarr_meta_cache_get(self->meta_cache, sample->uri, &meta);
  if (ms != DAMACY_OK)
    return ms;

  if (!cast_path_supported(self->cfg.dtype, meta->dtype))
    return DAMACY_DTYPE;
  if (sample->aabb.rank != meta->rank)
    return DAMACY_RANK;
  if (self->batch_pool.allocated &&
      !sample_shape_matches_pool(self, &sample->aabb))
    return DAMACY_INVAL;

  if (lookahead_push(&self->lookahead, sample))
    return DAMACY_OOM;
  return DAMACY_OK;
}

// Drain n_samples from the lookahead, plan them into the given batch slot,
// and transition it FREE→FILLING. If n_samples == 0 this is a no-op
// returning OK.
static enum damacy_status
plan_into_slot(struct damacy* self, uint16_t slot_idx, uint32_t n_samples)
{
  if (n_samples == 0)
    return DAMACY_OK;
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  if (slot->state != BATCH_FREE)
    return DAMACY_INVAL;

  lookahead_drain(&self->lookahead, self->batch_samples, n_samples);

  enum damacy_status status =
    batch_pool_allocate(self, &self->batch_samples[0].aabb);
  if (status != DAMACY_OK)
    goto Cleanup;

  for (uint32_t i = 0; i < n_samples; ++i) {
    self->batch_stage[i].uri = self->batch_samples[i].uri;
    self->batch_stage[i].aabb = self->batch_samples[i].aabb;
  }

  struct planner_output plan_out = {
    .read_ops = slot->read_ops,
    .read_ops_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .chunk_plans = slot->chunk_plans,
    .chunk_plans_cap = DAMACY_MAX_CHUNKS_PER_BATCH,
    .sample_plans = slot->sample_plans,
    .sample_plans_cap = self->cfg.batch_size,
  };
  uint64_t plan_t0 = monotonic_ns();
  status = planner_plan(self->planner,
                        self->batch_stage,
                        n_samples,
                        slot_idx,
                        self->batch_pool.strides,
                        self->batch_pool.rank,
                        &plan_out);
  metric_record(
    &self->stats.plan, (float)((monotonic_ns() - plan_t0) / 1.0e6), 0, 0);
  if (status != DAMACY_OK)
    goto Cleanup;

  slot->n_chunks = plan_out.n_chunk_plans;
  slot->n_chunks_dispatched = 0;
  slot->chunks_remaining = (int32_t)plan_out.n_chunk_plans;
  slot->n_sample_plans = plan_out.n_sample_plans;
  slot->n_samples = n_samples;
  slot->batch_id = self->next_batch_id++;
  slot->state = BATCH_FILLING;

  // Upload sample_plans to device once per batch. Waves consume them
  // alongside their per-wave chunk records.
  if (plan_out.n_sample_plans > 0) {
    if (cuMemcpyHtoD(CUDPTR(slot->d_sample_plans),
                     slot->sample_plans,
                     (size_t)plan_out.n_sample_plans *
                       sizeof(struct sample_plan)) != CUDA_SUCCESS) {
      status = DAMACY_CUDA;
      goto Cleanup;
    }
  }

  // Degenerate batch: planner emits no chunks → output stays
  // zero-initialized; transition straight to READY after zeroing.
  if (slot->n_chunks == 0) {
    if (cuMemsetD8(CUDPTR(slot->dev_ptr), 0, self->batch_pool.n_bytes) !=
        CUDA_SUCCESS) {
      status = DAMACY_CUDA;
      goto Cleanup;
    }
    slot->state = BATCH_READY;
  }

Cleanup:
  for (uint32_t i = 0; i < n_samples; ++i)
    sample_slot_clear(&self->batch_samples[i]);
  if (status != DAMACY_OK)
    self->failed_status = status;
  return status;
}

// --- wave kick / advance --------------------------------------------------

// Pack as many of slot's remaining chunks as fit in wave's host slab,
// preserving page alignment. Build the read_op layout (dst_buf_offset
// into host slab) and dev_decompressed arena offsets. Submits store
// reads, captures io_event, transitions wave to WAVE_IO. Returns number
// of chunks taken (0 on no-progress).
static enum damacy_status
peel_wave(struct damacy* self, uint16_t wave_idx, uint16_t slot_idx)
{
  struct damacy_wave* wave = &self->waves[wave_idx];
  struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
  uint32_t base = slot->n_chunks_dispatched;
  uint32_t remaining = slot->n_chunks - base;
  if (remaining == 0)
    return DAMACY_OK;

  // Greedy fill of host slab + dev_decompressed arena.
  uint64_t host_cursor = 0;
  uint64_t dev_cursor = 0;
  uint32_t take = 0;
  for (; take < remaining && take < DAMACY_MAX_CHUNKS_PER_WAVE; ++take) {
    struct read_op* r = &slot->read_ops[base + take];
    struct chunk_plan* c = &slot->chunk_plans[base + take];
    if (host_cursor + r->nbytes > wave->host_slab_cap)
      break;
    if (dev_cursor + c->decompressed_nbytes > wave->dev_decompressed_cap)
      break;
    r->dst_buf_offset = host_cursor;
    c->dev_decompressed_offset = dev_cursor;
    host_cursor += r->nbytes;
    dev_cursor += c->decompressed_nbytes;
  }
  if (take == 0) {
    // A single chunk doesn't fit. Per-wave caps are too tight for this
    // workload. Failure mode: surface it loud rather than livelocking.
    log_error("wave: chunk too large for wave slab "
              "(host_slab_cap=%llu device_buf_cap=%llu)",
              (unsigned long long)wave->host_slab_cap,
              (unsigned long long)wave->dev_decompressed_cap);
    self->failed_status = DAMACY_OOM;
    return DAMACY_OOM;
  }

  // Build store_read[] and submit.
  for (uint32_t i = 0; i < take; ++i) {
    struct read_op* r = &slot->read_ops[base + i];
    wave->store_reads[i] = (struct store_read){
      .key = r->shard_path,
      .dst = (uint8_t*)wave->host_slab + r->dst_buf_offset,
      .offset = r->file_offset,
      .len = r->nbytes,
    };
  }
  wave->io_t_start_ns = monotonic_ns();
  wave->io_event = store_read_submit(self->store, wave->store_reads, take);
  if (wave->io_event.seq == 0) {
    self->failed_status = DAMACY_IO;
    return DAMACY_IO;
  }

  wave->batch_pool_slot = slot_idx;
  wave->batch_chunk_offset = base;
  wave->n_chunks = take;
  wave->host_used_bytes = host_cursor;
  wave->io_bytes = host_cursor;
  // Tally per-chunk byte totals for the later stages.
  wave->decomp_in_bytes = 0;
  wave->decomp_out_bytes = 0;
  wave->assemble_out_bytes = 0;
  for (uint32_t i = 0; i < take; ++i) {
    struct chunk_plan* c = &slot->chunk_plans[base + i];
    wave->decomp_in_bytes += c->compressed_nbytes;
    wave->decomp_out_bytes += c->decompressed_nbytes;
  }
  wave->state = WAVE_IO;
  slot->n_chunks_dispatched += take;
  self->stats.waves_emitted++;
  self->stats.chunks_dispatched += take;
  return DAMACY_OK;
}

// host_slab and dev_compressed share offsets because kick_h2d copies
// the slab byte-for-byte.
static void
build_blosc1_host_chunks(struct damacy* self, struct damacy_wave* wave)
{
  struct damacy_batch_slot* slot =
    &self->batch_pool.slots[wave->batch_pool_slot];
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    struct chunk_plan* c = &slot->chunk_plans[wave->batch_chunk_offset + i];
    struct read_op* r = &slot->read_ops[wave->batch_chunk_offset + i];
    struct blosc1_host_chunk* hc = &wave->h_chunks[i];
    size_t base_off = (size_t)r->dst_buf_offset + (size_t)c->offset_in_read;
    hc->h_compressed = (const uint8_t*)wave->host_slab + base_off;
    hc->d_compressed = (uint8_t*)wave->dev_compressed + base_off;
    hc->d_decompressed =
      (uint8_t*)wave->dev_decompressed + c->dev_decompressed_offset;
    hc->compressed_nbytes = c->compressed_nbytes;
    hc->decompressed_nbytes = c->decompressed_nbytes;
    hc->codec_id = c->codec_id;
  }
}

// Walk this wave's per-chunk hdrs and emit one log line per failing
// chunk so the user can map errors back to a specific sample. The
// blosc1_host layer logs only the count; this fills in sample_idx +
// chunk_d coords + the err-code string.
static void
log_blosc1_parse_errors(struct damacy* self, struct damacy_wave* wave)
{
  struct damacy_batch_slot* slot =
    &self->batch_pool.slots[wave->batch_pool_slot];
  struct strbuf coords = { 0 };
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    const struct blosc1_chunk_hdr* h = &wave->scratch.hdrs[i];
    if (h->err == 0)
      continue;
    const struct chunk_plan* c =
      &slot->chunk_plans[wave->batch_chunk_offset + i];
    const struct sample_plan* sp = &slot->sample_plans[c->sample_idx_in_batch];
    strbuf_reset(&coords);
    strbuf_append_cstr(&coords, "[");
    for (uint8_t d = 0; d < sp->rank; ++d)
      strbuf_appendf(&coords, d == 0 ? "%u" : ",%u", c->chunk_d[d]);
    strbuf_append_cstr(&coords, "]");
    log_error("blosc1: parse failed: batch_id=%llu sample=%u chunk_d=%s "
              "codec_id=%u err=%u (%s)",
              (unsigned long long)slot->batch_id,
              (unsigned)c->sample_idx_in_batch,
              strbuf_cstr(&coords),
              (unsigned)c->codec_id,
              (unsigned)h->err,
              blosc1_host_parse_err_str(h->err));
  }
  strbuf_free(&coords);
}

// Bulk H2D, host parse overlapping the DMA, fanout/op H2Ds, then
// h2d_end. Codec streams + stream_compute gate on h2d_end.
static enum damacy_status
kick_h2d(struct damacy* self, struct damacy_wave* wave)
{
  CR(CudaFail, cuEventRecord(wave->ev.h2d_start, self->stream_h2d));
  CR(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->dev_compressed),
                       wave->host_slab,
                       wave->host_used_bytes,
                       self->stream_h2d));
  // Record bulk_h2d_end before queueing fanout/op H2Ds so stats.h2d
  // measures just the slab copy. The event may include a stream-idle
  // gap if the host parse outruns the bulk copy, but no extra ops are
  // folded in.
  CR(CudaFail, cuEventRecord(wave->ev.bulk_h2d_end, self->stream_h2d));

  build_blosc1_host_chunks(self, wave);

  uint64_t parse_t0 = monotonic_ns();
  int rc = blosc1_host_parse(&(struct blosc1_host_parse_args){
    .pool = self->compute_pool,
    .chunks = wave->h_chunks,
    .n_chunks = wave->n_chunks,
    .scratch = wave->scratch,
    .zstd = wave->h_zstd_fan,
    .lz4 = wave->h_lz4_fan,
    .memcpy_ops = wave->h_memcpy_ops,
    .unshuffle_ops = wave->h_unshuffle_ops,
    .bitunshuffle_ops = wave->h_bitunshuffle_ops,
    .out_totals = wave->h_blosc1_totals,
  });
  wave->parse_ms = (float)((monotonic_ns() - parse_t0) / 1.0e6);
  if (rc) {
    log_blosc1_parse_errors(self, wave);
    // Record h2d_end so cleanup paths gated on it can drain.
    cuEventRecord(wave->ev.h2d_end, self->stream_h2d);
    self->failed_status = DAMACY_DECODE;
    return DAMACY_DECODE;
  }

  const struct blosc1_totals* tot = wave->h_blosc1_totals;
  if (tot->n_zstd > 0) {
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->zstd_fan.d_comp_ptrs),
                         wave->h_zstd_fan.comp_ptrs,
                         (size_t)tot->n_zstd * sizeof(void*),
                         self->stream_h2d));
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->zstd_fan.d_comp_sizes),
                         wave->h_zstd_fan.comp_sizes,
                         (size_t)tot->n_zstd * sizeof(size_t),
                         self->stream_h2d));
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->zstd_fan.d_decomp_ptrs),
                         wave->h_zstd_fan.decomp_ptrs,
                         (size_t)tot->n_zstd * sizeof(void*),
                         self->stream_h2d));
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->zstd_fan.d_decomp_buf_sizes),
                         wave->h_zstd_fan.decomp_buf_sizes,
                         (size_t)tot->n_zstd * sizeof(size_t),
                         self->stream_h2d));
  }
  if (tot->n_lz4 > 0) {
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->lz4_fan.d_comp_ptrs),
                         wave->h_lz4_fan.comp_ptrs,
                         (size_t)tot->n_lz4 * sizeof(void*),
                         self->stream_h2d));
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->lz4_fan.d_comp_sizes),
                         wave->h_lz4_fan.comp_sizes,
                         (size_t)tot->n_lz4 * sizeof(size_t),
                         self->stream_h2d));
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->lz4_fan.d_decomp_ptrs),
                         wave->h_lz4_fan.decomp_ptrs,
                         (size_t)tot->n_lz4 * sizeof(void*),
                         self->stream_h2d));
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->lz4_fan.d_decomp_buf_sizes),
                         wave->h_lz4_fan.decomp_buf_sizes,
                         (size_t)tot->n_lz4 * sizeof(size_t),
                         self->stream_h2d));
  }
  if (tot->n_memcpy > 0)
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_memcpy_ops),
                         wave->h_memcpy_ops,
                         (size_t)tot->n_memcpy * sizeof(struct gpu_memcpy_op),
                         self->stream_h2d));
  if (tot->n_unshuffle > 0)
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_unshuffle_ops),
                         wave->h_unshuffle_ops,
                         (size_t)tot->n_unshuffle *
                           sizeof(struct gpu_shuffle_op),
                         self->stream_h2d));
  if (tot->n_bitunshuffle > 0)
    CR(CudaFail,
       cuMemcpyHtoDAsync(CUDPTR(wave->d_bitunshuffle_ops),
                         wave->h_bitunshuffle_ops,
                         (size_t)tot->n_bitunshuffle *
                           sizeof(struct gpu_shuffle_op),
                         self->stream_h2d));

  // Zero so status_reduce's atomicAdds land in a clean n_codec_errors.
  CR(CudaFail,
     cuMemsetD8Async(CUDPTR(wave->d_blosc1_totals),
                     0,
                     sizeof(struct blosc1_totals),
                     self->stream_h2d));

  CR(CudaFail, cuEventRecord(wave->ev.h2d_end, self->stream_h2d));
  wave->state = WAVE_H2D;
  return DAMACY_OK;
CudaFail:
  self->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
}

// Build per-wave-chunk assemble metadata. For each chunk in the wave,
// emit the {src_base_byte_off, sample_idx_in_batch, chunk_d} record
// the kernel needs. Also computes max_blocks_per_chunk over the wave
// for grid sizing and accumulates assemble.output_bytes (effective
// in-AABB voxels per chunk × bpe).
//
// Sets wave->assemble_max_blocks_per_chunk and wave->assemble_rank.
// The sample_plans for the batch slot are uploaded once at plan time
// and shared across all waves of that batch.
static void
build_assemble_meta(struct damacy* self, struct damacy_wave* wave)
{
  struct damacy_batch_slot* slot =
    &self->batch_pool.slots[wave->batch_pool_slot];
  uint32_t bpe = damacy_dtype_bpe(self->cfg.dtype);
  uint8_t spatial_rank = (uint8_t)(self->batch_pool.rank - 1);
  uint32_t max_bpc = 0;
  wave->assemble_rank = spatial_rank;
  for (uint32_t i = 0; i < wave->n_chunks; ++i) {
    struct chunk_plan* c = &slot->chunk_plans[wave->batch_chunk_offset + i];
    struct assemble_chunk* a = &wave->h_assemble_chunks[i];
    a->src_base_byte_off = (uint64_t)c->dev_decompressed_offset;
    a->sample_idx_in_batch = c->sample_idx_in_batch;
    for (uint8_t d = 0; d < spatial_rank; ++d)
      a->chunk_d[d] = c->chunk_d[d];

    const struct sample_plan* sp = &slot->sample_plans[c->sample_idx_in_batch];
    uint32_t bpc = assemble_blocks_per_chunk(spatial_rank, sp->dims);
    if (bpc > max_bpc)
      max_bpc = bpc;

    // Effective in-AABB extent for this chunk along each axis. The
    // chunk's footprint in sample-local coords is
    //   [chunk_d[d]*S - aabb_lo_relative,
    //    chunk_d[d]*S + S - aabb_lo_relative)
    // intersected with [0, aabb_extent[d]).
    uint64_t eff = 1;
    for (uint8_t d = 0; d < spatial_rank; ++d) {
      int64_t S = (int64_t)sp->dims[d].chunk_shape;
      int64_t origin_in_sample =
        (int64_t)c->chunk_d[d] * S - sp->dims[d].aabb_lo_relative;
      int64_t lo = origin_in_sample > 0 ? origin_in_sample : 0;
      int64_t hi = origin_in_sample + S < sp->dims[d].aabb_extent
                     ? origin_in_sample + S
                     : sp->dims[d].aabb_extent;
      int64_t extent = hi > lo ? hi - lo : 0;
      eff *= (uint64_t)extent;
    }
    wave->assemble_out_bytes += eff * (uint64_t)bpe;
  }
  if (max_bpc == 0)
    max_bpc = 1;
  wave->assemble_max_blocks_per_chunk = max_bpc;
}

// Parallel Zstd / LZ4 nvcomp batches on dedicated streams; each waits
// on h2d_end before launching and records its *_done event. After each
// batch a small reduce kernel sums non-zero nvcomp statuses into
// d_blosc1_totals->n_codec_errors; the host reads this via a D2H at
// the end of post_decode and surfaces it at finalize time.
static enum damacy_status
kick_codec_batches(struct damacy* self,
                   struct damacy_wave* wave,
                   const struct blosc1_totals* tot)
{
  uint32_t* d_err = &wave->d_blosc1_totals->n_codec_errors;
  if (tot->n_zstd > 0) {
    CR(CudaFail, cuStreamWaitEvent(self->stream_zstd, wave->ev.h2d_end, 0));
    if (decoder_zstd_batch_device(wave->zstd_decoder,
                                  self->stream_zstd,
                                  wave->zstd_fan.d_comp_ptrs,
                                  wave->zstd_fan.d_comp_sizes,
                                  wave->zstd_fan.d_decomp_ptrs,
                                  wave->zstd_fan.d_decomp_buf_sizes,
                                  tot->n_zstd))
      goto DecodeFail;
    if (decoder_status_reduce_launch(
          self->stream_zstd,
          decoder_zstd_d_statuses(wave->zstd_decoder),
          d_err,
          tot->n_zstd))
      goto DecodeFail;
    CR(CudaFail, cuEventRecord(wave->ev.zstd_done, self->stream_zstd));
  }
  if (tot->n_lz4 > 0) {
    CR(CudaFail, cuStreamWaitEvent(self->stream_lz4, wave->ev.h2d_end, 0));
    if (decoder_lz4_batch_device(wave->lz4_decoder,
                                 self->stream_lz4,
                                 wave->lz4_fan.d_comp_ptrs,
                                 wave->lz4_fan.d_comp_sizes,
                                 wave->lz4_fan.d_decomp_ptrs,
                                 wave->lz4_fan.d_decomp_buf_sizes,
                                 tot->n_lz4))
      goto DecodeFail;
    if (decoder_status_reduce_launch(self->stream_lz4,
                                     decoder_lz4_d_statuses(wave->lz4_decoder),
                                     d_err,
                                     tot->n_lz4))
      goto DecodeFail;
    CR(CudaFail, cuEventRecord(wave->ev.lz4_done, self->stream_lz4));
  }
  return DAMACY_OK;
DecodeFail:
  self->failed_status = DAMACY_DECODE;
  return DAMACY_DECODE;
CudaFail:
  self->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
}

// Re-join codec streams onto stream_compute, then run CODEC_NONE /
// chunk-MEMCPYED bulk copies and the (bit)unshuffle filters.
static enum damacy_status
kick_post_decode(struct damacy* self,
                 struct damacy_wave* wave,
                 CUstream s,
                 const struct blosc1_totals* tot)
{
  if (tot->n_zstd > 0)
    CR(CudaFail, cuStreamWaitEvent(s, wave->ev.zstd_done, 0));
  if (tot->n_lz4 > 0)
    CR(CudaFail, cuStreamWaitEvent(s, wave->ev.lz4_done, 0));
  CR(CudaFail, cuEventRecord(wave->ev.post_start, s));

  if (tot->n_memcpy > 0 &&
      decoder_memcpy_launch(s, wave->d_memcpy_ops, tot->n_memcpy))
    goto DecodeFail;
  if (tot->n_unshuffle > 0 && gpu_unshuffle_launch(s,
                                                   wave->d_unshuffle_ops,
                                                   tot->n_unshuffle,
                                                   wave->dev_decompressed,
                                                   wave->dev_unshuffle_scratch))
    goto DecodeFail;
  if (tot->n_bitunshuffle > 0 &&
      gpu_bitunshuffle_launch(s,
                              wave->d_bitunshuffle_ops,
                              tot->n_bitunshuffle,
                              wave->dev_decompressed,
                              wave->dev_unshuffle_scratch))
    goto DecodeFail;
  // Narrowed to the 4-byte n_codec_errors so the host parse's count
  // fields in h_blosc1_totals stay intact for drain_wave_metrics.
  CR(CudaFail,
     cuMemcpyDtoHAsync(&wave->h_blosc1_totals->n_codec_errors,
                       CUDPTR(&wave->d_blosc1_totals->n_codec_errors),
                       sizeof(uint32_t),
                       s));
  CR(CudaFail, cuEventRecord(wave->ev.decomp_end, s));
  return DAMACY_OK;
DecodeFail:
  self->failed_status = DAMACY_DECODE;
  return DAMACY_DECODE;
CudaFail:
  self->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
}

// H2D meta for the assemble stage and the assemble kernel itself.
static enum damacy_status
kick_assemble(struct damacy* self, struct damacy_wave* wave, CUstream s)
{
  struct damacy_batch_slot* slot =
    &self->batch_pool.slots[wave->batch_pool_slot];

  build_assemble_meta(self, wave);
  CR(CudaFail,
     cuMemcpyHtoDAsync(CUDPTR(wave->d_assemble_chunks),
                       wave->h_assemble_chunks,
                       (size_t)wave->n_chunks * sizeof(struct assemble_chunk),
                       s));
  CR(CudaFail, cuEventRecord(wave->ev.asm_start, s));
  if (assemble_launch(s,
                      wave->assemble_rank,
                      (const struct sample_plan*)slot->d_sample_plans,
                      slot->n_sample_plans,
                      wave->d_assemble_chunks,
                      wave->n_chunks,
                      wave->assemble_max_blocks_per_chunk,
                      wave->dev_decompressed,
                      slot->dev_ptr,
                      self->cfg.dtype)) {
    self->failed_status = DAMACY_CUDA;
    return DAMACY_CUDA;
  }
  CR(CudaFail, cuEventRecord(wave->ev.asm_end, s));
  return DAMACY_OK;
CudaFail:
  self->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
}

// Codec batches on parallel streams gate on h2d_end; memcpy +
// (un)shuffles fold back onto stream_compute; then assemble.
static enum damacy_status
kick_compute(struct damacy* self, struct damacy_wave* wave)
{
  CUstream s = self->stream_compute;
  CR(CudaFail, cuStreamWaitEvent(s, wave->ev.h2d_end, 0));
  CR(CudaFail, cuEventRecord(wave->ev.decomp_start, s));

  const struct blosc1_totals tot = *wave->h_blosc1_totals;
  enum damacy_status st = kick_codec_batches(self, wave, &tot);
  if (st != DAMACY_OK)
    return st;
  st = kick_post_decode(self, wave, s, &tot);
  if (st != DAMACY_OK)
    return st;
  st = kick_assemble(self, wave, s);
  if (st != DAMACY_OK)
    return st;

  wave->state = WAVE_ASSEMBLE;
  return DAMACY_OK;
CudaFail:
  self->failed_status = DAMACY_CUDA;
  return DAMACY_CUDA;
}

// All wave events have fired (asm_end signaled implies everything
// earlier on the same stream did too, and we cuStreamWaitEvent'd
// h2d_end before kicking compute). Pull the elapsed times into stats.
static void
drain_wave_metrics(struct damacy* self, struct damacy_wave* wave)
{
  // IO is host-side; ns → ms.
  float io_ms = (float)((wave->io_t_end_ns - wave->io_t_start_ns) / 1.0e6);
  metric_record(&self->stats.io, io_ms, wave->io_bytes, wave->io_bytes);

  float ms = 0.f;
  if (cuEventElapsedTime(&ms, wave->ev.h2d_start, wave->ev.bulk_h2d_end) ==
      CUDA_SUCCESS)
    metric_record(&self->stats.h2d, ms, wave->io_bytes, wave->io_bytes);
  if (cuEventElapsedTime(&ms, wave->ev.decomp_start, wave->ev.decomp_end) ==
      CUDA_SUCCESS)
    metric_record(&self->stats.decompress,
                  ms,
                  wave->decomp_in_bytes,
                  wave->decomp_out_bytes);
  metric_record(&self->stats.decompress_parse, wave->parse_ms, 0, 0);
  const struct blosc1_totals tot = *wave->h_blosc1_totals;
  if (tot.n_zstd > 0 &&
      cuEventElapsedTime(&ms, wave->ev.h2d_end, wave->ev.zstd_done) ==
        CUDA_SUCCESS)
    metric_record(&self->stats.decompress_zstd, ms, 0, 0);
  if (tot.n_lz4 > 0 &&
      cuEventElapsedTime(&ms, wave->ev.h2d_end, wave->ev.lz4_done) ==
        CUDA_SUCCESS)
    metric_record(&self->stats.decompress_lz4, ms, 0, 0);
  if (cuEventElapsedTime(&ms, wave->ev.post_start, wave->ev.decomp_end) ==
      CUDA_SUCCESS)
    metric_record(&self->stats.decompress_post, ms, 0, 0);
  if (cuEventElapsedTime(&ms, wave->ev.asm_start, wave->ev.asm_end) ==
      CUDA_SUCCESS)
    metric_record(&self->stats.assemble,
                  ms,
                  wave->decomp_out_bytes,
                  wave->assemble_out_bytes);
}

// asm_end signaled — finalize this wave: drain timings, decrement the
// batch slot's chunks_remaining, mark slot READY when zero, free the
// wave. If the post-decode D2H surfaced any nvcomp status errors, set
// failed_status before the slot transitions; damacy_pop's per-iteration
// failed_status check then bails before handing the corrupt batch out.
static void
finalize_wave(struct damacy* self, struct damacy_wave* wave)
{
  drain_wave_metrics(self, wave);
  if (wave->h_blosc1_totals->n_codec_errors > 0 &&
      self->failed_status == DAMACY_OK) {
    log_error("nvcomp: %u substream(s) reported non-success status",
              wave->h_blosc1_totals->n_codec_errors);
    self->failed_status = DAMACY_DECODE;
  }
  struct damacy_batch_slot* slot =
    &self->batch_pool.slots[wave->batch_pool_slot];
  slot->chunks_remaining -= (int32_t)wave->n_chunks;
  if (slot->chunks_remaining <= 0) {
    slot->chunks_remaining = 0;
    if (slot->state == BATCH_FILLING)
      slot->state = BATCH_READY;
  }
  wave->state = WAVE_FREE;
  wave->n_chunks = 0;
}

// Phase 1: poll each in-flight wave; advance state when its current
// stage's event has retired.
static enum damacy_status
advance_waves(struct damacy* self)
{
  for (int w = 0; w < 2; ++w) {
    struct damacy_wave* wave = &self->waves[w];
    switch (wave->state) {
      case WAVE_FREE:
        break;
      case WAVE_IO:
        if (store_event_query(self->store, wave->io_event)) {
          wave->io_t_end_ns = monotonic_ns();
          enum damacy_status s = kick_h2d(self, wave);
          if (s != DAMACY_OK)
            return s;
        }
        break;
      case WAVE_H2D: {
        CUresult qe = cuEventQuery(wave->ev.h2d_end);
        if (qe == CUDA_SUCCESS) {
          enum damacy_status s = kick_compute(self, wave);
          if (s != DAMACY_OK)
            return s;
        } else if (qe != CUDA_ERROR_NOT_READY) {
          self->failed_status = DAMACY_CUDA;
          return DAMACY_CUDA;
        }
      } break;
      case WAVE_ASSEMBLE: {
        CUresult qe = cuEventQuery(wave->ev.asm_end);
        if (qe == CUDA_SUCCESS) {
          finalize_wave(self, wave);
        } else if (qe != CUDA_ERROR_NOT_READY) {
          self->failed_status = DAMACY_CUDA;
          return DAMACY_CUDA;
        }
      } break;
    }
  }
  return DAMACY_OK;
}

// Phase 2: kick new work into FREE wave slots until either both are
// busy or there's nothing to do.
static enum damacy_status
kick_new_waves(struct damacy* self)
{
  for (;;) {
    int w = find_free_wave(self);
    if (w < 0)
      break;

    // Find a batch slot with chunks left to dispatch.
    int target_slot = -1;
    uint64_t oldest = UINT64_MAX;
    for (int s = 0; s < 2; ++s) {
      struct damacy_batch_slot* slot = &self->batch_pool.slots[s];
      if (slot->state == BATCH_FILLING &&
          slot->n_chunks_dispatched < slot->n_chunks &&
          slot->batch_id < oldest) {
        target_slot = s;
        oldest = slot->batch_id;
      }
    }
    if (target_slot < 0) {
      // No FILLING slot has unfinished chunks. Try planning a new batch.
      int free_slot = find_free_batch_slot(self);
      if (free_slot < 0)
        break; // both slots taken
      if (self->lookahead.size < self->cfg.batch_size)
        break; // no full batch to plan
      enum damacy_status s =
        plan_into_slot(self, (uint16_t)free_slot, self->cfg.batch_size);
      if (s != DAMACY_OK)
        return s;
      // If the planned batch had chunks, retry the wave-pick on the
      // next loop iteration. If it was degenerate (already READY),
      // continue the kick loop to look for more work.
      continue;
    }

    enum damacy_status s = peel_wave(self, (uint16_t)w, (uint16_t)target_slot);
    if (s != DAMACY_OK)
      return s;
  }
  return DAMACY_OK;
}

// --- public API: create / destroy ----------------------------------------

enum damacy_status
damacy_create(const struct damacy_config* cfg, struct damacy** out)
{
  enum damacy_status s = DAMACY_INVAL;
  struct damacy* self = NULL;

  CHECK_SILENT(InvalidArg, out);
  *out = NULL;

  s = validate_config(cfg);
  if (s != DAMACY_OK)
    return s;

  s = DAMACY_OOM;
  self = (struct damacy*)calloc(1, sizeof(*self));
  CHECK(Fail, self);
  self->cfg = *cfg;
  self->failed_status = DAMACY_OK;
  self->page_alignment = (uint64_t)platform_page_alignment();
  stats_init(&self->stats);

  s = DAMACY_CUDA;
  CR(Fail, cuInit(0));

  self->retained_primary_device = -1;
  CUcontext caller_ctx = NULL;
  CR(Fail, cuCtxGetCurrent(&caller_ctx));
  if (cfg->device >= 0) {
    if (caller_ctx) {
      CUdevice cur_dev;
      CR(Fail, cuCtxGetDevice(&cur_dev));
      if ((int)cur_dev != cfg->device) {
        s = DAMACY_INVAL;
        log_error("damacy_create: Config.device=%d but a CUcontext is "
                  "already current on device %d — likely a missing "
                  "cuCtxSetCurrent / torch.cuda.set_device(%d)",
                  cfg->device,
                  (int)cur_dev,
                  cfg->device);
        goto Fail;
      }
    }
    CUdevice dev;
    CR(Fail, cuDeviceGet(&dev, cfg->device));
    CUcontext primary = NULL;
    CR(Fail, cuDevicePrimaryCtxRetain(&primary, dev));
    self->retained_primary_device = cfg->device;
    CR(Fail, cuCtxPushCurrent(primary));
    self->cuda_device = cfg->device;
  } else {
    if (!caller_ctx) {
      log_error("damacy_create: no CUcontext is current on calling thread");
      s = DAMACY_INVAL;
      goto Fail;
    }
    CUdevice dev;
    CR(Fail, cuCtxGetDevice(&dev));
    self->cuda_device = (int)dev;
  }
  // Non-blocking so damacy doesn't force-serialize against the legacy
  // default stream that some user code still lands on.
  CR(Fail, cuStreamCreate(&self->stream_h2d, CU_STREAM_NON_BLOCKING));
  CR(Fail, cuStreamCreate(&self->stream_compute, CU_STREAM_NON_BLOCKING));
  CR(Fail, cuStreamCreate(&self->stream_zstd, CU_STREAM_NON_BLOCKING));
  CR(Fail, cuStreamCreate(&self->stream_lz4, CU_STREAM_NON_BLOCKING));

  self->compute_pool = threadpool_new((int)cfg->n_compute_threads);
  if (!self->compute_pool) {
    s = DAMACY_OOM;
    goto Fail;
  }

  // Predict wave-resident GPU bytes and reject early if over budget.
  // Batch-output tensors (sized from the first AABB) are checked
  // separately at batch_pool_allocate.
  self->gpu_bytes_budget = cfg->max_gpu_memory_bytes;
  {
    struct gpu_budget budget = { 0 };
    s = gpu_budget_compute(cfg, &budget);
    if (s != DAMACY_OK)
      goto Fail;
    self->gpu_bytes_committed = budget.total;
    if (self->gpu_bytes_budget > 0 &&
        self->gpu_bytes_committed > self->gpu_bytes_budget) {
      log_error(
        "damacy: GPU budget exceeded at create: total=%llu cap=%llu "
        "(dev_compressed=%llu dev_decompressed=%llu unshuffle_scratch=%llu "
        "blosc1_meta=%llu fanout_soa=%llu nvcomp_temp=%llu batch_meta=%llu)",
        (unsigned long long)budget.total,
        (unsigned long long)self->gpu_bytes_budget,
        (unsigned long long)budget.dev_compressed,
        (unsigned long long)budget.dev_decompressed,
        (unsigned long long)budget.dev_unshuffle_scratch,
        (unsigned long long)budget.blosc1_meta,
        (unsigned long long)budget.fanout_soa,
        (unsigned long long)budget.nvcomp_temp,
        (unsigned long long)budget.batch_metadata);
      s = DAMACY_OOM;
      goto Fail;
    }
  }

  s = DAMACY_OOM;
  // Sample.uri is absolute; the fs store joins root+key, so empty root
  // turns join into a pass-through.
  struct store_fs_config sc = {
    .root = "",
    .nthreads = (int)cfg->n_io_threads,
  };
  self->store = store_fs_create(&sc);
  CHECK(Fail, self->store);

  self->meta_cache =
    zarr_meta_cache_create(self->store, cfg->n_zarrs_meta_cache);
  CHECK(Fail, self->meta_cache);
  self->shard_cache =
    zarr_shard_cache_create(self->store, cfg->n_shards_meta_cache);
  CHECK(Fail, self->shard_cache);

  const uint64_t runtime_chunk_cap = resolve_max_chunk_uncompressed(cfg);
  struct planner_config pcfg = {
    .meta_cache = self->meta_cache,
    .shard_cache = self->shard_cache,
    .page_alignment = self->page_alignment,
    .max_chunk_uncompressed_bytes = runtime_chunk_cap,
  };
  CHECK(Fail, planner_create(&pcfg, &self->planner) == DAMACY_OK);

  for (int b = 0; b < 2; ++b)
    CHECK(Fail,
          batch_slot_init(&self->batch_pool.slots[b], cfg->batch_size) == 0);

  uint64_t host_per_wave = cfg->host_buffer_bytes / 2;
  uint64_t dev_per_wave = cfg->device_buffer_bytes / 2;
  s = DAMACY_INVAL;
  CHECK(Fail, host_per_wave > 0 && dev_per_wave > 0);
  s = DAMACY_OOM;
  const uint8_t max_bpe = resolve_max_bpe(cfg);
  for (int w = 0; w < 2; ++w)
    CHECK(Fail,
          wave_init(&self->waves[w],
                    host_per_wave,
                    dev_per_wave,
                    max_bpe,
                    runtime_chunk_cap) == 0);

  CHECK(Fail,
        lookahead_init(&self->lookahead,
                       cfg->lookahead_batches * cfg->batch_size) == 0);

  self->batch_samples = (struct damacy_sample_slot*)calloc(
    cfg->batch_size, sizeof(struct damacy_sample_slot));
  CHECK(Fail, self->batch_samples);
  self->batch_stage = (struct damacy_sample*)calloc(
    cfg->batch_size, sizeof(struct damacy_sample));
  CHECK(Fail, self->batch_stage);

  self->handle.d = self;

  *out = self;
  return DAMACY_OK;

Fail:
  damacy_destroy(self);
  return s;

InvalidArg:
  return DAMACY_INVAL;
}

void
damacy_destroy(struct damacy* self)
{
  if (!self)
    return;
  // The four owned streams. NULL/0 is the legacy default stream — sync
  // and destroy on it would either be a no-op-on-the-wrong-thing or
  // outright invalid, so the loop's single guard is structural, not
  // defensive: it distinguishes "we created this stream" from "this
  // field never got set in a partially-failed create".
  CUstream* const streams[] = {
    &self->stream_compute,
    &self->stream_h2d,
    &self->stream_zstd,
    &self->stream_lz4,
  };
  for (size_t i = 0; i < countof(streams); ++i)
    if (*streams[i]) {
      cuStreamSynchronize(*streams[i]);
      cuStreamDestroy(*streams[i]);
    }

  threadpool_free(self->compute_pool);

  free(self->batch_stage);
  free(self->batch_samples);
  lookahead_destroy(&self->lookahead);
  for (int w = 0; w < 2; ++w)
    wave_destroy(&self->waves[w]);
  batch_pool_destroy(&self->batch_pool);

  planner_destroy(self->planner);
  zarr_shard_cache_destroy(self->shard_cache);
  zarr_meta_cache_destroy(self->meta_cache);
  store_destroy(self->store);

  if (self->retained_primary_device >= 0) {
    cuCtxPopCurrent(NULL);
    cuDevicePrimaryCtxRelease((CUdevice)self->retained_primary_device);
  }
  free(self);
}

int
damacy_get_device(const struct damacy* d)
{
  return d ? d->cuda_device : -1;
}

// --- push -----------------------------------------------------------------

struct damacy_push_result
damacy_push(struct damacy* self, struct damacy_sample_slice samples)
{
  struct damacy_push_result r = { .unconsumed = samples, .status = DAMACY_OK };
  if (!self) {
    r.status = DAMACY_INVAL;
    return r;
  }
  if (self->failed_status != DAMACY_OK) {
    r.status = DAMACY_SHUTDOWN;
    return r;
  }
  if (samples.beg > samples.end) {
    r.status = DAMACY_INVAL;
    return r;
  }
  for (const struct damacy_sample* s = samples.beg; s != samples.end; ++s) {
    if (self->lookahead.size == self->lookahead.cap) {
      r.unconsumed.beg = s;
      r.status = DAMACY_AGAIN;
      return r;
    }
    enum damacy_status ps = push_one(self, s);
    if (ps != DAMACY_OK) {
      r.unconsumed.beg = s;
      r.status = ps;
      return r;
    }
  }
  r.unconsumed.beg = samples.end;
  return r;
}

// --- pop ------------------------------------------------------------------

enum damacy_status
damacy_pop(struct damacy* self, struct damacy_batch** out)
{
  CHECK_SILENT(InvalidArg, self);
  CHECK_SILENT(InvalidArg, out);
  *out = NULL;
  if (self->failed_status != DAMACY_OK)
    return self->failed_status;

  for (;;) {
    enum damacy_status s = advance_waves(self);
    if (s != DAMACY_OK)
      return s;
    // finalize_wave can set failed_status on a post-decode codec error
    // even when advance_waves itself returned OK; bail before handing
    // out a possibly-corrupt batch.
    if (self->failed_status != DAMACY_OK)
      return self->failed_status;
    s = kick_new_waves(self);
    if (s != DAMACY_OK)
      return s;

    int slot_idx = find_oldest_ready_slot(self);
    if (slot_idx >= 0) {
      struct damacy_batch_slot* slot = &self->batch_pool.slots[slot_idx];
      slot->state = BATCH_HELD;
      self->handle.slot_idx = (uint16_t)slot_idx;
      self->handle.batch_id = slot->batch_id;
      self->stats.batches_emitted++;
      *out = &self->handle;
      return DAMACY_OK;
    }
    if (!any_wave_in_flight(self) && !any_batch_in_flight(self) &&
        self->lookahead.size < self->cfg.batch_size)
      return DAMACY_AGAIN;
    // Attribute the upcoming poll to whichever stage we're blocked on.
    // Prefer compute when any wave has reached H2D/ASSEMBLE — that's
    // what we'd most likely be waiting on; fall back to IO otherwise.
    int waiting_compute = 0;
    for (int w = 0; w < 2; ++w) {
      enum wave_state ws = self->waves[w].state;
      if (ws == WAVE_H2D || ws == WAVE_ASSEMBLE) {
        waiting_compute = 1;
        break;
      }
    }
    uint64_t poll_t0 = monotonic_ns();
    platform_sleep_ns(DAMACY_POP_POLL_NS);
    float poll_ms = (float)((monotonic_ns() - poll_t0) / 1.0e6);
    metric_record(waiting_compute ? &self->stats.pop_wait_compute
                                  : &self->stats.pop_wait_io,
                  poll_ms,
                  0,
                  0);
  }

InvalidArg:
  return DAMACY_INVAL;
}

void
damacy_release(struct damacy* self, struct damacy_batch* b)
{
  if (!self || !b || b != &self->handle)
    return;
  uint16_t s = b->slot_idx;
  if (s >= 2)
    return;
  if (self->batch_pool.slots[s].state != BATCH_HELD)
    return;
  self->batch_pool.slots[s].state = BATCH_FREE;
  self->batch_pool.slots[s].n_chunks = 0;
  self->batch_pool.slots[s].n_chunks_dispatched = 0;
}

// --- flush ----------------------------------------------------------------

enum damacy_status
damacy_flush(struct damacy* self)
{
  if (!self)
    return DAMACY_INVAL;
  if (self->failed_status != DAMACY_OK)
    return self->failed_status;

  // Plan a partial batch from the remaining lookahead, if any.
  if (self->lookahead.size > 0 && self->lookahead.size < self->cfg.batch_size) {
    int free_slot = find_free_batch_slot(self);
    if (free_slot < 0) {
      // Both slots in use; drain one wave-cycle's worth and try again.
      // For step 5 we allow one drain pass; if still no slot, return AGAIN.
      enum damacy_status s = advance_waves(self);
      if (s != DAMACY_OK)
        return s;
      free_slot = find_free_batch_slot(self);
      if (free_slot < 0)
        return DAMACY_AGAIN;
    }
    uint32_t n = self->lookahead.size;
    enum damacy_status s = plan_into_slot(self, (uint16_t)free_slot, n);
    if (s != DAMACY_OK)
      return s;
    self->stats.batches_truncated++;
  }

  // Drain everything in flight by spinning the scheduler until no
  // FILLING slots remain.
  uint64_t flush_t0 = monotonic_ns();
  while (any_wave_in_flight(self) || find_oldest_filling_slot(self) >= 0) {
    enum damacy_status s = advance_waves(self);
    if (s != DAMACY_OK)
      return s;
    s = kick_new_waves(self);
    if (s != DAMACY_OK)
      return s;
    if (any_wave_in_flight(self))
      platform_sleep_ns(DAMACY_POP_POLL_NS);
  }
  metric_record(&self->stats.flush_wait,
                (float)((monotonic_ns() - flush_t0) / 1.0e6),
                0,
                0);
  return DAMACY_OK;
}

// --- batch info / stats ---------------------------------------------------

void
damacy_batch_info(const struct damacy_batch* b, struct damacy_batch_info* out)
{
  if (!out)
    return;
  memset(out, 0, sizeof(*out));
  if (!b || !b->d || b->slot_idx >= 2)
    return;
  const struct damacy* self = b->d;
  const struct damacy_batch_slot* slot = &self->batch_pool.slots[b->slot_idx];
  if (slot->state != BATCH_HELD)
    return;
  out->device_ptr = slot->dev_ptr;
  out->rank = self->batch_pool.rank;
  out->dtype = self->cfg.dtype;
  out->ready_stream = (void*)self->stream_compute;
  out->batch_id = slot->batch_id;
  for (uint8_t d = 0; d < self->batch_pool.rank; ++d)
    out->shape[d] = self->batch_pool.shape[d];
  // shape[0] reflects the actual sample count (== n_samples for the
  // batch, which equals cfg.batch_size for full batches and < that
  // for flushed partials).
  out->shape[0] = (int64_t)slot->n_samples;
}

void
damacy_stats_get(const struct damacy* self, struct damacy_stats* out)
{
  if (!out)
    return;
  if (!self) {
    memset(out, 0, sizeof(*out));
    return;
  }
  *out = self->stats;
  if (self->meta_cache) {
    struct zarr_meta_cache_stats ms;
    zarr_meta_cache_stats_get(self->meta_cache, &ms);
    out->zarr_meta_hits = ms.counters.hits;
    out->zarr_meta_misses = ms.counters.misses;
  }
  if (self->shard_cache) {
    struct zarr_shard_cache_stats ss;
    zarr_shard_cache_stats_get(self->shard_cache, &ss);
    out->shard_idx_hits = ss.counters.hits;
    out->shard_idx_misses = ss.counters.misses;
  }
  out->gpu_bytes_committed = self->gpu_bytes_committed;
}

void
damacy_stats_reset(struct damacy* self)
{
  if (!self)
    return;
  stats_init(&self->stats);
}
