// damacy: streaming loader.
//
// Step 4 (current): naive end-to-end. Single wave slot, single batch
// slot, no overlap. damacy_pop drains a full batch synchronously:
// plan → store_read → cuMemcpyHtoDAsync → nvCOMP zstd decompress →
// assemble kernel → cudaStreamSynchronize. Wave scheduling and
// double-buffering land in step 5 (see
// docs/api-design-internals-draft.md).
#include "damacy.h"

#include "assemble.h"
#include "decoder.h"
#include "dtype.h"
#include "log/log.h"
#include "planner.h"
#include "platform/platform.h"
#include "store.h"
#include "util/prelude.h"
#include "zarr_meta_cache.h"
#include "zarr_metadata.h"
#include "zarr_shard_cache.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

// Per-batch hard caps (v1; configurable when we land cfg knobs).
// Max chunks bounds the planner output, the decoder fanout, and the
// assemble metadata buffer; max-uncompressed bounds nvCOMP's temp
// scratch via max_chunks * max_chunk_uncompressed (so the product
// can't be raised casually — it controls a cudaMalloc).
#define DAMACY_MAX_CHUNKS_PER_BATCH 256u
#define DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES (1ull << 20) // 1 MB

#define CK_CUDA(label, expr)                                                   \
  do {                                                                         \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
      log_error("cuda: %s -> %s", #expr, cudaGetErrorString(_e));              \
      goto label;                                                              \
    }                                                                          \
  } while (0)

struct damacy_batch
{
  struct damacy* d;
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

struct damacy_buffers
{
  void* host_slab;        // pinned, page-aligned, cfg.host_buffer_bytes
  void* dev_compressed;   // mirrors host_slab on device
  void* dev_decompressed; // device decompress arena, cfg.device_buffer_bytes
};

struct damacy_plan
{
  struct read_op* read_ops;
  uint32_t read_ops_cap;
  struct chunk_plan* chunk_plans;
  uint32_t chunk_plans_cap;

  // Decoder fanout (host side; decoder owns its own pinned/device copies).
  const void** h_compressed_ptrs;
  size_t* h_compressed_sizes;
  void** h_decompressed_ptrs;
  size_t* h_decompressed_sizes;

  // Assemble metadata staging.
  struct assemble_chunk* h_assemble;
  struct assemble_chunk* d_assemble;

  // store_read[] scratch (one per read_op).
  struct store_read* store_reads;
};

struct damacy_batch_pool
{
  void* dev_ptr;                        // device pointer to output tensor
  uint64_t n_bytes;                     // allocated bytes
  uint8_t rank;                         // includes leading N axis
  int64_t shape[DAMACY_MAX_RANK + 1];   // [N, ...sample_axes]
  int64_t strides[DAMACY_MAX_RANK + 1]; // row-major, in elements
  int allocated;
  int ready; // a batch has been assembled and is awaiting pop
  int held;  // pop returned the handle; release flips this back
  uint64_t batch_id;
};

struct damacy
{
  struct damacy_config cfg;
  char* store_root;
  enum damacy_status failed_status;
  uint64_t next_batch_id;
  uint64_t page_alignment;
  int cuda_device;

  struct store* store;
  struct zarr_meta_cache* meta_cache;
  struct zarr_shard_cache* shard_cache;
  struct planner* planner;
  struct decoder* decoder;
  cudaStream_t stream;

  struct damacy_buffers buffers;
  struct damacy_plan plan;
  struct damacy_lookahead lookahead;
  struct damacy_batch_pool batch;

  // Sample working set used while assembling one batch.
  struct damacy_sample_slot* batch_samples;
  // Mirror of batch_samples in damacy_sample shape, passed to planner.
  struct damacy_sample* batch_stage;

  struct damacy_batch handle;
  struct damacy_stats stats;
};

// --- dtype helpers --------------------------------------------------------

static uint32_t
damacy_dtype_bpe(enum damacy_dtype dt)
{
  switch (dt) {
    case DAMACY_U8:
      return 1;
    case DAMACY_U16:
    case DAMACY_I16:
    case DAMACY_F16:
      return 2;
    case DAMACY_U32:
    case DAMACY_F32:
      return 4;
  }
  return 0;
}

static int
damacy_dtype_match(enum damacy_dtype d, enum dtype z)
{
  switch (d) {
    case DAMACY_U8:
      return z == dtype_u8;
    case DAMACY_U16:
      return z == dtype_u16;
    case DAMACY_I16:
      return z == dtype_i16;
    case DAMACY_U32:
      return z == dtype_u32;
    case DAMACY_F16:
      return z == dtype_f16;
    case DAMACY_F32:
      return z == dtype_f32;
  }
  return 0;
}

// --- config validation ----------------------------------------------------

static enum damacy_status
validate_config(const struct damacy_config* cfg)
{
  CHECK_SILENT(invalid, cfg);
  CHECK_SILENT(invalid, cfg->store_root);
  CHECK_SILENT(invalid, cfg->batch_size > 0);
  CHECK_SILENT(invalid, cfg->lookahead_batches >= 2);
  CHECK_SILENT(invalid, cfg->n_io_threads > 0);
  CHECK_SILENT(invalid, cfg->host_buffer_bytes > 0);
  CHECK_SILENT(invalid, cfg->device_buffer_bytes > 0);
  CHECK_SILENT(invalid, cfg->n_zarrs_meta_cache > 0);
  CHECK_SILENT(invalid, cfg->n_shards_meta_cache > 0);
  CHECK_SILENT(invalid, damacy_dtype_bpe(cfg->dtype) > 0);
  return DAMACY_OK;
invalid:
  return DAMACY_INVAL;
}

// --- lookahead ring -------------------------------------------------------

static void
sample_slot_clear(struct damacy_sample_slot* slot)
{
  free(slot->uri);
  slot->uri = NULL;
  memset(&slot->aabb, 0, sizeof(slot->aabb));
}

static int
lookahead_init(struct damacy_lookahead* la, uint32_t cap)
{
  la->slots =
    (struct damacy_sample_slot*)calloc(cap, sizeof(struct damacy_sample_slot));
  if (!la->slots)
    return 1;
  la->cap = cap;
  la->head = 0;
  la->tail = 0;
  la->size = 0;
  return 0;
}

static void
lookahead_destroy(struct damacy_lookahead* la)
{
  if (!la->slots)
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

// Move `n` slots from the head of `la` into `out` (caller-owned buffer of
// size >= n). Slot ownership transfers; caller must call sample_slot_clear
// on each output slot when done.
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

// --- output buffer setup --------------------------------------------------

// Allocate output device tensor sized for cfg.batch_size × sample volume.
// Sample shape is taken from the first sample's AABB. Subsequent batches
// reuse this allocation.
static enum damacy_status
batch_pool_allocate(struct damacy* self, const struct damacy_aabb* sample_aabb)
{
  if (self->batch.allocated)
    return DAMACY_OK;

  uint8_t spatial_rank = sample_aabb->rank;
  uint8_t full_rank = (uint8_t)(spatial_rank + 1);
  if (full_rank > DAMACY_MAX_RANK + 1)
    return DAMACY_RANK;

  uint32_t bpe = damacy_dtype_bpe(self->cfg.dtype);
  int64_t spatial_volume = 1;
  self->batch.shape[0] = (int64_t)self->cfg.batch_size;
  for (uint8_t d = 0; d < spatial_rank; ++d) {
    int64_t extent = sample_aabb->dims[d].end - sample_aabb->dims[d].beg;
    if (extent <= 0)
      return DAMACY_INVAL;
    self->batch.shape[d + 1] = extent;
    spatial_volume *= extent;
  }
  self->batch.rank = full_rank;

  // Row-major strides in elements.
  self->batch.strides[full_rank - 1] = 1;
  for (int d = (int)full_rank - 2; d >= 0; --d)
    self->batch.strides[d] =
      self->batch.strides[d + 1] * self->batch.shape[d + 1];

  uint64_t n_bytes =
    (uint64_t)self->cfg.batch_size * (uint64_t)spatial_volume * (uint64_t)bpe;
  CK_CUDA(fail, cudaMalloc(&self->batch.dev_ptr, n_bytes));
  self->batch.n_bytes = n_bytes;
  self->batch.allocated = 1;
  return DAMACY_OK;

fail:
  return DAMACY_CUDA;
}

// Validate an aabb has the same per-axis shape as the established batch
// allocation. v1 requires every sample in every batch to share one shape.
static int
sample_shape_matches_batch(const struct damacy* self,
                           const struct damacy_aabb* aabb)
{
  uint8_t spatial_rank = (uint8_t)(self->batch.rank - 1);
  if (aabb->rank != spatial_rank)
    return 0;
  for (uint8_t d = 0; d < spatial_rank; ++d) {
    int64_t extent = aabb->dims[d].end - aabb->dims[d].beg;
    if (extent != self->batch.shape[d + 1])
      return 0;
  }
  return 1;
}

// --- plan-buffer setup ----------------------------------------------------

static int
plan_init(struct damacy_plan* p, uint32_t cap)
{
  p->read_ops_cap = cap;
  p->chunk_plans_cap = cap;
  p->read_ops = (struct read_op*)calloc(cap, sizeof(struct read_op));
  p->chunk_plans = (struct chunk_plan*)calloc(cap, sizeof(struct chunk_plan));
  p->h_compressed_ptrs = (const void**)calloc(cap, sizeof(void*));
  p->h_compressed_sizes = (size_t*)calloc(cap, sizeof(size_t));
  p->h_decompressed_ptrs = (void**)calloc(cap, sizeof(void*));
  p->h_decompressed_sizes = (size_t*)calloc(cap, sizeof(size_t));
  p->h_assemble =
    (struct assemble_chunk*)calloc(cap, sizeof(struct assemble_chunk));
  p->store_reads = (struct store_read*)calloc(cap, sizeof(struct store_read));
  if (!p->read_ops || !p->chunk_plans || !p->h_compressed_ptrs ||
      !p->h_compressed_sizes || !p->h_decompressed_ptrs ||
      !p->h_decompressed_sizes || !p->h_assemble || !p->store_reads)
    return 1;
  if (cudaMalloc((void**)&p->d_assemble, cap * sizeof(struct assemble_chunk)) !=
      cudaSuccess) {
    p->d_assemble = NULL;
    return 1;
  }
  return 0;
}

static void
plan_destroy(struct damacy_plan* p)
{
  free(p->read_ops);
  free(p->chunk_plans);
  free(p->h_compressed_ptrs);
  free(p->h_compressed_sizes);
  free(p->h_decompressed_ptrs);
  free(p->h_decompressed_sizes);
  free(p->h_assemble);
  free(p->store_reads);
  if (p->d_assemble)
    cudaFree(p->d_assemble);
  memset(p, 0, sizeof(*p));
}

// --- create / destroy -----------------------------------------------------

enum damacy_status
damacy_create(const struct damacy_config* cfg, struct damacy** out)
{
  CHECK_SILENT(invalid_arg, out);
  *out = NULL;

  enum damacy_status s = validate_config(cfg);
  if (s != DAMACY_OK)
    return s;

  struct damacy* self = (struct damacy*)calloc(1, sizeof(*self));
  if (!self)
    return DAMACY_OOM;
  self->cfg = *cfg;
  self->failed_status = DAMACY_OK;
  self->page_alignment = (uint64_t)platform_page_alignment();

  self->store_root = strdup(cfg->store_root);
  if (!self->store_root)
    goto oom;

  // Capture current device. Driver-level CUcontext binding can land in
  // step 5; runtime-API setDevice is sufficient for v1.
  if (cudaGetDevice(&self->cuda_device) != cudaSuccess)
    goto cuda_fail;

  CK_CUDA(cuda_fail, cudaStreamCreate(&self->stream));
  CK_CUDA(cuda_fail,
          cudaMallocHost(&self->buffers.host_slab, cfg->host_buffer_bytes));
  CK_CUDA(cuda_fail,
          cudaMalloc(&self->buffers.dev_compressed, cfg->host_buffer_bytes));
  CK_CUDA(
    cuda_fail,
    cudaMalloc(&self->buffers.dev_decompressed, cfg->device_buffer_bytes));

  struct store_fs_config sc = {
    .root = self->store_root,
    .nthreads = (int)cfg->n_io_threads,
  };
  self->store = store_fs_create(&sc);
  if (!self->store)
    goto oom;

  self->meta_cache =
    zarr_meta_cache_create(self->store, cfg->n_zarrs_meta_cache);
  if (!self->meta_cache)
    goto oom;
  self->shard_cache =
    zarr_shard_cache_create(self->store, cfg->n_shards_meta_cache);
  if (!self->shard_cache)
    goto oom;

  struct planner_config pcfg = {
    .meta_cache = self->meta_cache,
    .shard_cache = self->shard_cache,
    .page_alignment = self->page_alignment,
  };
  if (planner_create(&pcfg, &self->planner) != DAMACY_OK)
    goto oom;

  if (plan_init(&self->plan, DAMACY_MAX_CHUNKS_PER_BATCH))
    goto oom;

  self->decoder = decoder_create(self->cuda_device,
                                 DAMACY_MAX_CHUNKS_PER_BATCH,
                                 DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES);
  if (!self->decoder)
    goto oom;

  if (lookahead_init(&self->lookahead,
                     cfg->lookahead_batches * cfg->batch_size))
    goto oom;

  self->batch_samples = (struct damacy_sample_slot*)calloc(
    cfg->batch_size, sizeof(struct damacy_sample_slot));
  if (!self->batch_samples)
    goto oom;
  self->batch_stage = (struct damacy_sample*)calloc(
    cfg->batch_size, sizeof(struct damacy_sample));
  if (!self->batch_stage)
    goto oom;

  self->handle.d = self;
  self->handle.batch_id = 0;

  *out = self;
  return DAMACY_OK;

oom:
  damacy_destroy(self);
  return DAMACY_OOM;
cuda_fail:
  damacy_destroy(self);
  return DAMACY_CUDA;
invalid_arg:
  return DAMACY_INVAL;
}

void
damacy_destroy(struct damacy* self)
{
  if (!self)
    return;
  if (self->stream)
    cudaStreamSynchronize(self->stream);

  free(self->batch_stage);
  free(self->batch_samples);
  lookahead_destroy(&self->lookahead);
  if (self->decoder)
    decoder_destroy(self->decoder);
  plan_destroy(&self->plan);
  if (self->planner)
    planner_destroy(self->planner);
  if (self->shard_cache)
    zarr_shard_cache_destroy(self->shard_cache);
  if (self->meta_cache)
    zarr_meta_cache_destroy(self->meta_cache);
  if (self->store)
    store_destroy(self->store);

  if (self->batch.dev_ptr)
    cudaFree(self->batch.dev_ptr);
  if (self->buffers.dev_decompressed)
    cudaFree(self->buffers.dev_decompressed);
  if (self->buffers.dev_compressed)
    cudaFree(self->buffers.dev_compressed);
  if (self->buffers.host_slab)
    cudaFreeHost(self->buffers.host_slab);
  if (self->stream)
    cudaStreamDestroy(self->stream);

  free(self->store_root);
  free(self);
}

// --- push -----------------------------------------------------------------

// Validate one sample against the meta cache + cfg. Advances bp by 1 on
// success and returns DAMACY_OK; otherwise returns the appropriate error
// without consuming.
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

  if (!damacy_dtype_match(self->cfg.dtype, meta->dtype))
    return DAMACY_DTYPE;
  if (sample->aabb.rank != meta->rank)
    return DAMACY_RANK;
  if (self->batch.allocated && !sample_shape_matches_batch(self, &sample->aabb))
    return DAMACY_INVAL;

  if (lookahead_push(&self->lookahead, sample))
    return DAMACY_OOM;
  return DAMACY_OK;
}

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

// --- batch production -----------------------------------------------------

// Pack read_op.dst_buf_offset values into a sequential layout in the host
// slab. Returns total bytes used; sets failed_status to OOM on overflow.
static enum damacy_status
pack_read_ops(struct damacy* self,
              uint32_t n_read_ops,
              uint64_t* out_used_bytes)
{
  uint64_t cursor = 0;
  for (uint32_t i = 0; i < n_read_ops; ++i) {
    struct read_op* r = &self->plan.read_ops[i];
    r->dst_buf_offset = cursor;
    if (cursor > UINT64_MAX - r->nbytes)
      return DAMACY_OOM;
    cursor += r->nbytes;
    if (cursor > self->cfg.host_buffer_bytes)
      return DAMACY_OOM;
  }
  *out_used_bytes = cursor;
  return DAMACY_OK;
}

static enum damacy_status
pack_chunk_arena(struct damacy* self, uint32_t n_chunks)
{
  uint64_t cursor = 0;
  for (uint32_t i = 0; i < n_chunks; ++i) {
    struct chunk_plan* c = &self->plan.chunk_plans[i];
    c->dev_decompressed_offset = cursor;
    if (cursor > UINT64_MAX - c->decompressed_nbytes)
      return DAMACY_OOM;
    cursor += c->decompressed_nbytes;
    if (cursor > self->cfg.device_buffer_bytes)
      return DAMACY_OOM;
  }
  return DAMACY_OK;
}

// Build host-side decompress fanout (compressed/decompressed device
// pointers + sizes) from the plan. n_chunks must already be packed.
static void
build_decoder_fanout(struct damacy* self, uint32_t n_chunks)
{
  for (uint32_t i = 0; i < n_chunks; ++i) {
    const struct chunk_plan* c = &self->plan.chunk_plans[i];
    const struct read_op* r = &self->plan.read_ops[c->read_op_idx];
    self->plan.h_compressed_ptrs[i] =
      (const void*)((const uint8_t*)self->buffers.dev_compressed +
                    r->dst_buf_offset + c->offset_in_read);
    self->plan.h_compressed_sizes[i] = c->compressed_nbytes;
    self->plan.h_decompressed_ptrs[i] =
      (void*)((uint8_t*)self->buffers.dev_decompressed +
              c->dev_decompressed_offset);
    self->plan.h_decompressed_sizes[i] = c->decompressed_nbytes;
  }
}

// Build per-chunk assemble metadata. Returns max window-element count
// across chunks (for grid sizing).
static uint32_t
build_assemble_meta(struct damacy* self, uint32_t n_chunks)
{
  uint32_t bpe = damacy_dtype_bpe(self->cfg.dtype);
  uint8_t spatial_rank = (uint8_t)(self->batch.rank - 1);
  uint32_t max_window = 1;
  for (uint32_t i = 0; i < n_chunks; ++i) {
    const struct chunk_plan* c = &self->plan.chunk_plans[i];
    struct assemble_chunk* a = &self->plan.h_assemble[i];

    int64_t src_off_elems = 0;
    for (uint8_t d = 0; d < spatial_rank; ++d)
      src_off_elems += c->src.dims[d].beg * c->src_strides[d];
    a->src_base_byte_off =
      c->dev_decompressed_offset + (uint64_t)src_off_elems * (uint64_t)bpe;

    int64_t sample_idx = c->dst.dims[0].beg;
    int64_t dst_off_elems = sample_idx * self->batch.strides[0];
    for (uint8_t d = 0; d < spatial_rank; ++d)
      dst_off_elems += c->dst.dims[d + 1].beg * self->batch.strides[d + 1];
    a->dst_base_byte_off = (uint64_t)dst_off_elems * (uint64_t)bpe;

    a->rank = spatial_rank;
    uint64_t win_elems = 1;
    for (uint8_t d = 0; d < spatial_rank; ++d) {
      uint32_t w = (uint32_t)(c->src.dims[d].end - c->src.dims[d].beg);
      a->win[d] = w;
      a->src_strides[d] = c->src_strides[d];
      a->dst_strides[d] = self->batch.strides[d + 1];
      win_elems *= w;
    }
    if (win_elems > max_window)
      max_window = (uint32_t)win_elems;
  }
  return max_window;
}

// Run one full batch end-to-end through the pipeline. Consumes the
// first n_samples from the lookahead. On success leaves
// self->batch.ready = 1 with shape[0] = n_samples.
static enum damacy_status
produce_batch(struct damacy* self, uint32_t n_samples)
{
  enum damacy_status status = DAMACY_OK;
  lookahead_drain(&self->lookahead, self->batch_samples, n_samples);

  status = batch_pool_allocate(self, &self->batch_samples[0].aabb);
  if (status != DAMACY_OK)
    goto cleanup;
  // shape[0] reflects the number of complete samples in this batch
  // (== batch_size for full, < for flush).
  self->batch.shape[0] = (int64_t)n_samples;

  // Stage damacy_sample[] for the planner. URI ownership stays in
  // batch_samples until cleanup; planner only reads them transiently.
  if (n_samples > self->cfg.batch_size) {
    status = DAMACY_INVAL;
    goto cleanup;
  }
  for (uint32_t i = 0; i < n_samples; ++i) {
    self->batch_stage[i].uri = self->batch_samples[i].uri;
    self->batch_stage[i].aabb = self->batch_samples[i].aabb;
  }

  struct planner_output plan_out = {
    .read_ops = self->plan.read_ops,
    .read_ops_cap = self->plan.read_ops_cap,
    .chunk_plans = self->plan.chunk_plans,
    .chunk_plans_cap = self->plan.chunk_plans_cap,
  };
  status =
    planner_plan(self->planner, self->batch_stage, n_samples, 0, &plan_out);
  if (status != DAMACY_OK)
    goto cleanup;

  uint64_t host_used_bytes = 0;
  status = pack_read_ops(self, plan_out.n_read_ops, &host_used_bytes);
  if (status != DAMACY_OK)
    goto cleanup;
  status = pack_chunk_arena(self, plan_out.n_chunk_plans);
  if (status != DAMACY_OK)
    goto cleanup;

  // Issue reads.
  for (uint32_t i = 0; i < plan_out.n_read_ops; ++i) {
    const struct read_op* r = &self->plan.read_ops[i];
    self->plan.store_reads[i] = (struct store_read){
      .key = r->shard_path,
      .dst = (uint8_t*)self->buffers.host_slab + r->dst_buf_offset,
      .offset = r->file_offset,
      .len = r->nbytes,
    };
  }
  if (plan_out.n_read_ops > 0) {
    if (store_read_many(
          self->store, self->plan.store_reads, plan_out.n_read_ops)) {
      status = DAMACY_IO;
      goto cleanup;
    }
  }

  // Pre-zero the output (empty chunks rely on zero-init).
  CK_CUDA(
    cuda_fail,
    cudaMemsetAsync(self->batch.dev_ptr, 0, self->batch.n_bytes, self->stream));

  if (plan_out.n_chunk_plans > 0) {
    // H2D the used portion of the host slab.
    CK_CUDA(cuda_fail,
            cudaMemcpyAsync(self->buffers.dev_compressed,
                            self->buffers.host_slab,
                            host_used_bytes,
                            cudaMemcpyHostToDevice,
                            self->stream));

    build_decoder_fanout(self, plan_out.n_chunk_plans);
    if (decoder_decompress_batch(self->decoder,
                                 self->stream,
                                 self->plan.h_compressed_ptrs,
                                 self->plan.h_compressed_sizes,
                                 self->plan.h_decompressed_ptrs,
                                 self->plan.h_decompressed_sizes,
                                 plan_out.n_chunk_plans)) {
      status = DAMACY_DECODE;
      goto cleanup;
    }

    uint32_t max_window = build_assemble_meta(self, plan_out.n_chunk_plans);
    CK_CUDA(cuda_fail,
            cudaMemcpyAsync(self->plan.d_assemble,
                            self->plan.h_assemble,
                            (size_t)plan_out.n_chunk_plans *
                              sizeof(struct assemble_chunk),
                            cudaMemcpyHostToDevice,
                            self->stream));

    if (assemble_launch(self->stream,
                        self->plan.d_assemble,
                        plan_out.n_chunk_plans,
                        max_window,
                        self->buffers.dev_decompressed,
                        self->batch.dev_ptr,
                        damacy_dtype_bpe(self->cfg.dtype))) {
      status = DAMACY_CUDA;
      goto cleanup;
    }
  }

  CK_CUDA(cuda_fail, cudaStreamSynchronize(self->stream));

  self->batch.ready = 1;
  self->batch.batch_id = self->next_batch_id++;
  self->stats.batches_emitted++;
  self->stats.waves_emitted++;
  goto cleanup;

cuda_fail:
  status = DAMACY_CUDA;

cleanup:
  for (uint32_t i = 0; i < n_samples; ++i)
    sample_slot_clear(&self->batch_samples[i]);
  if (status != DAMACY_OK)
    self->failed_status = status;
  return status;
}

// --- pop / flush / release ------------------------------------------------

enum damacy_status
damacy_pop(struct damacy* self, struct damacy_batch** out)
{
  CHECK_SILENT(invalid_arg, self);
  CHECK_SILENT(invalid_arg, out);
  *out = NULL;
  if (self->failed_status != DAMACY_OK)
    return self->failed_status;

  if (self->batch.ready && !self->batch.held) {
    self->handle.batch_id = self->batch.batch_id;
    self->batch.held = 1;
    *out = &self->handle;
    return DAMACY_OK;
  }
  if (self->batch.held)
    return DAMACY_AGAIN;

  if (self->lookahead.size < self->cfg.batch_size)
    return DAMACY_AGAIN;

  enum damacy_status ps = produce_batch(self, self->cfg.batch_size);
  if (ps != DAMACY_OK)
    return ps;

  self->handle.batch_id = self->batch.batch_id;
  self->batch.held = 1;
  *out = &self->handle;
  return DAMACY_OK;

invalid_arg:
  return DAMACY_INVAL;
}

void
damacy_release(struct damacy* self, struct damacy_batch* b)
{
  if (!self || !b || b != &self->handle)
    return;
  self->batch.held = 0;
  self->batch.ready = 0;
}

enum damacy_status
damacy_flush(struct damacy* self)
{
  if (!self)
    return DAMACY_INVAL;
  if (self->failed_status != DAMACY_OK)
    return self->failed_status;
  // If a full ready batch is already queued, leave it for pop.
  if (self->batch.ready)
    return DAMACY_OK;
  if (self->lookahead.size == 0)
    return DAMACY_OK;

  uint32_t n = self->lookahead.size;
  if (n > self->cfg.batch_size)
    n = self->cfg.batch_size;
  enum damacy_status ps = produce_batch(self, n);
  if (ps != DAMACY_OK)
    return ps;
  if (n < self->cfg.batch_size)
    self->stats.batches_truncated++;
  return DAMACY_OK;
}

// --- batch info / stats ---------------------------------------------------

void
damacy_batch_info(const struct damacy_batch* b, struct damacy_batch_info* out)
{
  if (!out)
    return;
  memset(out, 0, sizeof(*out));
  if (!b || !b->d || !b->d->batch.ready)
    return;
  const struct damacy* self = b->d;
  out->device_ptr = self->batch.dev_ptr;
  out->rank = self->batch.rank;
  out->dtype = self->cfg.dtype;
  out->ready_stream = (void*)self->stream;
  out->batch_id = self->batch.batch_id;
  for (uint8_t d = 0; d < self->batch.rank; ++d)
    out->shape[d] = self->batch.shape[d];
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
}

void
damacy_stats_reset(struct damacy* self)
{
  if (!self)
    return;
  memset(&self->stats, 0, sizeof(self->stats));
}
