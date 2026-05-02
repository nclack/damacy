#include "pipeline.h"

#include "limits.h"
#include "store.h"
#include "zarr.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct slot
{
  void* host_compressed;
  void* device_compressed;
  void* device_output;

  const void** chunk_compressed_ptrs;
  void** chunk_decompressed_ptrs;
  size_t* chunk_compressed_sizes;
  size_t* chunk_uncompressed_sizes;

  cudaEvent_t reader_done;
  cudaEvent_t ready;
  uint64_t batch_id;
  uint64_t bytes_compressed_this_batch;
  uint64_t bytes_decompressed_this_batch;
  int in_flight;
};

struct pipeline
{
  struct store* store;
  struct zarr_reader* reader;
  struct decoder* decoder;
  enum pipeline_pattern pattern;

  int batch_size;
  int chunks_per_sample;
  int chunks_per_batch;
  int prefetch_depth;
  int device_id;

  size_t max_compressed_chunk_bytes;
  size_t inner_uncompressed_bytes;

  uint8_t rank;
  uint64_t n_chunks_per_dim[DAMACY_MAX_RANK];
  uint64_t n_chunks_total;

  uint64_t seq_cursor;
  uint64_t prng_state;

  cudaStream_t reader_stream;
  cudaStream_t decoder_stream;

  struct slot* slots;
  int n_slots;
  int next_issue;
  int next_return;
  uint64_t next_batch_id;

  uint64_t total_batches;
  uint64_t total_compressed_bytes;
  uint64_t total_decompressed_bytes;

  struct store_read* read_scratch;
};

// xorshift64* — cheap, decent quality for scattering linear indices.
static uint64_t
prng_next(uint64_t* s)
{
  uint64_t x = *s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  *s = x;
  return x * 2685821657736338717ull;
}

static void
linear_to_chunk_coord(const struct pipeline* p, uint64_t linear, int64_t* coord)
{
  for (int d = (int)p->rank - 1; d >= 0; --d) {
    uint64_t n = p->n_chunks_per_dim[d];
    coord[d] = (int64_t)(linear % n);
    linear /= n;
  }
}

static void
slot_destroy(struct slot* s)
{
  if (!s)
    return;
  if (s->reader_done)
    cudaEventDestroy(s->reader_done);
  if (s->ready)
    cudaEventDestroy(s->ready);
  cudaFreeHost(s->host_compressed);
  cudaFree(s->device_compressed);
  cudaFree(s->device_output);
  free(s->chunk_compressed_ptrs);
  free(s->chunk_decompressed_ptrs);
  free(s->chunk_compressed_sizes);
  free(s->chunk_uncompressed_sizes);
  memset(s, 0, sizeof(*s));
}

static int
slot_init(struct slot* s,
          int chunks_per_batch,
          size_t max_compressed_chunk_bytes,
          size_t inner_uncompressed_bytes)
{
  size_t compressed_total =
    (size_t)chunks_per_batch * max_compressed_chunk_bytes;
  size_t output_total = (size_t)chunks_per_batch * inner_uncompressed_bytes;

  if (cudaMallocHost(&s->host_compressed, compressed_total) != cudaSuccess)
    return 1;
  if (cudaMalloc(&s->device_compressed, compressed_total) != cudaSuccess)
    return 1;
  if (cudaMalloc(&s->device_output, output_total) != cudaSuccess)
    return 1;
  if (cudaEventCreateWithFlags(&s->reader_done, cudaEventDisableTiming) !=
      cudaSuccess)
    return 1;
  if (cudaEventCreateWithFlags(&s->ready, cudaEventDisableTiming) !=
      cudaSuccess)
    return 1;

  s->chunk_compressed_ptrs =
    (const void**)calloc((size_t)chunks_per_batch, sizeof(void*));
  s->chunk_decompressed_ptrs =
    (void**)calloc((size_t)chunks_per_batch, sizeof(void*));
  s->chunk_compressed_sizes =
    (size_t*)calloc((size_t)chunks_per_batch, sizeof(size_t));
  s->chunk_uncompressed_sizes =
    (size_t*)calloc((size_t)chunks_per_batch, sizeof(size_t));
  if (!s->chunk_compressed_ptrs || !s->chunk_decompressed_ptrs ||
      !s->chunk_compressed_sizes || !s->chunk_uncompressed_sizes)
    return 1;
  return 0;
}

static int
issue_batch(struct pipeline* p, int slot_idx, uint64_t batch_id)
{
  struct slot* s = &p->slots[slot_idx];

  s->bytes_compressed_this_batch = 0;
  s->bytes_decompressed_this_batch = 0;

  for (int b = 0; b < p->batch_size; ++b) {
    uint64_t start;
    if (p->pattern == PIPELINE_PATTERN_RANDOM) {
      start = prng_next(&p->prng_state) % p->n_chunks_total;
    } else {
      start = p->seq_cursor;
      p->seq_cursor =
        (p->seq_cursor + (uint64_t)p->chunks_per_sample) % p->n_chunks_total;
    }
    for (int c = 0; c < p->chunks_per_sample; ++c) {
      int k = b * p->chunks_per_sample + c;
      int64_t coord[DAMACY_MAX_RANK];
      linear_to_chunk_coord(
        p, (start + (uint64_t)c) % p->n_chunks_total, coord);

      struct zarr_chunk_loc loc;
      if (zarr_reader_locate(p->reader, coord, &loc))
        return 1;
      if (loc.len > p->max_compressed_chunk_bytes)
        return 1;

      uint8_t* host_chunk = (uint8_t*)s->host_compressed +
                            (size_t)k * p->max_compressed_chunk_bytes;
      uint8_t* dev_chunk = (uint8_t*)s->device_compressed +
                           (size_t)k * p->max_compressed_chunk_bytes;
      uint8_t* dev_dec =
        (uint8_t*)s->device_output + (size_t)k * p->inner_uncompressed_bytes;

      p->read_scratch[k] = (struct store_read){
        .key = loc.key,
        .dst = host_chunk,
        .offset = loc.offset,
        .len = loc.len,
      };
      s->chunk_compressed_ptrs[k] = dev_chunk;
      s->chunk_decompressed_ptrs[k] = dev_dec;
      s->chunk_compressed_sizes[k] = loc.len;
      s->chunk_uncompressed_sizes[k] = p->inner_uncompressed_bytes;
      s->bytes_compressed_this_batch += loc.len;
      s->bytes_decompressed_this_batch += p->inner_uncompressed_bytes;
    }
  }

  if (store_read_many(p->store, p->read_scratch, (size_t)p->chunks_per_batch))
    return 1;

  size_t arena_bytes =
    (size_t)p->chunks_per_batch * p->max_compressed_chunk_bytes;
  if (cudaMemcpyAsync(s->device_compressed,
                      s->host_compressed,
                      arena_bytes,
                      cudaMemcpyHostToDevice,
                      p->reader_stream) != cudaSuccess)
    return 1;
  if (cudaEventRecord(s->reader_done, p->reader_stream) != cudaSuccess)
    return 1;

  if (cudaStreamWaitEvent(p->decoder_stream, s->reader_done, 0) != cudaSuccess)
    return 1;
  if (decoder_decompress_batch(p->decoder,
                               p->decoder_stream,
                               s->chunk_compressed_ptrs,
                               s->chunk_compressed_sizes,
                               s->chunk_decompressed_ptrs,
                               s->chunk_uncompressed_sizes,
                               (size_t)p->chunks_per_batch))
    return 1;
  if (cudaEventRecord(s->ready, p->decoder_stream) != cudaSuccess)
    return 1;

  s->batch_id = batch_id;
  s->in_flight = 1;
  return 0;
}

struct pipeline*
pipeline_create(const struct pipeline_config* cfg)
{
  if (!cfg || !cfg->store || !cfg->reader || !cfg->decoder ||
      cfg->batch_size <= 0 || cfg->chunks_per_sample <= 0 ||
      cfg->prefetch_depth <= 0)
    return NULL;
  if (cudaSetDevice(cfg->device_id) != cudaSuccess)
    return NULL;

  struct pipeline* p = (struct pipeline*)calloc(1, sizeof(*p));
  if (!p)
    return NULL;

  p->store = cfg->store;
  p->reader = cfg->reader;
  p->decoder = cfg->decoder;
  p->pattern = cfg->pattern;
  p->batch_size = cfg->batch_size;
  p->chunks_per_sample = cfg->chunks_per_sample;
  p->chunks_per_batch = cfg->batch_size * cfg->chunks_per_sample;
  p->prefetch_depth = cfg->prefetch_depth;
  p->device_id = cfg->device_id;
  p->prng_state = cfg->seed ? cfg->seed : 0xDEADBEEFCAFEBABEull;

  const struct zarr_array_info* info = zarr_reader_info(cfg->reader);
  if (!info)
    goto fail;
  p->rank = info->rank;
  p->n_chunks_total = 1;
  for (uint8_t d = 0; d < info->rank; ++d) {
    uint64_t nc = (info->dims[d].size + info->dims[d].chunk_size - 1) /
                  info->dims[d].chunk_size;
    p->n_chunks_per_dim[d] = nc;
    p->n_chunks_total *= nc;
  }
  p->inner_uncompressed_bytes =
    zarr_reader_chunk_uncompressed_bytes(cfg->reader);
  p->max_compressed_chunk_bytes = cfg->max_compressed_chunk_bytes
                                    ? cfg->max_compressed_chunk_bytes
                                    : (p->inner_uncompressed_bytes * 2);

  if (cudaStreamCreateWithFlags(&p->reader_stream, cudaStreamNonBlocking) !=
      cudaSuccess)
    goto fail;
  if (cudaStreamCreateWithFlags(&p->decoder_stream, cudaStreamNonBlocking) !=
      cudaSuccess)
    goto fail;

  p->n_slots = cfg->prefetch_depth;
  p->slots = (struct slot*)calloc((size_t)p->n_slots, sizeof(struct slot));
  if (!p->slots)
    goto fail;
  for (int i = 0; i < p->n_slots; ++i) {
    if (slot_init(&p->slots[i],
                  p->chunks_per_batch,
                  p->max_compressed_chunk_bytes,
                  p->inner_uncompressed_bytes))
      goto fail;
  }

  p->read_scratch = (struct store_read*)calloc((size_t)p->chunks_per_batch,
                                               sizeof(struct store_read));
  if (!p->read_scratch)
    goto fail;

  p->next_issue = 0;
  p->next_return = 0;
  p->next_batch_id = 1;
  return p;

fail:
  pipeline_destroy(p);
  return NULL;
}

void
pipeline_destroy(struct pipeline* p)
{
  if (!p)
    return;
  if (p->reader_stream)
    cudaStreamSynchronize(p->reader_stream);
  if (p->decoder_stream)
    cudaStreamSynchronize(p->decoder_stream);
  if (p->slots) {
    for (int i = 0; i < p->n_slots; ++i)
      slot_destroy(&p->slots[i]);
    free(p->slots);
  }
  free(p->read_scratch);
  if (p->reader_stream)
    cudaStreamDestroy(p->reader_stream);
  if (p->decoder_stream)
    cudaStreamDestroy(p->decoder_stream);
  free(p);
}

int
pipeline_next(struct pipeline* p, struct pipeline_batch* out)
{
  if (!p || !out)
    return 1;

  // Prime the pipeline up to prefetch_depth in flight.
  for (int i = 0; i < p->n_slots; ++i) {
    int idx = (p->next_issue) % p->n_slots;
    if (p->slots[idx].in_flight)
      break;
    if (issue_batch(p, idx, p->next_batch_id))
      return 1;
    p->next_batch_id++;
    p->next_issue = (idx + 1) % p->n_slots;
  }

  int head = p->next_return;
  struct slot* hs = &p->slots[head];
  if (!hs->in_flight)
    return 1;

  if (cudaEventSynchronize(hs->ready) != cudaSuccess)
    return 1;

  out->device_ptr = hs->device_output;
  out->nbytes = hs->bytes_decompressed_this_batch;
  out->ready_event = hs->ready;
  out->batch_id = hs->batch_id;

  p->total_batches++;
  p->total_compressed_bytes += hs->bytes_compressed_this_batch;
  p->total_decompressed_bytes += hs->bytes_decompressed_this_batch;
  return 0;
}

void
pipeline_release(struct pipeline* p, uint64_t batch_id)
{
  if (!p)
    return;
  int head = p->next_return;
  if (p->slots[head].batch_id != batch_id)
    return;
  p->slots[head].in_flight = 0;
  p->next_return = (p->next_return + 1) % p->n_slots;
  // Issue a replacement batch into the now-free slot.
  (void)issue_batch(p, head, p->next_batch_id++);
  p->next_issue = (head + 1) % p->n_slots;
}

void
pipeline_stats_get(const struct pipeline* p, struct pipeline_stats* out)
{
  if (!p || !out)
    return;
  out->n_batches = p->total_batches;
  out->bytes_read_compressed = p->total_compressed_bytes;
  out->bytes_decompressed = p->total_decompressed_bytes;
  out->wall_seconds = 0.0; // bench harness times around pipeline_next
}
