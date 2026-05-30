// Compile-time limits shared across damacy's modules.
#pragma once

#include <stddef.h>
#include <stdint.h>

// Maximum tensor rank we'll plan over.
#define DAMACY_MAX_RANK 31

// Per-chunk byte caps. compressed_nbytes / decompressed_nbytes in
// chunk_plan are uint32_t; we deliberately cap chunks at 4 GB so the
// 32-bit fields are safe. Target chunks are ~1 MB.
#define DAMACY_MAX_CHUNK_BYTES UINT32_MAX

// Per-shard file size cap. file_offset is uint64_t but we only ever
// expect to use ~46 bits. Asserted at shard-index parse time. Target
// shards are ~1 GB.
#define DAMACY_MAX_SHARD_BYTES (1ull << 46)

// Default for damacy_config.max_chunk_uncompressed_bytes when the user
// leaves it at 0. Matches the pre-runtime-cap behaviour and keeps the
// 8 GB-class GPU budget viable out of the box.
#define DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES (512ull << 10) // 512 KB

// Default cap on post-coalesce read_op size. read_op.nbytes is uint32_t,
// so the cap must fit there too — coalesce_chunks won't fuse past it.
#define DAMACY_DEFAULT_READ_OP_MAX_BYTES (512ull << 10)
#ifdef __cplusplus
static_assert(DAMACY_DEFAULT_READ_OP_MAX_BYTES <= UINT32_MAX,
              "read_op.nbytes is uint32_t");
#else
_Static_assert(DAMACY_DEFAULT_READ_OP_MAX_BYTES <= UINT32_MAX,
               "read_op.nbytes is uint32_t");
#endif

// In-flight wave count. Fixed; sets the minimum host_buffer_waves and
// the device-side decode concurrency depth.
#define DAMACY_N_WAVES 2

// Default depth of the pinned-host slab pool, in waves. = N_WAVES is
// the minimum; bumping higher lets IO for upcoming waves prefill before
// a wave struct frees, useful for slow / variable-latency IO backends.
// On fast local NVMe the extra IO concurrency adds queueing overhead
// that outweighs the prefill benefit, so the default stays at N_WAVES.
#define DAMACY_DEFAULT_HOST_BUFFER_WAVES DAMACY_N_WAVES

// Upper bound on cfg.host_buffer_waves. 8 covers any realistic IO
// look-ahead need; each extra slot costs one dev_compressed_per_wave of
// pinned host memory.
#define DAMACY_MAX_HOST_BUFFER_WAVES 8

// Default for damacy_tuning.max_chunks_per_wave. nvcomp temp scratch is
// sized as max_chunks_per_wave × runtime_chunk_cap (× 2 waves) so the
// default stays small and large batches split across multiple waves.
#define DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE 512u

// Hard upper bound on damacy_tuning.max_chunks_per_wave: d_block_chunk_map
// packs chunk_idx into the upper 16 bits.
#define DAMACY_HARD_MAX_CHUNKS_PER_WAVE 0xFFFFu

// Per-batch hard cap. Bounds the planner output and the assemble
// metadata buffer.
#define DAMACY_MAX_CHUNKS_PER_BATCH 16384u

// Sizes io_queue's per-worker arrays (see io_queue.posix.c). Generous:
// consumer NVMe saturates well below 32 in-flight reads. Bump if a real
// workload demonstrates need.
#define DAMACY_MAX_IO_THREADS 32u

// Metadata prefetch is dominated by small, latency-bound reads. Keep its
// default materially higher than the bulk IO default used by most examples.
#define DAMACY_DEFAULT_PREFETCH_IO_THREADS 16u

// Matches DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE. Must be a power of two —
// io_queue indexes via bitmask.
#define DAMACY_IO_QUEUE_INITIAL_CAP 512u

// Default for damacy_tuning.max_substreams_per_chunk. Parser rejects
// blosc1 layouts with more sub-streams (DAMACY_DECODE).
#define DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK 32u

// Bounded so the product with HARD_MAX_CHUNKS_PER_WAVE fits uint32
// (max_substreams_per_wave).
#define DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK 0xFFFFu

#define DAMACY_HARD_MAX_SUBSTREAMS_PER_WAVE_U64                                \
  ((uint64_t)DAMACY_HARD_MAX_CHUNKS_PER_WAVE *                                 \
   (uint64_t)DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK)
#ifdef __cplusplus
static_assert(DAMACY_HARD_MAX_SUBSTREAMS_PER_WAVE_U64 <= UINT32_MAX,
              "max_substreams_per_wave must fit in uint32");
#else
_Static_assert(DAMACY_HARD_MAX_SUBSTREAMS_PER_WAVE_U64 <= UINT32_MAX,
               "max_substreams_per_wave must fit in uint32");
#endif

// Defensive cap on header.nbytes parsed from a blosc1 chunk. Prevents
// overflow in the nblocks ceil-div for adversarial inputs.
#define DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES (16ull << 20) // 16 MB

// Initial substream-batch cap for the pool-shared zstd decoder + per-wave
// fanout SOAs. Sized off a typical wave (hundreds of substreams);
// grows on demand when a wave's actual substream count exceeds the cap.
#define DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP 1024u

static inline uint32_t
damacy_max_substreams_per_wave(uint32_t chunks, uint32_t substreams_per_chunk)
{
  return (uint32_t)((uint64_t)chunks * (uint64_t)substreams_per_chunk);
}
