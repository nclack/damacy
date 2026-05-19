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

// Wave cap. Decoupled from the per-batch cap below: nvcomp temp scratch
// is sized as MAX_CHUNKS_PER_WAVE × runtime_chunk_cap (× 2 waves), so we
// keep this small and let large batches split across multiple waves.
#define DAMACY_MAX_CHUNKS_PER_WAVE 512u

// Per-batch hard cap. Bounds the planner output and the assemble
// metadata buffer.
#define DAMACY_MAX_CHUNKS_PER_BATCH 16384u

// Sizes io_queue's per-worker arrays (see io_queue.posix.c). Generous:
// consumer NVMe saturates well below 32 in-flight reads. Bump if a real
// workload demonstrates need.
#define DAMACY_MAX_IO_THREADS 32u

// Sized to absorb one full wave (DAMACY_MAX_CHUNKS_PER_WAVE) without
// growing. Must be a power of two — io_queue indexes via bitmask.
#define DAMACY_IO_QUEUE_INITIAL_CAP 512u

// Per-zarr-chunk blosc1 nblocks cap. Two roles:
//   (a) parser hard reject — zarr_chunk_layout_probe rejects layouts
//       with more blocks (DAMACY_DECODE);
//   (b) bound on kernel smem bstarts/sorted arrays in the parse/emit
//       kernels.
#define DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK 32u
#ifdef __cplusplus
static_assert(DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK <= UINT16_MAX,
              "observed_max_nblocks_per_chunk slot is uint16_t");
#else
_Static_assert(DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK <= UINT16_MAX,
               "observed_max_nblocks_per_chunk slot is uint16_t");
#endif

// Structural ceiling on blosc1 sub-streams across a wave. Caps fanout
// SOA growth and zstd-decoder batch growth.
#define WAVE_ZSUBS_STRUCTURAL_MAX                                              \
  ((size_t)DAMACY_MAX_CHUNKS_PER_WAVE *                                        \
   (size_t)DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK)
#ifdef __cplusplus
static_assert(WAVE_ZSUBS_STRUCTURAL_MAX <= UINT32_MAX,
              "fanout cap is uint32_t");
#else
_Static_assert(WAVE_ZSUBS_STRUCTURAL_MAX <= UINT32_MAX,
               "fanout cap is uint32_t");
#endif

// Defensive cap on header.nbytes parsed from a blosc1 chunk. Prevents
// overflow in the nblocks ceil-div for adversarial inputs.
#define DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES (16ull << 20) // 16 MB

// d_block_chunk_map packs chunk_idx into the upper 16 bits; the GPU
// kernel unpacks via `packed >> 16` and indexes d_chunks/d_sample_plans
// directly. Raising the cap past 0xFFFFu silently truncates.
#ifdef __cplusplus
static_assert(DAMACY_MAX_CHUNKS_PER_WAVE <= 0xFFFFu,
              "DAMACY_MAX_CHUNKS_PER_WAVE must fit in 16 bits");
#else
_Static_assert(DAMACY_MAX_CHUNKS_PER_WAVE <= 0xFFFFu,
               "DAMACY_MAX_CHUNKS_PER_WAVE must fit in 16 bits");
#endif
// Initial substream-batch cap for the pool-shared zstd decoder + per-wave
// fanout SOAs. Sized off a typical wave (hundreds of substreams) rather
// than the hard ceiling above; grows on demand when a wave's actual
// substream count exceeds the current cap.
#define DAMACY_BLOSC_ZSTD_INITIAL_BATCH_CAP 1024u
// Memcpy + (bit)unshuffle ops cap: every chunk could be MEMCPY/SHUFFLE'd.
#define DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE DAMACY_MAX_CHUNKS_PER_WAVE
#define DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE DAMACY_MAX_CHUNKS_PER_WAVE
