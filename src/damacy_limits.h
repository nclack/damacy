// Compile-time limits shared across damacy's modules.
#pragma once

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

// Max bytes (including trailing NUL) for a shard path inlined into
// struct read_op. The plan queue stores read_ops by value across batches
// (step 5+), so paths can't be heap pointers owned by the planner —
// they'd dangle on the next plan. 224 leaves headroom for an absolute
// uri + chunk-grid coordinates without inflating chunk_plan-sized
// records too much.
#define DAMACY_MAX_PATH 224

// Hard ceiling on per-substream uncompressed bytes. Compile-time max for
// kernel array sizing (smem bstarts/sorted in the parse + emit kernels)
// and the upper bound the runtime config's max_chunk_uncompressed_bytes
// can request. Real nvcomp scratch is sized off the runtime config (see
// wave_init); 2 MB lets users opt into 1–4 MB chunks if they configure
// max_gpu_memory_bytes accordingly. Chunks exceeding the runtime cap
// are rejected at the planner with DAMACY_INVAL.
#define DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES (2ull << 20) // 2 MB

// Default for damacy_config.max_chunk_uncompressed_bytes when the user
// leaves it at 0. Matches the pre-runtime-cap behaviour and keeps the
// 8 GB-class GPU budget viable out of the box.
#define DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES (512ull << 10) // 512 KB

// Default for damacy_config.max_gpu_memory_bytes when the user leaves
// it at 0. ~1 GB fits comfortably on consumer GPUs.
#define DAMACY_DEFAULT_MAX_GPU_MEMORY_BYTES (1ull << 30) // 1 GB

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

// Per-zarr-chunk blosc1 nblocks cap. Derivation:
//   max chunk uncompressed = DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES (2 MB)
//   blosc encoder min blocksize = 64 KB (compute_blocksize floor for
//                                        splittable codecs, c-blosc)
//   ⇒ worst-case nblocks = 2 MB / 64 KB = 32.
// Inputs with more blocks are rejected at parse with DAMACY_DECODE.
#define DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK 32u

// Defensive cap on header.nbytes parsed from a blosc1 chunk. Prevents
// overflow in the nblocks ceil-div for adversarial inputs, independent
// of the runtime DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES cap (which gates
// the planner, not the parser).
#define DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES (16ull << 20) // 16 MB

// Worst-case substream count per wave for blosc1-zstd: 1 substream per
// blosc-block. Acts as the hard ceiling for the observe-and-grow runtime
// cap on the shared zstd decoder + per-wave fanout SOA.
#define DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE                                    \
  (DAMACY_MAX_CHUNKS_PER_WAVE * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK)
// Per-wave's tight substream upper bound (n_chunks * MAX_BLOCKS_PER_CHUNK)
// is structurally <= DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE because peel caps
// n_chunks at DAMACY_MAX_CHUNKS_PER_WAVE. The grow path relies on this so
// it never has to enforce a runtime ceiling. `static_assert` is a C11
// keyword in C and works under C++17; works in both translation units.
#ifdef __cplusplus
static_assert((uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE *
                  DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK <=
                DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE,
              "wave substream ceiling must cover peel cap");
// d_block_chunk_map packs chunk_idx into the upper 16 bits; the GPU
// kernel unpacks via `packed >> 16` and indexes d_chunks/d_sample_plans
// directly. Raising the cap past 0xFFFFu silently truncates.
static_assert(DAMACY_MAX_CHUNKS_PER_WAVE <= 0xFFFFu,
              "DAMACY_MAX_CHUNKS_PER_WAVE must fit in 16 bits");
#else
_Static_assert(
  (uint64_t)DAMACY_MAX_CHUNKS_PER_WAVE* DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK <=
    DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE,
  "wave substream ceiling must cover peel cap");
// d_block_chunk_map packs chunk_idx into the upper 16 bits; the GPU
// kernel unpacks via `packed >> 16` and indexes d_chunks/d_sample_plans
// directly. Raising the cap past 0xFFFFu silently truncates.
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
