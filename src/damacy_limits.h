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

// In-flight wave count. Fixed; sets the minimum host_buffer_waves and
// the device-side decode concurrency depth.
#define DAMACY_N_WAVES 2

// Caller-visible batch slots. The current orchestrator is double-buffered:
// one slot may be held by the caller while the other accumulates/renders.
#define DAMACY_N_BATCH_SLOTS 2

// Default input staging slot count, in waves.
#define DAMACY_DEFAULT_HOST_BUFFER_WAVES DAMACY_N_WAVES

// Upper bound on cfg.host_buffer_waves.
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

// Defensive cap on damacy_tuning.metadata_io_concurrency (async request
// depth). The default values live in damacy_tuning_defaults().
#define DAMACY_MAX_METADATA_IO_CONCURRENCY 4096u
// Sanity cap for blocking IO workers; they may exceed the CPU count.
#define DAMACY_MAX_IO_THREADS 1024u
#define DAMACY_DEFAULT_ARRAY_META_CACHE 64u
#define DAMACY_DEFAULT_SHARD_INDEX_CACHE 256u
#define DAMACY_DEFAULT_CHUNK_LAYOUT_CACHE 64u

// Declared upper bound on the shard count a single sample's AABB may
// intersect. Sizes the shard_index cache floor at config time
// (n_shard_index_cache >= lookahead_samples * max_shards_per_sample) and
// caps per-sample shard enumeration at runtime. Default pairs with the
// default shard_index cache (256) for a lookahead of >= 4.
#define DAMACY_DEFAULT_MAX_SHARDS_PER_SAMPLE 64u

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

#define DAMACY_MAX_SUBSTREAMS_PER_WAVE(chunks, substreams_per_chunk)           \
  ((uint32_t)((uint64_t)(chunks) * (uint64_t)(substreams_per_chunk)))

// Measured metadata-op latency histogram dimensions (see damacy_stats's
// metadata_op_latency). NKINDS matches the io_uring op kinds statx/open/read/
// close; NBUCKETS spans ~1ns..~tens of seconds on a log2 scale.
#define DAMACY_METADATA_OP_LATENCY_NKINDS 4
#define DAMACY_METADATA_OP_LATENCY_NBUCKETS 48
