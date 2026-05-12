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

// Upper bound on cfg.n_compute_threads. The compute pool fans the
// blosc1 host parse out across CPU cores; 32 is well past the largest
// EPYC/Threadripper core count typical in a host today, and rejecting
// pathological configs early keeps threadpool_new from clamping or
// failing later. 0 is allowed (synchronous parse on the caller).
#define DAMACY_MAX_COMPUTE_THREADS 32u

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
#define DAMACY_BLOSC_MAX_TYPESIZE 8u

// Defensive cap on header.nbytes parsed from a blosc1 chunk. Prevents
// overflow in the nblocks ceil-div for adversarial inputs, independent
// of the runtime DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES cap (which gates
// the planner, not the parser).
#define DAMACY_BLOSC_MAX_CHUNK_UNCOMPRESSED_BYTES (16ull << 20) // 16 MB

// Worst-case substream count per wave for blosc1-zstd: 1 substream per
// blosc-block. blosc1-lz4 splits each block into `typesize` substreams,
// so its per-wave cap scales with the runtime max_bytes_per_element knob
// (resolve_max_bpe(cfg)) — see lz4_subs_per_wave().
#define DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE                                    \
  (DAMACY_MAX_CHUNKS_PER_WAVE * DAMACY_BLOSC_MAX_BLOCKS_PER_CHUNK)
// Memcpy + (bit)unshuffle ops cap: every chunk could be MEMCPY/SHUFFLE'd.
#define DAMACY_MAX_BLOSC_MEMCPY_OPS_PER_WAVE DAMACY_MAX_CHUNKS_PER_WAVE
#define DAMACY_MAX_BLOSC_SHUFFLE_OPS_PER_WAVE DAMACY_MAX_CHUNKS_PER_WAVE

static inline uint64_t
lz4_subs_per_wave(uint8_t max_bpe)
{
  return (uint64_t)DAMACY_MAX_BLOSC_ZSTD_SUBS_PER_WAVE * (uint64_t)max_bpe;
}
