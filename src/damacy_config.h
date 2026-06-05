// Internal config helpers: dtype mapping, validation, and sizing resolvers.
#pragma once

#include "damacy.h"
#include "dtype/dtype.h"

#include <stdint.h>

uint32_t
damacy_dtype_bpe(enum damacy_dtype dt);

// 1 if the assemble kernel will accept (src, dst).
int
cast_path_supported(enum damacy_dtype dst, enum dtype src);

enum damacy_status
validate_config(const struct damacy_config* cfg);

// 0 → DAMACY_DEFAULT_CHUNK_UNCOMPRESSED_BYTES; clamped to DAMACY_MAX_*.
uint64_t
resolve_max_chunk_uncompressed(const struct damacy_config* cfg);

// 0 → DAMACY_DEFAULT_READ_OP_MAX_BYTES.
uint64_t
resolve_max_read_op_bytes(const struct damacy_config* cfg);

// 0 → DAMACY_DEFAULT_HOST_BUFFER_WAVES; clamped to
// [DAMACY_N_WAVES, DAMACY_MAX_HOST_BUFFER_WAVES].
uint8_t
resolve_host_buffer_waves(const struct damacy_config* cfg);

// 0 → DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE; clamped to
// DAMACY_HARD_MAX_CHUNKS_PER_WAVE.
uint32_t
resolve_max_chunks_per_wave(const struct damacy_config* cfg);

// 0 → DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK; clamped to
// DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK.
uint32_t
resolve_max_substreams_per_chunk(const struct damacy_config* cfg);

// Returns cfg->tuning.metadata_io_concurrency after validate_config has checked
// that it is positive and within the host concurrency bound.
uint32_t
resolve_metadata_io_concurrency(const struct damacy_config* cfg);

// Explicit config (ON/OFF) wins; AUTO defers to DAMACY_GDS_ENABLE=1.
// damacy_create rejects with DAMACY_INVAL when this resolves to 1 but
// libcufile.so.0 is not loadable / cuFileDriverOpen fails.
uint8_t
resolve_enable_gds(const struct damacy_config* cfg);

// Copies cfg->sample_shape[0..rank) into *out_shape and writes the
// rank into *out_rank. Returns DAMACY_INVAL if sample_rank is 0 or
// exceeds DAMACY_MAX_RANK, or if any dim is <= 0.
enum damacy_status
resolve_sample_shape(const struct damacy_config* cfg,
                     int64_t* out_shape,
                     uint8_t* out_rank);

// product(sample_shape) × samples_per_batch × dtype_bpe(dtype). Writes the
// value into *out_bytes. Returns DAMACY_INVAL on a bad sample_shape /
// rank (same conditions as resolve_sample_shape).
enum damacy_status
resolve_sample_volume_bytes(const struct damacy_config* cfg,
                            uint64_t* out_bytes);
