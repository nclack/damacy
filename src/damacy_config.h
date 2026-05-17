// Internal config helpers: dtype mapping, validation, and the
// "0 → default" knob resolvers.
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

// 0 → DAMACY_DEFAULT_MAX_GPU_MEMORY_BYTES.
uint64_t
resolve_max_gpu_memory(const struct damacy_config* cfg);

// 0 → DAMACY_DEFAULT_HOST_BUFFER_WAVES; clamped to
// [DAMACY_N_WAVES, DAMACY_MAX_HOST_BUFFER_WAVES].
uint8_t
resolve_host_buffer_waves(const struct damacy_config* cfg);

// cfg->tuning.enable_gds OR env DAMACY_GDS_ENABLE=1. damacy_create
// rejects with DAMACY_INVAL when this resolves to 1 but libcufile.so.0
// is not loadable / cuFileDriverOpen fails.
uint8_t
resolve_enable_gds(const struct damacy_config* cfg);
