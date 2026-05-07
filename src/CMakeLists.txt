# Helpers (add_src_lib, add_platform_sources) live in cmake/Helpers.cmake.

# Logging — internal log_* macros (log/log.h) + public callback-registry
# control surface (damacy_log.h). util/prelude.h's CHECK macros lower
# onto log_error, so any module that uses prelude must link this target.
add_src_lib(log SOURCES log/log.h log/log.c ${PROJECT_SOURCE_DIR}/src/damacy_log.h)

# Foundational types
add_src_lib(dtype     SOURCES dtype/dtype.c dtype/dtype.h)
add_src_lib(dimension SOURCES dimension/dimension.c dimension/dimension.h)

# Utilities
add_src_lib(slice  SOURCES util/slice.c  util/slice.h)
add_src_lib(strbuf SOURCES util/strbuf.c util/strbuf.h)
add_src_lib(json   SOURCES util/json.c   util/json.h LINKS slice)
add_src_lib(json_writer SOURCES util/json_writer.c util/json_writer.h LINKS strbuf)
add_src_lib(crc32c SOURCES util/crc32c.c util/crc32c.h LINKS platform)
add_src_lib(hash   SOURCES util/hash.c   util/hash.h)
add_src_lib(lru    SOURCES util/lru.c    util/lru.h LINKS log)

# Platform abstraction (posix-only for now; chucky-style platform-split lives here)
add_src_lib(platform    SOURCES platform/platform.h    LINKS Threads::Threads)
add_platform_sources(platform platform/platform)

add_src_lib(platform_io SOURCES platform/platform_io.h LINKS platform)
add_platform_sources(platform_io platform/platform_io)

# Concurrency primitives
add_src_lib(threadpool SOURCES threadpool/threadpool.c threadpool/threadpool.h
    LINKS platform log
)

add_src_lib(io_queue SOURCES io_queue/io_queue.h LINKS platform_io log)
add_platform_sources(io_queue io_queue/io_queue)

# Read-side store
add_src_lib(store SOURCES store/store.h store/store.c store/store_fs.h store/store_fs.c
    LINKS io_queue platform_io strbuf log
)

# Zarr v3 metadata + shard-index parsers (cache-layer building blocks).
add_src_lib(zarr SOURCES zarr/zarr_metadata.h zarr/zarr_metadata.c
    zarr/zarr_shard_index.h zarr/zarr_shard_index.c
    LINKS dtype json strbuf crc32c log
)

# damacy-side metadata cache (LRU around zarr_metadata_parse).
# damacy.h is header-included for the damacy_status enum; no link needed
# (no symbols referenced).
add_src_lib(zarr_meta_cache SOURCES zarr/zarr_meta_cache.h zarr/zarr_meta_cache.c
    LINKS zarr lru store strbuf hash log
)

# damacy-side shard index cache (LRU around zarr_shard_index_parse).
add_src_lib(zarr_shard_cache SOURCES zarr/zarr_shard_cache.h zarr/zarr_shard_cache.c
    LINKS zarr lru store strbuf hash log
)

# Planner: samples + cached metadata → page-aligned read_ops + chunk_plans.
add_src_lib(planner SOURCES planner/planner.h planner/planner.c
    LINKS zarr_meta_cache zarr_shard_cache zarr dtype platform strbuf log
)

# GPU pieces — add_cuda_lib is a no-op under DAMACY_FUZZ.
add_cuda_lib(decoder
    SOURCES decoder/decoder.h
            decoder/decoder_check.h
            decoder/decoder_zstd.h decoder/decoder_zstd.c
            decoder/decoder_lz4.h  decoder/decoder_lz4.c
            decoder/decoder_memcpy.h decoder/decoder_memcpy.cu
            decoder/blosc1.h decoder/blosc1.cu
            decoder/shuffle.h decoder/shuffle.cu
            decoder/bitshuffle.h decoder/bitshuffle.cu
    LINKS CUDA::cudart_static CUDA::cuda_driver nvcomp::nvcomp log zarr
)

# Bytewise gather kernel: chunk windows → output batch tensor.
# CUDA::cudart_static for the runtime-API hooks the `<<<...>>>` syntax
# lowers to; CUDA::cuda_driver for CUstream / CUdeviceptr at the
# public surface.
add_cuda_lib(assemble SOURCES assemble/assemble.h assemble/assemble.cu
    LINKS CUDA::cudart_static CUDA::cuda_driver log
)

# Public damacy surface. damacy.c uses the CUDA driver API exclusively
# (cuMem*, cuStream*, cuEvent*); the runtime API is only pulled in
# transitively via decoder/assemble (nvcomp's headers + the kernel
# launch in assemble.cu).
add_cuda_lib(damacy SOURCES damacy.h damacy.c damacy_status.c
    LINKS platform planner zarr_meta_cache zarr_shard_cache zarr store
          decoder assemble dtype log strbuf
          CUDA::cuda_driver
)
