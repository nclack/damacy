# Helpers (add_src_lib, add_platform_sources) live in cmake/Helpers.cmake.

# Logging — internal log_* macros (log/log.h) + public callback-registry
# control surface (damacy_log.h). util/prelude.h's CHECK macros lower
# onto log_error, so any module that uses prelude must link this target.
add_src_lib(log SOURCES log/log.h log/log.c ${PROJECT_SOURCE_DIR}/src/damacy_log.h LINKS platform)

# Phase 6 observability: NVTX wrappers + named CUDA streams.
# DAMACY_NVTX_ENABLED=ON (default for non-fuzz builds) links NVTX v3
# against CUDA::nvtx3. OFF uses no-op wrapper definitions. Toggle via:
#   cmake -B build -DDAMACY_NVTX_ENABLED=ON|OFF
# Fuzz builds always OFF (no CUDA, and the timeline isn't interesting
# in libFuzzer). The .c is always in the source list (gives the static
# lib a linker language); its body is #if-gated on DAMACY_NVTX_ENABLED.
add_cuda_lib(damacy_nvtx SOURCES nvtx/nvtx.h nvtx/nvtx.c LINKS CUDA::cuda_driver platform)
if(DAMACY_CUDA AND DAMACY_NVTX_ENABLED)
    target_link_libraries(damacy_nvtx PUBLIC CUDA::nvtx3)
    target_compile_definitions(damacy_nvtx PUBLIC DAMACY_NVTX_ENABLED=1)
endif()

# Foundational types
add_src_lib(dtype     SOURCES dtype/dtype.c dtype/dtype.h)

# Utilities
add_src_lib(slice  SOURCES util/slice.c  util/slice.h)
add_src_lib(strbuf SOURCES util/strbuf.c util/strbuf.h)
add_src_lib(json   SOURCES util/json.c   util/json.h LINKS slice)
add_src_lib(json_writer SOURCES util/json_writer.c util/json_writer.h LINKS strbuf)
add_src_lib(crc32c SOURCES util/crc32c.c util/crc32c.h LINKS platform)
add_src_lib(hash   SOURCES util/hash.c   util/hash.h)
add_src_lib(lru    SOURCES util/lru.c    util/lru.h LINKS log)
add_src_lib(pool   SOURCES util/pool.c   util/pool.h LINKS log platform)
add_src_lib(path_intern SOURCES util/path_intern.c util/path_intern.h LINKS hash)

# Platform abstraction: POSIX sources on Linux and macOS. macOS has no NUMA
# or CPU affinity controls, so it builds the numa.darwin.c stub instead.
# platform.posix.c uses dlopen for platform_dl*; numa.posix.c uses libnuma via
# platform_dl* so a single binary works on hosts with or without libnuma installed.
add_src_lib(platform
    SOURCES platform/platform.h platform/numa.h
    LINKS Threads::Threads ${CMAKE_DL_LIBS}
)
add_platform_sources(platform platform/platform)
add_platform_sources(platform platform/numa)

# CUDA-side GPU→host-NUMA-node resolution + scope/affinity wrappers over
# the platform NUMA primitives. CUDA::cuda_driver for cuDeviceGetAttribute
# (HOST_NUMA_ID) and the BDF sysfs fallback.
add_src_lib(numa SOURCES numa/numa.h numa/affinity.c LINKS log platform)
add_cuda_lib(numa_cuda SOURCES numa/numa_cuda.h numa/numa.c
    LINKS numa log platform CUDA::cuda_driver
)

add_src_lib(platform_io SOURCES platform/platform_io.h LINKS platform)
add_platform_sources(platform_io platform/platform_io)

# Concurrency primitives
add_src_lib(threadpool SOURCES threadpool/threadpool.c threadpool/threadpool.h
    LINKS platform log
)

# Single-worker scheduler: drives a caller-supplied step on a platform
# thread; pairs with a shared mutex + cv so external callers (damacy_pop
# etc.) can lock / wait / broadcast in lockstep with the worker.
add_src_lib(scheduler SOURCES scheduler/scheduler.h scheduler/scheduler.c
    LINKS platform log numa
)

add_src_lib(io_queue SOURCES io_queue/io_queue.h LINKS platform_io log numa)
add_platform_sources(io_queue io_queue/io_queue)

# Internal stats helpers (cumulative damacy_metric record / reset, plus a
# project-local monotonic clock). Shared by API and wave orchestration.
add_src_lib(damacy_stats SOURCES damacy_stats.h damacy_stats.c)

# Internal config helpers: dtype mapping, validate_config, explicit tuning
# defaults, and sizing resolvers. Shared by lifecycle, gpu_budget/, and wave/.
add_src_lib(damacy_config SOURCES damacy_config.h damacy_config.c
    LINKS dtype log platform
)

add_src_lib(query_selection SOURCES query/selection.h query/selection.c)

add_src_lib(lookahead SOURCES lookahead/lookahead.h lookahead/lookahead.c
    LINKS log platform query_selection
)

add_src_lib(prefetch_cache SOURCES prefetch/prefetch_cache.h prefetch/prefetch_cache.c
    LINKS lru log
)

add_src_lib(array_meta SOURCES prefetch/array_meta.h prefetch/array_meta.c
    LINKS prefetch_cache zarr store strbuf log
)

add_src_lib(shard_index SOURCES prefetch/shard_index.h prefetch/shard_index.c
    LINKS prefetch_cache zarr store strbuf hash log
)

add_src_lib(chunk_layout SOURCES prefetch/chunk_layout.h prefetch/chunk_layout.c
    LINKS prefetch_cache shard_index zarr zarr_chunk_layout store strbuf hash log
)

add_src_lib(prefetcher SOURCES prefetch/prefetcher.h prefetch/prefetcher.c
    LINKS prefetch_cache chunk_layout shard_index zarr hash lookahead platform log
)

# Runtime GPU-memory committer: tracks bytes committed against a
# configured cap. Single source of truth for "have we exceeded?" and
# "how many bytes are committed?" across wave_pool, batch_pool, and the
# observe-and-grow paths. The predictor that consumes the resolved
# geometry lives in wave_budget (gpu_budget_predict) — splitting them
# avoids the cycle between committer and per-component byte math.
add_cuda_lib(gpu_budget SOURCES gpu_budget/gpu_budget.h gpu_budget/gpu_budget.c
    LINKS log
)

# Two-slot batch state machine + lazy device-tensor pool. Owns the
# caller-visible minibatch lifetime; render_job owns planner scratch.
add_cuda_lib(batch_pool SOURCES batch_pool/batch_pool.h batch_pool/batch_pool.c
    LINKS dispatch_utils path_intern log CUDA::cuda_driver
)

# Planner output plus dispatch cursor for one sealed batch. V1 keeps
# one render job per batch slot.
add_cuda_lib(render_job SOURCES render_job/render_job.h render_job/render_job.c
    LINKS dispatch_utils path_intern store log CUDA::cuda_driver
)

# Input staging slot pool.
add_cuda_lib(input_slot SOURCES wave/input_slot.h wave/input_slot.c
    LINKS store CUDA::cuda_driver
)

add_cuda_lib(input_transfer SOURCES wave/input_transfer.h wave/input_transfer.c
    LINKS input_slot store damacy_nvtx CUDA::cuda_driver
)

# Pinned-host + device SOA fanout used by the zstd decoder batch path.
# Pure: no streams, no scheduler — alloc/free/upload/grow primitives.
add_cuda_lib(fanout SOURCES wave/fanout.h wave/fanout.c
    LINKS decoder gpu_budget log CUDA::cuda_driver
)

# GPU memory geometry for a wave: per-wave byte predictions, the
# resolver that picks per-wave extents from a configured cap, and the
# decoder-scratch grow path that shares the same caps math. No state;
# all functions pure aside from the decoder grow's
# cuStreamSynchronize.
add_cuda_lib(wave_budget
    SOURCES wave/input_transfer.h wave/wave_budget.h wave/wave_budget.c
    LINKS input_transfer decoder fanout dispatch_utils execution_plan assemble gpu_budget
          damacy_config log
          CUDA::cuda_driver
)

# Per-wave struct + alloc/destroy. Owns its dev_compressed,
# dev_decompressed, blosc1 parse scratch, nvcomp fanout SOA, and the 8
# per-stage CUevents. No streams, no scheduler — that lives in
# wave_pool.
add_cuda_lib(wave SOURCES wave/wave.h wave/wave.c
    LINKS decoder assemble fanout log CUDA::cuda_driver
)

# wave_pool: the conductor the scheduler drives. Owns both waves + the
# input_slot pool + 3 streams + the shared zstd decoder + decode anchor
# ring, and is the home of the per-stage submitters (kick_input /
# kick_decode / kick_assemble), input dispatch, and the wave_pool_advance
# state machine.
add_cuda_lib(wave_pool
    SOURCES wave/input_transfer.h wave/wave_input.h wave/wave_input.c wave/wave_pool.h wave/wave_pool.c
    LINKS wave input_slot input_transfer fanout wave_budget gpu_budget
          batch_pool decoder assemble dispatch_utils render_job store damacy_stats
          damacy_config damacy_nvtx log CUDA::cuda_driver
)

# Read-side store. store_fs.c is the host impl; store_fs_gds.c is an
# independent cuFile-backed store. DAMACY_ENABLE_GDS=OFF links the stub
# (store_fs_gds_create returns NULL); runtime GDS opt-in then fails INVAL.
add_src_lib(store
    SOURCES store/store.h store/store_internal.h store/store.c
            store/store_fs.h store/store_fs.c
            store/store_latency.h store/store_latency.c
    LINKS io_queue platform_io strbuf log lru pool hash m
)
if(DAMACY_CUDA AND DAMACY_ENABLE_GDS)
    target_sources(store PRIVATE store/store_fs_gds.h store/store_fs_gds.c)
    target_include_directories(store PRIVATE ${CuFile_INCLUDE_DIR})
    target_link_libraries(store PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
else()
    target_sources(store PRIVATE store/store_fs_gds.h store/store_fs_gds_stub.c)
endif()

if(APPLE)
    set(METADATA_STORE_IMPL store/metadata_store_async.posix.c)
else()
    set(METADATA_STORE_IMPL store/metadata_store_async.c)
endif()
add_src_lib(metadata_store_async
    SOURCES store/metadata_store_async.h store/metadata_store_async_common.h
            store/metadata_store_async_common.c ${METADATA_STORE_IMPL}
    LINKS store log platform
)
if(NOT APPLE)
    target_link_libraries(metadata_store_async PUBLIC PkgConfig::LIBURING)
endif()
if(TARGET numa)
    target_link_libraries(metadata_store_async PUBLIC numa)
    target_compile_definitions(
        metadata_store_async
        PRIVATE DAMACY_METADATA_ASYNC_NUMA=1
    )
endif()
target_link_libraries(array_meta PUBLIC metadata_store_async)
target_link_libraries(shard_index PUBLIC metadata_store_async)
target_link_libraries(chunk_layout PUBLIC metadata_store_async)

# Zarr v3 metadata + shard-index parsers (cache-layer building blocks).
add_src_lib(zarr SOURCES zarr/zarr_metadata.h zarr/zarr_metadata.c
    zarr/zarr_shard_index.h zarr/zarr_shard_index.c
    zarr/sample_shard_iterator.h zarr/sample_shard_iterator.c
    LINKS dtype json strbuf crc32c log
)

# Per-array blosc1 chunk-layout probe (16-byte header read via store).
# Used by the planner to pre-size decoder/fanout caps and to feed the
# GPU parse kernels their per-chunk blocksize / nblocks / shuffle bits.
add_src_lib(zarr_chunk_layout SOURCES zarr/zarr_chunk_layout.h zarr/zarr_chunk_layout.c
    LINKS zarr store log
)

add_src_lib(dispatch_utils
    SOURCES executor/coalesce.h executor/coalesce.c
            executor/group_chunks.h executor/group_chunks.c
            executor/read_op_sort.h executor/read_op_sort.c
    LINKS path_intern log damacy_config
)

add_src_lib(planner SOURCES planner/planner.h planner/planner.c
    LINKS plan_builder execution_plan prefetch_cache log
)

# GPU pieces — add_cuda_lib is a no-op under DAMACY_FUZZ.
add_cuda_lib(decoder
    SOURCES decoder/decoder.h
            decoder/decoder_check.h
            decoder/decoder_zstd.h decoder/decoder_zstd.c
            decoder/decoder_memcpy.h decoder/decoder_memcpy.cu
            decoder/blosc1.h
            decoder/blosc1_parse.h decoder/blosc1_parse.cu
            decoder/launch_check.h decoder/launch_check.cu
            decoder/status_reduce.h decoder/status_reduce.cu
    LINKS CUDA::cudart_static CUDA::cuda_driver nvcomp::nvcomp log zarr
          dispatch_utils assemble zarr_chunk_layout
)

# Bytewise gather kernel: chunk windows → output batch tensor.
# CUDA::cudart_static for the runtime-API hooks the `<<<...>>>` syntax
# lowers to; CUDA::cuda_driver for CUstream / CUdeviceptr at the
# public surface.
add_cuda_lib(assemble SOURCES assemble/assemble.h assemble/assemble.cu
    LINKS CUDA::cudart_static CUDA::cuda_driver log
)

add_src_lib(prepared_plan SOURCES planner/plan.h planner/plan.c
    LINKS query_selection
)

add_src_lib(plan_builder SOURCES planner/plan_builder.h planner/plan_builder.c
    LINKS prepared_plan prefetch_cache shard_index zarr damacy_config strbuf hash
)

add_src_lib(execution_plan SOURCES executor/dispatch.h executor/dispatch.c
    LINKS dispatch_utils prepared_plan path_intern
)

if(NOT DAMACY_FUZZ)
    add_src_lib(pipeline_components SOURCES pipeline/components.h pipeline/components.c damacy_status.c
        LINKS damacy_config store platform log
    )
    add_src_lib(metadata_planner SOURCES pipeline/zarr_planner.c
        LINKS pipeline_components plan_builder prefetcher array_meta shard_index
              metadata_store_async lookahead
    )
    add_src_lib(cpu_executor SOURCES executor/cpu_executor.c executor/cpu_read_plan.h
        LINKS pipeline_components prepared_plan dispatch_utils threadpool damacy_stats
              PkgConfig::ZSTD PkgConfig::BLOSC
    )
    add_src_lib(damacy
        SOURCES damacy.h damacy_pipeline.h damacy_internal.h
                damacy_lifecycle.c damacy_push.c damacy_plan.c damacy_pop.c
                damacy_scheduler.c
        LINKS pipeline_components metadata_planner cpu_executor scheduler damacy_stats
    )
    if(DAMACY_CUDA)
        add_cuda_lib(cuda_executor
            SOURCES executor/cuda_executor.h executor/cuda_executor.c executor/cuda_geometry.inc
            LINKS pipeline_components execution_plan numa_cuda gpu_budget batch_pool
                  render_job wave_pool wave_budget damacy_config zarr_chunk_layout
                  CUDA::cuda_driver
        )
        target_link_libraries(damacy PUBLIC cuda_executor)
        target_compile_definitions(damacy PUBLIC DAMACY_HAS_CUDA=1)
    else()
        target_sources(damacy PRIVATE executor/cuda_stub.c)
    endif()
endif()
