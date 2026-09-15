# Pipeline composition

A pipeline receives a planner, an executor, an output specification, and queue
limits. The planner owns metadata preparation; the executor owns bulk reads,
decoding, and output buffers. CPU and CUDA execution use the same rectangular
and indexed queries and prepared-plan contract.

## Construct a CPU pipeline

```python
import damacy
import numpy as np

metadata_reader = damacy.FileMetadataReader(concurrency=64)
chunk_reader = damacy.FileReader(workers=8, max_inflight_reads=4096)
metadata = damacy.ZarrMetadata(
    reader=metadata_reader,
    cache=damacy.MetadataCache(array_entries=256, shard_index_entries=8192),
)
planner = damacy.ChunkPlanner(
    metadata=metadata,
    limits=damacy.PlanLimits(
        max_chunks=16384,
        max_chunk_bytes=2 << 20,
        max_shards_per_sample=64,
        max_plan_bytes=64 << 20,
    ),
)
executor = damacy.CpuExecutor(
    reader=chunk_reader,
    limits=damacy.CpuLimits(
        max_memory_bytes=1 << 30,
        decode_workers=8,
        max_encoded_chunk_bytes=4 << 20,
        max_decoded_chunk_bytes=2 << 20,
    ),
)
output = damacy.BatchSpec(samples=2, shape=(64, 256, 256), dtype="f32")
queues = damacy.QueueLimits(lookahead_samples=4, prepared_batches=2)

with damacy.Pipeline(
    planner=planner, executor=executor, output=output, queues=queues
) as pipeline:
    pipeline.push([
        damacy.Sample(uri=uri, aabb=[(0, 64), (0, 256), (0, 256)])
        for uri in ["/data/image-1.zarr/0", "/data/image-2.zarr/0"]
    ])
    with pipeline.pop() as batch:
        array = np.from_dlpack(batch)
        assert array.shape == (2, 64, 256, 256)
        assert batch.info.device_type == damacy.DeviceType.CPU
        del array
```

The paths must name little-endian numeric Zarr v3 **arrays**. An NGFF group is
not resolved to a level automatically. The examples assume decoded chunks no larger than 2 MiB.

`FileMetadataReader` supplies small asynchronous metadata reads. `ZarrMetadata`
supplies Zarr interpretation and cache capacities; `MetadataCache` is a capacity
configuration. The planner fetches array descriptions and shard indexes on
demand across the URIs in the sample stream. It publishes plans that own their
paths, source descriptions, exact encoded ranges, and result operations.
Metadata caches can evict entries after preparation finishes.

`FileReader` has a separate queue for bulk chunk data. The CPU executor uses
ordinary RAM for input, decoded chunks, codec workspace, and output. It decodes
each distinct source chunk once per batch, then copies its intersections into
all samples that use it. There is no persistent decoded-chunk cache.

## Select CUDA execution

In a CUDA-enabled build, replace the executor with:

```python
executor = damacy.CudaExecutor(
    reader=chunk_reader,
    device=0,
    limits=damacy.CudaLimits(max_gpu_memory_bytes=1 << 30),
)
```

The remaining pipeline arguments and samples stay the same. Consume the
result with `torch.from_dlpack(batch)`. `device=0` retains that device's primary
CUDA context; `device=None` captures the caller's current context, which must
already exist. An externally managed context must outlive retained results.

CUDA execution owns read alignment/coalescing, Blosc layout probes, staging
buffers, streams, nvCOMP decoding, and GPU assembly. Those details do not occur
in prepared plans. The existing `Pipeline(Config(...))` constructor remains a
CUDA convenience adapter. It cannot select the CPU executor.

The `numa_strategy` argument to `CudaExecutor` controls its pinned buffers and
execution worker.
The injected readers inherit the caller's host affinity when their workers
start. The legacy `Config.numa_strategy` also applies while constructing those
readers, preserving placement of the complete legacy pipeline.

## Indexed queries

`IndexQuery(uri, selection=...)` accepts one index array or contiguous `slice`
per source axis. For a `(z, y, x)` array:

```python
query = damacy.IndexQuery(
    uri="/data/image.zarr/0",
    selection=([7, 2, 7], slice(16, 80), [100, 3, 40, 3]),
)
output = damacy.BatchSpec(samples=1, shape=query.shape, dtype="f32")
assert query.shape == (3, 64, 4)

with damacy.Pipeline(
    planner=planner, executor=executor, output=output,
    queues=damacy.QueueLimits(lookahead_samples=2),
) as pipeline:
    pipeline.push([query])
    with pipeline.pop() as batch:
        result = np.from_dlpack(batch)  # CPU executor
        del result
```

Index arrays select independently along each dimension: their Cartesian product
fills the tensor, matching MATLAB's per-dimension indexing or NumPy's `np.ix_`.
Caller order and repeated indices are preserved. A singleton index array keeps
its axis. A tuple such as `(7, 2)` is an index array in `IndexQuery`; use a slice
for a contiguous range. `Sample.aabb` retains its existing interval syntax.

Indices must be nonnegative integers below `2**63 - 1`. Slices use half-open
bounds, require an explicit stop, and accept a step of one. A `range` object can
supply a strided or reversed index array. Empty selections are rejected. Source
bounds are validated when metadata arrives; out-of-bounds selections raise
`InvalidArgument` from `pop`. Each query's result shape must match `BatchSpec`.
Rectangular and indexed queries can share a batch when their result shapes match.

Python copies index values into the immutable query at construction. Native
admission copies accepted queries, and prepared plans own their index data.
The planner enumerates only selected shards and chunks, including when indices
span large gaps. `max_chunks` counts each selected chunk once per sample;
repeated indices inside that chunk do not consume extra chunk entries.

The C API uses `damacy_sample.rank` and one tagged `damacy_axis_selection`
per axis. C callers must rebuild against the updated headers and explicitly
set every active axis to `DAMACY_AXIS_INTERVAL` or `DAMACY_AXIS_INDICES`:

```c
int64_t rows[] = {7, 2, 7};
struct damacy_sample query = {
    .uri = "/data/image.zarr/0",
    .rank = 2,
    .axes = {
        {.kind = DAMACY_AXIS_INDICES, .indices = {.values = rows, .count = 3}},
        {.kind = DAMACY_AXIS_INTERVAL, .interval = {.beg = 4, .end = 8}},
    },
};
```

This requests a `(3, 4)` sample. The union's `kind` selects its active member;
there is no default kind. A zero or unknown tag, a null or empty index array,
or an interval with negative, empty, or reversed bounds returns `DAMACY_INVAL`
from `damacy_push`. Each axis's length must match the configured sample shape.
Only `axes[0..rank)` are active, in the Zarr array's stored axis order. Preparation
derives the bounding AABB from these selections.

`damacy_push` copies the URI and index values for its consumed prefix. The
unconsumed suffix remains caller-owned and can be retried.

## Limits and backpressure

Sizes are bytes, with positive explicit limits. `CudaLimits.max_index_bytes`
also accepts zero to disable indexed CUDA queries. Defaults come from the
Python value objects.

| Setting | Scope |
| --- | --- |
| `BatchSpec` | Samples per batch, per-sample output shape, and `f32` or `bf16` output dtype. |
| `QueueLimits.lookahead_samples` | Sample requests waiting for metadata/preparation; at least one full batch. |
| `QueueLimits.prepared_batches` | Complete owned plans waiting for execution. |
| `MetadataCache` | Number of array descriptions and shard indexes retained by preparation. |
| `PlanLimits.max_chunks` | Chunk uses per batch, including chunks used by multiple samples. |
| `PlanLimits.max_chunk_bytes` | Decoded source bytes per chunk, before output conversion. |
| `PlanLimits.max_shards_per_sample` | Maximum number of shard files touched by one sample. |
| `PlanLimits.max_plan_bytes` | Owned storage per prepared plan, including index arrays; also caps copied index data per queued query. |
| `CpuLimits.max_memory_bytes` | CPU executor's input, decoded, codec-workspace, and two output buffers. |
| `CpuLimits.decode_workers` | Total decoding/assembly workers, including the calling scheduler thread. |
| `FileReader.workers` | Bulk I/O workers, separate from decoding workers. |
| `FileReader.max_inflight_reads` | Bulk read capacity; execution respects this bound and retries saturation. |
| `CudaLimits` | GPU memory and execution geometry, plus the CUDA codec-layout cache capacity. |
| `CudaLimits.max_index_bytes` | Device index storage per batch: eight bytes per index across all indexed axes and samples. Default 64 MiB. |

CUDA reserves index storage for its two execution slots within
`max_gpu_memory_bytes`. Each slot allocates the smaller of `max_index_bytes`
and the maximum index data possible for the configured output shape. An
identical amount of pinned host staging is allocated. A query exceeding the
index capacity raises `BudgetExceeded`. `Config.max_index_bytes` provides the
same setting through the CUDA convenience adapter.

For injected pipelines, define `floor = queues.lookahead_samples + output.samples`.
The metadata cache requires at least `floor` array entries and
`floor * plan_limits.max_shards_per_sample` shard-index entries. These floors
cover pending samples and the batch being prepared. Queued plans own their
metadata and do not pin cache entries. The legacy `Config` adapter retains its
older, stricter cache validation.

CPU memory admission includes a conservative allowance for Blosc scratch
storage. `Stats.host_bytes_committed` reports that reservation, which can exceed
the bytes actually touched. It excludes metadata caches, prepared plans, reader
queues, thread stacks, and allocator overhead: it is not a process RSS limit.
Queued plan storage is bounded separately by
`prepared_batches * max_plan_bytes`, with up to two accepted plans and one plan
being built in addition. Retaining results across repeated pipeline restarts
also retains their allocations outside the new executor's budget.

Both executors have two output buffers. A batch or a DLPack consumer holds its
buffer until all references are released. Holding both buffers pauses output
production. Release views promptly, or copy the result when it must be retained
while subsequent batches continue. Increasing `prepared_batches` increases
preparation capacity, not the output pool.

Only complete batches are emitted. Trailing requests that do not fill a batch
are not returned. Invalid sample geometry is rejected at push; metadata, codec,
and memory-limit failures normally surface at pop. A terminal execution error
requires closing the pipeline and constructing another one.

## Lifetimes and interoperation

Use a context manager or call `Pipeline.close()`. Python retains the injected
components and their dependencies. Planners, executors, metadata providers,
and metadata readers each serve one active pipeline; simultaneous reuse is
rejected. They may be reused after close. Closing stops pending work and wakes
blocked pops. Reuse creates fresh active caches and execution resources.

`np.from_dlpack(batch)` produces a CPU view; `torch.from_dlpack(batch)` accepts
CPU or CUDA results. A DLPack view remains valid after releasing the `Batch` and
closing the pipeline. Releasing the Python batch does not force a live view's
buffer back into the output pool. NumPy does not support `bf16` through DLPack;
use `f32` or a consumer with bfloat16 support.

CPU batches report DLPack device `(1, 0)` and reject a stream argument other than
`None`. CUDA batches report `(2, device_id)` and preserve the existing stream
handoff. `BatchInfo.data` is the address on either device; `device_ptr` remains
an alias for compatibility. CPU data is ready when `pop()` returns and has no
ready stream. Use the device type before interpreting a raw pointer.

The C factories and configuration structures are declared in
`src/damacy_pipeline.h`; push/pop and batch functions remain in `src/damacy.h`.
C pipelines borrow components until `damacy_shutdown`. Destroy dependencies in
reverse construction order. `damacy_batch_retain` / `damacy_batch_release`
manage result lifetimes independently of a pipeline; release each acquired
reference once. After destroying a pipeline, use `damacy_batch_release` for
retained results rather than a function requiring the old pipeline pointer.
Operation tables and prepared-plan records are private implementation APIs.

## Build without CUDA

On Linux, install C/C++ build tools, CMake, Ninja, pkg-config, liburing, zstd,
and C-Blosc development packages. For example, Ubuntu packages are
`build-essential cmake ninja-build pkg-config python3-dev liburing-dev libzstd-dev libblosc-dev`.

```sh
cmake --preset cpu
cmake --build build
ctest --test-dir build --output-on-failure
```

The examples use NumPy (`pip install numpy`). The fixture generator uses `uv`.
Enable `DAMACY_PYTHON=ON` and provide a Python 3.11+ interpreter with pytest,
pytest-cov, and NumPy to include Python tests.
Build a CPU wheel with:

```sh
pip install . --config-settings=cmake.define.DAMACY_CUDA=OFF
```

The extension has no CUDA or nvCOMP dependency in this configuration.
`CudaExecutor` reports that CUDA support was not built. CPU builds retain the
Linux io_uring metadata requirements. CUDA support is enabled by default;
turning it on builds both executors and requires the CUDA toolkit, nvCOMP, and
a runtime NVIDIA driver. GDS requires a CUDA build.

## Future queries

Rectangular and indexed queries copy or gather existing voxels. Spatial
queries will describe a fixed output tensor, a transform
from output coordinates to source space, and sampler settings including
interpolation, antialiasing, and boundary handling.

For NGFF images, resolving a spatial query will choose an appropriate source
level from the requested sampling scale before enumerating chunks. The plan
will record the chosen array and transform. Resampling also needs interpolation
halos and dependencies across chunks. These are later planner and executor
operations; they are not implemented by the current box query.
