# Pipeline composition

A pipeline receives a planner, an executor, an output specification, and queue
limits. The planner owns metadata preparation; the executor owns bulk reads,
decoding, and output buffers. CPU and CUDA execution use the same rectangular
queries and prepared-plan contract.

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
        max_memory_bytes=3 << 30,
        decode_workers=8,
        max_encoded_chunk_bytes=4 << 20,
        max_decoded_chunk_bytes=2 << 20,
        chunks_per_input_buffer=256,
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

## Limits and backpressure

Sizes are bytes, with positive explicit limits. Zero is invalid for these
capacities. Defaults come from the Python value objects.

| Setting | Scope |
| --- | --- |
| `BatchSpec` | Samples per batch, per-sample output shape, and `f32` or `bf16` output dtype. |
| `QueueLimits.lookahead_samples` | Sample requests waiting for metadata/preparation; at least one full batch. |
| `QueueLimits.prepared_batches` | Complete owned plans waiting for execution. |
| `MetadataCache` | Number of array descriptions and shard indexes retained by preparation. |
| `PlanLimits.max_chunks` | Chunk uses per batch, including chunks used by multiple samples. |
| `PlanLimits.max_chunk_bytes` | Decoded source bytes per chunk, before output conversion. |
| `PlanLimits.max_shards_per_sample` | Maximum number of shard files touched by one sample. |
| `PlanLimits.max_plan_bytes` | Owned storage per prepared plan. |
| `CpuLimits.max_memory_bytes` | CPU executor's buffers, codec workspace, active read plans, and temporary read-planning scratch. |
| `CpuLimits.decode_workers` | Total decoding/assembly workers, including the calling scheduler thread. |
| `CpuLimits.chunks_per_input_buffer` | Chunks each of the two encoded-input buffers holds; from `decode_workers` to 16384. |
| `FileReader.workers` | Bulk I/O workers, separate from decoding workers. |
| `FileReader.max_inflight_reads` | Bulk read capacity; execution respects this bound and retries saturation. |
| `CudaLimits` | GPU memory and execution geometry, plus the CUDA codec-layout cache capacity. |

For injected pipelines, define `floor = queues.lookahead_samples + output.samples`.
The metadata cache requires at least `floor` array entries and
`floor * plan_limits.max_shards_per_sample` shard-index entries. These floors
cover pending samples and the batch being prepared. Queued plans own their
metadata and do not pin cache entries. The legacy `Config` adapter retains its
older, stricter cache validation.

The CPU executor merges adjacent or overlapping encoded ranges within each
shard, then interleaves the reads across shards. It decodes each unique source
chunk once per batch, including when several output samples use that chunk.
Each of its two encoded-input buffers holds `chunks_per_input_buffer` chunks
(default 256). A merged read contains at most that many chunks, so reads merge
even with one decode worker; submission also respects the reader limit.
Decoder workspaces remain per worker.

The two input buffers reserve
`2 * chunks_per_input_buffer * max_encoded_chunk_bytes` bytes, plus per-chunk
bookkeeping. With the defaults, 256 chunks and a 4 MiB encoded-chunk bound,
that is 2 GiB before decoder workspaces and output buffers. Set the bound to
match the largest encoded chunk expected, and include this reserve in
`max_memory_bytes`. If the budget is too small, starting the pipeline reports
`BUDGET` and logs how many bytes the input buffers, decoder workspaces, and
output buffers need.

CPU memory admission includes active read plans, temporary planning scratch,
and a conservative allowance for Blosc scratch storage.
`Stats.host_bytes_committed` reports current buffers, read plans, and codec
reservations, which can exceed the bytes actually touched. Temporary planning
scratch is checked against the cap and released before submission returns.
If an active batch's read plan temporarily prevents admission, submission
retries after that batch completes. A plan that cannot fit by itself reports
`BUDGET`.
The cap excludes metadata caches, prepared plans, reader queues, thread stacks,
and allocator overhead: it is not a process RSS limit.
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
components and their dependencies. Planners and executors each serve one
active pipeline; simultaneous reuse is rejected. They may be reused after
close. Metadata providers and metadata readers hold only settings. Each planner
copies them and builds its own caches, so any number of planners may share one.
Closing stops pending work and wakes blocked pops. Reuse creates fresh active
caches and execution resources.

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

On macOS, install the native dependencies with Homebrew:

```sh
brew install cmake ninja pkg-config zstd c-blosc uv
```

Then build and test on either platform:

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
`CudaExecutor` reports that CUDA support was not built. Linux retains its
io_uring metadata requirements. macOS uses a POSIX metadata worker pool,
with one worker per `metadata_io_concurrency`, and shared POSIX bulk reads.
NUMA placement and CPU affinity are unavailable. CUDA defaults off on macOS
and cannot be enabled. On either platform, `FileReader(workers=...)` and the
legacy `Config.n_io_threads` cannot exceed the online CPU count; larger values
raise `InvalidArgument`.
On Linux CUDA defaults on, builds both executors, and requires the CUDA toolkit,
nvCOMP, and a runtime NVIDIA driver. GDS requires a CUDA build.

## Future queries

The first milestone implements rectangular copy/cast queries. Index-array
queries can add ordered selections and repeated indices without a separate
stencil type. Spatial queries will describe a fixed output tensor, a transform
from output coordinates to source space, and sampler settings including
interpolation, antialiasing, and boundary handling.

For NGFF images, resolving a spatial query will choose an appropriate source
level from the requested sampling scale before enumerating chunks. The plan
will record the chosen array and transform. Resampling also needs interpolation
halos and dependencies across chunks. These are later planner and executor
operations; they are not implemented by the current box query.
