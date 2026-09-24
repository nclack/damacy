# Damacy

[![build](https://github.com/nclack/damacy/actions/workflows/build.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/build.yml)
[![test](https://github.com/nclack/damacy/actions/workflows/test.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/test.yml)
[![fuzz](https://github.com/nclack/damacy/actions/workflows/fuzz.yml/badge.svg?branch=main&event=schedule)](https://github.com/nclack/damacy/actions/workflows/fuzz.yml)
[![codecov](https://codecov.io/gh/nclack/damacy/branch/main/graph/badge.svg)](https://codecov.io/gh/nclack/damacy)
[![bench](https://github.com/nclack/damacy/actions/workflows/bench.yml/badge.svg)](https://nclack.github.io/damacy/throughput/)
[![docs](https://github.com/nclack/damacy/actions/workflows/docs.yml/badge.svg)](https://nclack.github.io/damacy/)

Streamed assembly of n-dimensional tensors from Zarr sources into RAM or GPU memory.

Damacy loads array metadata and shard indexes, plans the chunks needed for a
batch, then reads, decodes, and assembles them with a CPU or CUDA executor.
Both paths support raw bytes, zstd, and Blosc-zstd and return contiguous,
DLPack-compatible tensors. CPU builds have no CUDA or nvCOMP dependency.

Source URIs identify concrete Zarr v3 arrays, including arrays inside
[NGFF](https://ngff.openmicroscopy.org/) multiscale images. Queries currently
select rectangular regions in array coordinates. Index queries, transformed
crops, and automatic NGFF level selection are planned extensions.

## CPU quick start

```python
import damacy
import numpy as np

metadata_reader = damacy.FileMetadataReader(concurrency=64)
chunk_reader = damacy.FileReader(workers=8, max_inflight_reads=4096)
metadata = damacy.ZarrMetadata(
    reader=metadata_reader,
    cache=damacy.MetadataCache(array_entries=256, shard_index_entries=8192),
)
planner = damacy.ChunkPlanner(metadata=metadata, limits=damacy.PlanLimits())
executor = damacy.CpuExecutor(
    reader=chunk_reader,
    limits=damacy.CpuLimits(max_memory_bytes=1 << 30, decode_workers=8),
)

with damacy.Pipeline(
    planner=planner,
    executor=executor,
    output=damacy.BatchSpec(samples=2, shape=(64, 256, 256), dtype="f32"),
    queues=damacy.QueueLimits(lookahead_samples=4, prepared_batches=2),
) as pipeline:
    pipeline.push([
        damacy.Sample(uri=uri, aabb=[(0, 64), (0, 256), (0, 256)])
        for uri in ["/data/image-1.zarr/0", "/data/image-2.zarr/0"]
    ])
    with pipeline.pop() as batch:
        array = np.from_dlpack(batch)
        print(array.shape)
        del array
```

The metadata provider loads descriptions on demand from the pushed URIs;
the chunk reader loads the encoded byte ranges that planning produces.
`np.from_dlpack` shares the output buffer. A live array keeps that buffer
occupied even after its `Batch` is released; use `.copy()` to retain data
while allowing the pool to reuse its storage.

See [Pipeline composition](docs/pipeline.md) for the complete component,
limit, ownership, and build contracts. Build a CPU Python package from source
with `libzstd` and `libblosc` installed (plus `liburing` on Linux). On macOS,
install dependencies with `brew install cmake ninja pkg-config zstd c-blosc`:

```sh
pip install . --config-settings=cmake.define.DAMACY_CUDA=OFF
```

## CUDA quick start

```python
import random
import damacy
import torch

cfg = damacy.Config(
    samples_per_batch=8,
    sample_shape=(64, 256, 256),
    max_gpu_memory_bytes=1 << 30,  # primary GPU budget
    dtype="bf16",                  # source dtype is cast on assemble
    device=0,
)

# A Sample names an absolute uri and a per-axis half-open AABB into
# the stored array (np.s_[...] also accepted). Build them however
# suits — your own sampler, a torch Dataset, a curriculum, a fixed
# tile grid, ...
volumes = {  # absolute uri → full ZYX shape
    "/data/cells/brain-001.zarr":  (512, 4096, 4096),
    "/data/cells/brain-002.zarr":  (768, 4096, 4096),
    "/data/cells/kidney-007.zarr": (256, 2048, 2048),
}
def random_crop(size=(64, 256, 256)):
    uri, full = random.choice(list(volumes.items()))
    origin = [random.randint(0, f - s) for f, s in zip(full, size)]
    return damacy.Sample(uri=uri, aabb=[(o, o + s) for o, s in zip(origin, size)])

samples = [random_crop() for _ in range(64)]

with damacy.Pipeline(cfg) as p:
    p.push(samples)                                # producer side
    for batch in p.batches(len(samples) // cfg.samples_per_batch):
        with batch as t:                           # consumer side
            x = torch.from_dlpack(t)               # zero-copy + stream-fenced
            ...                                    # train step
```

`torch.from_dlpack` (or any DLPack v1 consumer — cupy, jax, …) handles the
stream handoff: damacy hands over `BatchInfo.ready_stream`, the consumer
records a `cuStreamWaitEvent` against it, and the resulting tensor is
fenced for downstream kernels. Damacy's internal streams are non-blocking
with respect to the legacy default stream, so don't read
`BatchInfo.device_ptr` directly without a matching `cuStreamWaitEvent` on
`ready_stream`.

## Streaming

`push` accepts any iterable, including infinite generators — samples
are pulled lazily as `pop` frees space. For unbounded training, hand
the pipeline a generator and let it drain:

```python
def crops():
    while True:
        yield random_crop()  # from the example above

with damacy.Pipeline(cfg) as p:
    p.push(crops())                    # pulled on demand
    for step in range(N_STEPS):
        with p.pop() as t:
            x = torch.from_dlpack(t)
            ...                        # train step
```

For training loops that prefetch the next batch on a background
thread, see [Async prefetch](https://nclack.github.io/damacy/prefetch/)
— zero-copy with deferred release, plus the dedicated-copy-stream
variant.

## Zarr support

Damacy reads zarr v3 (sharded and non-sharded). What's recognized today:

| | supported | notes |
|---|---|---|
| Array versions | v3 | v2 stores are not read |
| Layout (sharded) | `sharding_indexed` | with `index_location` either `"start"` or `"end"` (default) |
| Layout (non-sharded) | yes | each chunk is a separate file at `c/<i>/<j>/...` |
| Inner / chunk codec | `bytes` (passthrough), `zstd`, `blosc` (cname=`zstd`) | `blosc` with `cname=lz4`/`lz4hc` is recognized at parse time and rejected by the executor |
| Sharding index codec | `bytes` + `crc32c` | the shard index itself; the data codec is separate |
| Missing chunks | yes — read as `fill_value` | per zarr v3 spec; sharded "empty" entries (`offset==nbytes==2^64−1`) and missing chunk files both route here |

Not yet handled — arrays declaring any of these will fail to parse:

- Non-trivial transposes (`transpose` codec)
- Compression codecs other than the list above (`gzip`, `lz4` raw, `crc32c` as a data codec, future v3 codecs)
- Complex / fixed-bytes / variable-length dtypes

If you have data that uses one of the unsupported codecs and you'd like it added, please open an issue with a sample `zarr.json`.

## Runtime dependencies

CPU builds support Linux and macOS and require the CPU codec libraries.
Linux uses io_uring for async metadata I/O; macOS uses a POSIX worker pool.
CUDA builds additionally link the NVIDIA driver and nvCOMP. Build with
`DAMACY_CUDA=OFF` to import and run on a host without a CUDA driver.

| Library | Used by | How it is loaded |
|---|---|---|
| `liburing` | Linux async metadata I/O | normal dynamic loader |
| `libzstd`, `libblosc` | CPU decoding, included in both builds | normal dynamic loader |
| `libcuda.so.1`, nvCOMP | CUDA builds | driver loader; nvCOMP may be linked statically |
| `libnuma.so.1` | Optional CUDA placement and host affinity | `dlopen`; absence disables placement |
| `libcufile.so.0` | Optional CUDA GPUDirect Storage | `dlopen`; requires `DAMACY_ENABLE_GDS=ON` |
| `libmount.so.1`, `libudev.so.1` | cuFile initialization when GDS is used | dynamic loader |

On Linux, metadata reads require a kernel with the io_uring operations damacy uses:
`STATX`, `OPENAT2`, `READ`, and `CLOSE`. If the kernel does not advertise
those operations, pipeline construction fails instead of falling back to a thread
pool. On macOS, `metadata_io_concurrency` sets the number of metadata workers;
bulk reads use the shared POSIX file backend. NUMA placement and CPU affinity
are unavailable. CUDA defaults off on macOS and cannot be enabled there.
See [native build instructions](docs/pipeline.md#build-without-cuda).

GDS notes:

- Build with `cmake -DDAMACY_ENABLE_GDS=ON` to link the cuFile backend. The default-OFF build still accepts `enable_gds = DAMACY_GDS_ON` but `damacy_create` returns `DAMACY_INVAL` (no silent fallback).
- `enable_gds = DAMACY_GDS_AUTO` (default, the value from designated-init) defers to env `DAMACY_GDS_ENABLE=1`; explicit `DAMACY_GDS_ON` / `DAMACY_GDS_OFF` override the env.
- On hosts without nvidia-fs, point `CUFILE_ENV_PATH_JSON` at a JSON with `{"properties":{"allow_compat_mode":true}}` to enable cuFile compat mode — reads go through cuFile's host-bounce buffer instead of DMA. Useful for correctness testing on consumer GPUs.
- If libcufile can't be loaded or `cuFileDriverOpen` fails, `damacy_create` returns `DAMACY_INVAL`.

## Documentation

Full API reference and guides: **<https://nclack.github.io/damacy/>**

- [Multi-GPU & distributed training](https://nclack.github.io/damacy/distributed/)
  — device binding model + torchrun / DDP examples.

Performance dashboards (auto-published from `bench.yml`):

- [Throughput](https://nclack.github.io/damacy/throughput/) — bigger is better
- [Timings](https://nclack.github.io/damacy/timings/) — smaller is better
