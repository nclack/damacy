# Damacy

[![build](https://github.com/nclack/damacy/actions/workflows/build.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/build.yml)
[![test](https://github.com/nclack/damacy/actions/workflows/test.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/nclack/damacy/branch/main/graph/badge.svg)](https://codecov.io/gh/nclack/damacy)
[![bench](https://github.com/nclack/damacy/actions/workflows/bench.yml/badge.svg)](https://nclack.github.io/damacy/throughput/)
[![docs](https://github.com/nclack/damacy/actions/workflows/docs.yml/badge.svg)](https://nclack.github.io/damacy/)

High-speed streamed assembly of tensors from zarr sources to GPU.

Damacy reads sharded [NGFF](https://ngff.openmicroscopy.org/) [zarr
v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) stores
directly on the GPU: per-shard chunk indexing, parallel host I/O, in-flight
GPU-side decompression (zstd, blosc1-zstd), and assembly of each batch
as a DLPack-ready device tensor.

## Quick start

```python
import random
import damacy
import torch

cfg = damacy.Config(
    batch_size=8,
    # Resource caps are fixed at construction; nothing grows after.
    max_gpu_memory_bytes=1 << 30,  # primary GPU budget
    dtype="bf16",                  # source dtype is cast on assemble
    # One pipeline binds to one GPU. Omit `device=` to capture the
    # current CUDA context (handy single-GPU; PyTorch sets one up
    # implicitly). For multi-GPU pass `device=local_rank` — see
    # https://nclack.github.io/damacy/distributed/
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
    for batch in p.batches(len(samples) // cfg.batch_size):
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

## Zarr support

Damacy reads zarr v3 (sharded and non-sharded). What's recognized today:

| | supported | notes |
|---|---|---|
| Array versions | v3 | v2 stores are not read |
| Layout (sharded) | `sharding_indexed` | with `index_location` either `"start"` or `"end"` (default) |
| Layout (non-sharded) | yes | each chunk is a separate file at `c/<i>/<j>/...` |
| Inner / chunk codec | `bytes` (passthrough), `zstd`, `blosc` (cname=`zstd`) | `blosc` with `cname=lz4`/`lz4hc` is recognized at parse time and rejected at planning |
| Sharding index codec | `bytes` + `crc32c` | the shard index itself; the data codec is separate |
| Missing chunks | yes — read as `fill_value` | per zarr v3 spec; sharded "empty" entries (`offset==nbytes==2^64−1`) and missing chunk files both route here |

Not yet handled — arrays declaring any of these will fail to parse:

- Non-trivial transposes (`transpose` codec)
- Compression codecs other than the list above (`gzip`, `lz4` raw, `crc32c` as a data codec, future v3 codecs)
- Complex / fixed-bytes / variable-length dtypes

If you have data that uses one of the unsupported codecs and you'd like it added, please open an issue with a sample `zarr.json`.

## Runtime dependencies

Damacy links only the essentials. Optional features dlopen their backends lazily, so a damacy binary loads on any host with a working CUDA driver — the feature simply turns off when its library isn't present.

| Library | Required at runtime | What you lose if missing | How damacy finds it |
|---|---|---|---|
| `libcuda.so.1` | always | nothing — damacy cannot run without it | NVIDIA driver install (`/run/opengl-driver/lib`, `/usr/lib/x86_64-linux-gnu`, …) |
| `libnuma.so.1` | optional | NUMA pinning of pinned-host slabs + io_queue / scheduler threads (single-socket hosts: no effect) | `dlopen` via dynamic loader (`LD_LIBRARY_PATH`, `ld.so.cache`) |
| `libcufile.so.0` | optional | `damacy_config.enable_gds = 1` — direct read of compressed chunks into device memory via NVIDIA GPUDirect Storage | `dlopen` via dynamic loader; ships with the CUDA toolkit and with nvidia-fs. Requires a build with `-DDAMACY_ENABLE_GDS=ON` (default OFF) |
| `libmount.so.1`, `libudev.so.1` | required *if and only if* using GDS | cuFile dlopen's these at driver init even in compat mode | dynamic loader |

GDS notes:

- Build with `cmake -DDAMACY_ENABLE_GDS=ON` to link the cuFile backend. The default-OFF build still accepts `enable_gds = 1` but `damacy_create` returns `DAMACY_INVAL` (no silent fallback).
- On hosts without nvidia-fs, point `CUFILE_ENV_PATH_JSON` at a JSON with `{"properties":{"allow_compat_mode":true}}` to enable cuFile compat mode — reads go through cuFile's host-bounce buffer instead of DMA. Useful for correctness testing on consumer GPUs.
- If libcufile can't be loaded or `cuFileDriverOpen` fails, `damacy_create` returns `DAMACY_INVAL`.

## Documentation

Full API reference and guides: **<https://nclack.github.io/damacy/>**

- [Multi-GPU & distributed training](https://nclack.github.io/damacy/distributed/)
  — device binding model + torchrun / DDP examples.

Performance dashboards (auto-published from `bench.yml`):

- [Throughput](https://nclack.github.io/damacy/throughput/) — bigger is better
- [Timings](https://nclack.github.io/damacy/timings/) — smaller is better
