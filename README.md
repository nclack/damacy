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
GPU-side decompression (blosc1 / zstd), and assembly of each batch as a
DLPack-ready device tensor.

## Quick start

```python
import random
import damacy
import torch

cfg = damacy.Config(
    batch_size=8,
    # Resource caps are fixed at construction; nothing grows after.
    host_buffer_bytes=1 << 30,    # pinned host staging pool
    device_buffer_bytes=1 << 30,  # device decompress scratch
    dtype="bf16",                 # source dtype is cast on assemble
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

## Documentation

Full API reference and guides: **<https://nclack.github.io/damacy/>**

- [Multi-GPU & distributed training](https://nclack.github.io/damacy/distributed/)
  — device binding model + torchrun / DDP examples.

Performance dashboards (auto-published from `bench.yml`):

- [Throughput](https://nclack.github.io/damacy/throughput/) — bigger is better
- [Timings](https://nclack.github.io/damacy/timings/) — smaller is better
