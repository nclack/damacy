# damacy

**High-speed streamed assembly of tensors from zarr sources to GPU.**

damacy reads sharded [NGFF](https://ngff.openmicroscopy.org/) zarr stores
straight onto the GPU: per-shard chunk indexing, parallel host I/O,
in-flight GPU-side decompression (zstd, blosc1-zstd), and a typed
assemble kernel that lands each batch as a DLPack-ready device tensor.

[![build](https://github.com/nclack/damacy/actions/workflows/build.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/build.yml)
[![test](https://github.com/nclack/damacy/actions/workflows/test.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/nclack/damacy/branch/main/graph/badge.svg)](https://codecov.io/gh/nclack/damacy)

---

## Quick start

```python
import damacy
import torch

cfg = damacy.Config(
    batch_size=8,
    sample_shape=(64, 256, 256),
    max_gpu_memory_bytes=1 << 30,
    dtype="bf16",
)
samples = [
    damacy.Sample(uri="/data/cells/cell-1.zarr", aabb=[(0, 64), (0, 256), (0, 256)]),
    damacy.Sample(uri="/data/cells/cell-2.zarr", aabb=[(0, 64), (0, 256), (0, 256)]),
]

with damacy.Pipeline(cfg) as d:
    d.push(samples)
    for batch in d.batches(len(samples) // cfg.batch_size):
        with batch as t:
            x = torch.from_dlpack(t)
            ...  # train step
```

By default the pipeline captures whatever CUDA context is current on
the calling thread; PyTorch sets one up implicitly, and bare-Python
users can call `damacy._native.cuda_init_primary()` once. For
multi-GPU setups, see [Distributed](distributed.md) for the device
binding model and a torchrun example.

## Concepts

You hand damacy a stream of `Sample`s; it returns a stream of
`Batch`es, each one a device tensor of shape
`(batch_size, *sample_shape)`.

- A **`Sample`** is one crop request: a zarr URI plus an `aabb`
  (axis-aligned bounding box) given as a list of `(start, stop)`
  tuples — one per spatial axis. Every `aabb` must produce the same
  per-sample shape, and that shape is `Config.sample_shape`.
- A **`Pipeline`** is a streaming context. You `push` an iterable
  of samples (lazy generators are fine — and recommended for long
  runs) and call `pop()` (or iterate `batches(n)`) to receive
  completed batches.
- A **`Batch`** is a DLPack-ready handle to a GPU-resident tensor.
  Use it inside a `with` block so damacy can reclaim the slot when
  you're done.

`batch_size`, `sample_shape`, and `max_gpu_memory_bytes` are required
on `Config`; everything else has a sensible default. The assemble
kernel casts heterogeneous source dtypes
(`u8`/`u16`/`i16`/`u32`/`i32`/`f16`/`f32`) to the configured
destination `dtype` (`f32` or `bf16`) on the way out, so your zarrs
do not need to match it.

## Public surface

The published API lives entirely under the top-level `damacy` package.
The native extension (`damacy._native`) is an implementation detail
documented only via its `.pyi` stub.

- [API reference](api.md) — `Pipeline`, `Config`, `Sample`, `Batch`, the
  exception hierarchy, and the `Stats`/`Metric` value types.
- [GPU memory budget](budget.md) — how to think about
  `max_gpu_memory_bytes`, what it covers, and how to pick a value.
- [Distributed](distributed.md) — device binding model and torchrun /
  DDP examples.

## Performance dashboards

Continuous benchmark history (auto-published from
[`bench.yml`](https://github.com/nclack/damacy/actions/workflows/bench.yml)):

<!-- Absolute URLs because the dashboards live on the same gh-pages
     branch but outside the mkdocs source tree (published by bench.yml).
     mkdocs treats unresolved relative links as warnings under --strict. -->
- [Throughput](https://nclack.github.io/damacy/throughput/) — bigger is better
- [Timings](https://nclack.github.io/damacy/timings/) — smaller is better

