# damacy

**Streamed assembly of tensors from Zarr sources into RAM or GPU memory.**

Damacy separates metadata and chunk planning from CPU or CUDA execution. Both
executors read Zarr v3 arrays, decode raw bytes, zstd, or Blosc-zstd, and return
contiguous batches through DLPack. CPU builds require no CUDA toolkit or driver.

Start with [Pipeline composition](pipeline.md) for the CPU API, complete
component construction, resource limits, and buffer ownership.

[![build](https://github.com/nclack/damacy/actions/workflows/build.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/build.yml)
[![test](https://github.com/nclack/damacy/actions/workflows/test.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/nclack/damacy/branch/main/graph/badge.svg)](https://codecov.io/gh/nclack/damacy)

---

## CUDA quick start

```python
import damacy
import torch

cfg = damacy.Config(
    samples_per_batch=2,
    sample_shape=(64, 256, 256),
    max_gpu_memory_bytes=1 << 30,
    dtype="bf16",
    device=0,
)
samples = [
    damacy.Sample(uri="/data/cells/cell-1.zarr", aabb=[(0, 64), (0, 256), (0, 256)]),
    damacy.Sample(uri="/data/cells/cell-2.zarr", aabb=[(0, 64), (0, 256), (0, 256)]),
]

with damacy.Pipeline(cfg) as d:
    d.push(samples)
    for _ in range(len(samples) // cfg.samples_per_batch):
        with d.pop() as batch:
            x = torch.from_dlpack(batch)
            ...  # train step
```

By default the pipeline captures whatever CUDA context is current on
the calling thread; callers can initialize it through PyTorch, and bare-Python
users can call `damacy._native.cuda_init_primary()` once. For
multi-GPU setups, see [Distributed](distributed.md) for the device
binding model and a torchrun example.

## Concepts

You hand damacy a stream of `Sample`s; it returns a stream of
`Batch`es, each one a CPU or CUDA tensor of shape
`(samples_per_batch, *sample_shape)`.

- A **`Sample`** is one crop request: a zarr URI plus an `aabb`
  (axis-aligned bounding box) given as a list of `(start, stop)`
  tuples — one per spatial axis. Every `aabb` must produce the same
  per-sample shape, and that shape is `BatchSpec.shape` (or `Config.sample_shape`).
- A **`Pipeline`** is a streaming context. You `push` an iterable
  of samples (lazy generators are fine — and recommended for long
  runs) and call `pop()` to block for the next ready batch.
- A **`Batch`** is a DLPack-ready handle to a CPU or CUDA tensor.
  Use it inside a `with` block and release consumer views so damacy can
  reclaim the buffer when you are done.

`samples_per_batch`, `sample_shape`, and `max_gpu_memory_bytes` are required
on `Config`; everything else has a sensible default. Assembly casts heterogeneous source dtypes
(`u8`/`u16`/`i16`/`u32`/`i32`/`f16`/`f32`) to the configured
destination `dtype` (`f32` or `bf16`) on the way out, so your zarrs
do not need to match it.

## Public surface

The published API lives entirely under the top-level `damacy` package.
The native extension (`damacy._native`) is an implementation detail
documented only via its `.pyi` stub.

- [Pipeline composition](pipeline.md) — CPU and CUDA components, builds, and lifetimes.
- [API reference](api.md) — `Pipeline`, `Config`, `Sample`, `Batch`, the
  exception hierarchy, and the `Stats`/`Metric` value types.
- [GPU memory budget](budget.md) — how to think about
  `max_gpu_memory_bytes`, what it covers, and how to pick a value.
- [Distributed](distributed.md) — device binding model and torchrun /
  DDP examples.
- [Async prefetch](prefetch.md) — zero-copy with deferred release
  for training loops that prefetch the next batch on a background
  thread.
- [Troubleshooting](troubleshooting.md) — common errors
  (`PoolStarved`, `BudgetExceeded`, missing CUDA context) and what
  to check first.

## Performance dashboards

Continuous benchmark history (auto-published from
[`bench.yml`](https://github.com/nclack/damacy/actions/workflows/bench.yml)):

<!-- Absolute URLs because the dashboards live on the same gh-pages
     branch but outside the mkdocs source tree (published by bench.yml).
     mkdocs treats unresolved relative links as warnings under --strict. -->
- [Throughput](https://nclack.github.io/damacy/throughput/) — bigger is better
- [Timings](https://nclack.github.io/damacy/timings/) — smaller is better
