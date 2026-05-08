# damacy

**High-speed streamed assembly of tensors from zarr sources to GPU.**

damacy reads sharded [NGFF](https://ngff.openmicroscopy.org/) zarr stores
straight onto the GPU: per-shard chunk indexing, parallel host I/O,
in-flight GPU-side decompression (blosc1 / zstd), and a typed assemble
kernel that lands each batch as a DLPack-ready device tensor.

[![build](https://github.com/nclack/damacy/actions/workflows/build.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/build.yml)
[![test](https://github.com/nclack/damacy/actions/workflows/test.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/nclack/damacy/branch/main/graph/badge.svg)](https://codecov.io/gh/nclack/damacy)

---

## Quick start

```python
import damacy
import torch

cfg = damacy.Config(
    store_root="/data/cells",
    batch_size=8,
    host_buffer_bytes=1 << 30,
    device_buffer_bytes=1 << 30,
    dtype="bf16",
)
samples = [
    damacy.Sample(uri="cell-1.zarr", aabb=[(0, 64), (0, 256), (0, 256)]),
    damacy.Sample(uri="cell-2.zarr", aabb=[(0, 64), (0, 256), (0, 256)]),
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
multi-GPU setups, prefer `Config(..., device=local_rank)` — see
the [README](https://github.com/nclack/damacy#failure-modes).

## Public surface

The published API lives entirely under the top-level `damacy` package.
The native extension (`damacy._native`) is an implementation detail
documented only via its `.pyi` stub.

- [API reference](api.md) — `Pipeline`, `Config`, `Sample`, `Batch`, the
  exception hierarchy, and the `Stats`/`Metric` value types.

## Performance dashboards

Continuous benchmark history (auto-published from
[`bench.yml`](https://github.com/nclack/damacy/actions/workflows/bench.yml)):

<!-- Absolute URLs because the dashboards live on the same gh-pages
     branch but outside the mkdocs source tree (published by bench.yml).
     mkdocs treats unresolved relative links as warnings under --strict. -->
- [Throughput](https://nclack.github.io/damacy/throughput/) — bigger is better
- [Timings](https://nclack.github.io/damacy/timings/) — smaller is better

## Design notes

- A single in-flight **wave** is processed at a time; output batches are
  double-buffered so the consumer can overlap training compute with the
  next wave's decompression and assembly.
- Resource caps (host pinned-buffer pool, device decompress-scratch
  pool, GPU memory) are fixed at `Pipeline(...)` construction; nothing
  grows after that.
- The assemble kernel casts heterogeneous source dtypes
  (`u8`/`u16`/`i16`/`u32`/`i32`/`f16`/`f32`) to the configured
  destination dtype (`f32` or `bf16`) on the way out.
