# Damacy

[![build](https://github.com/nclack/damacy/actions/workflows/build.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/build.yml)
[![test](https://github.com/nclack/damacy/actions/workflows/test.yml/badge.svg)](https://github.com/nclack/damacy/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/nclack/damacy/branch/main/graph/badge.svg)](https://codecov.io/gh/nclack/damacy)
[![bench](https://github.com/nclack/damacy/actions/workflows/bench.yml/badge.svg)](https://nclack.github.io/damacy/throughput/)
[![docs](https://github.com/nclack/damacy/actions/workflows/docs.yml/badge.svg)](https://nclack.github.io/damacy/)

High-speed streamed assembly of tensors from zarr sources to GPU.

Damacy reads sharded [NGFF](https://ngff.openmicroscopy.org/) [zarr
v3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) stores
straight onto the GPU: per-shard chunk indexing, parallel host I/O, in-flight
GPU-side decompression (blosc1 / zstd), and assembly of each batch as a
DLPack-ready device tensor.

## Quick start

```python
import damacy
import torch

cfg = damacy.Config(
    store_root="/data/cells",
    batch_size=8,
    host_buffer_bytes=1 << 30,
    device_buffer_bytes=1 << 30,
    dtype="bf16", # source data will be cast to bf16
)
samples = [
    damacy.Sample(uri=f"cell-{i}.zarr", aabb=[(0, 64), (0, 256), (0, 256)])
    for i in range(64)
]

with damacy.Damacy(cfg) as d:
    d.push(samples)
    for batch in d.batches(len(samples) // cfg.batch_size):
        with batch as t:
            x = torch.from_dlpack(t)  # zero-copy view of the device tensor
            ...                       # train step
```

A current CUDA context is required on the calling thread. PyTorch sets
one up implicitly on first allocation; for bare-Python use call
`damacy._native.cuda_init_primary()` once first. With no current
context, `Damacy(cfg)` raises `damacy.NativeCudaError`.

`aabb` accepts ``(start, stop)`` 2-tuples or Python ``slice`` objects
per axis, so ``aabb=np.s_[0:64, 0:256, 0:256]`` works directly.

## Distributed (torchrun, 8 GPUs)

Each rank owns its own GPU, its own `damacy` pipeline, and its own slice
of the sample list. Resource caps are per-rank, so `host_buffer_bytes`
and `device_buffer_bytes` are scaled to a single GPU's budget.

```python
# train.py — launch with:
#   torchrun --standalone --nproc_per_node=8 train.py
import os
import torch
import torch.distributed as dist
import damacy


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    # Bind this rank to its GPU and prime the CUDA context. Damacy
    # captures the caller's context at construction; do this *before*
    # `damacy.Damacy(...)`. See "Failure modes" below for what goes
    # wrong if you skip either of these two lines.
    torch.cuda.set_device(local_rank)
    torch.empty(1, device="cuda")  # touches the primary context
    assert torch.cuda.current_device() == local_rank

    cfg = damacy.Config(
        store_root="/data/cells",
        batch_size=8,
        host_buffer_bytes=1 << 30,    # per-rank
        device_buffer_bytes=1 << 30,  # per-rank
        dtype="bf16",
        n_io_threads=4,
    )

    # Shard work by rank — strided so adjacent samples stay co-located
    # within their wave on each rank.
    all_samples = [
        damacy.Sample(uri=f"cell-{i}.zarr", aabb=[(0, 64), (0, 256), (0, 256)])
        for i in range(8192)
    ]
    my_samples = all_samples[local_rank::world_size]

    with damacy.Damacy(cfg) as d:
        d.push(my_samples)
        for batch in d.batches(len(my_samples) // cfg.batch_size):
            with batch as t:
                x = torch.from_dlpack(t)
                # forward / backward / optimizer step (DDP-wrapped model)
                ...

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

Notes:

- The pipeline is single-rank-aware. There is no cross-rank coordination
  inside Damacy; throughput scales linearly with rank count provided
  the storage layer keeps up.
- `n_io_threads` is per-rank; tune to your storage tier (NVMe pool,
  parallel filesystem, or object store).

### Failure modes

`Damacy(cfg)` snapshots whatever `CUcontext` is current on the calling
thread, **for the lifetime of the loader**. The two pre-`Damacy(cfg)`
lines above each guard a distinct foot-gun:

| if you skip… | what happens |
|---|---|
| `torch.cuda.set_device(local_rank)` | All 8 ranks bind to the default device (GPU 0). DDP gradient sync still works, so training **looks fine** — it's just silently single-GPU-bound at ~1/8 throughput. The `assert torch.cuda.current_device() == local_rank` line catches this fail-fast. |
| `torch.empty(1, device="cuda")` (or any other CUDA op) after `set_device` | The primary context for the new device hasn't been created yet, so no `CUcontext` is current when `Damacy(cfg)` runs. Construction raises `damacy.NativeCudaError` (status `CUDA`). |
| Both | Same as the second row — fail-fast with `NativeCudaError`. |

The "silent → wrong-device" mode is the dangerous one; the assertion
is there to convert it into a loud `AssertionError` before any data
moves.

## Documentation

Full API reference and design notes: **<https://nclack.github.io/damacy/>**

Performance dashboards (auto-published from `bench.yml`):

- [Throughput](https://nclack.github.io/damacy/throughput/) — bigger is better
- [Timings](https://nclack.github.io/damacy/timings/) — smaller is better
