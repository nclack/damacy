"""High-throughput streaming loader for sharded NGFF zarr stores.

The native extension is loaded eagerly at import so the in-process log
sink is registered before any C threads spin up — log records produced
by C threads are routed to ``logging.getLogger("damacy")`` via a
daemon-thread drain that holds the GIL only while delivering messages.

Typical use::

    import damacy, torch

    cfg = damacy.Config(
        batch_size=8,
        max_gpu_memory_bytes=1 << 30,
        dtype="bf16",
    )
    samples = [
        damacy.Sample(uri="/data/cells/cell-1.zarr", aabb=[(0, 64), (0, 256), (0, 256)]),
        damacy.Sample(uri="/data/cells/cell-2.zarr", aabb=[(0, 64), (0, 256), (0, 256)]),
    ]

    with damacy.Pipeline(cfg) as p:
        p.push(samples)
        for batch in p.batches(len(samples) // cfg.batch_size):
            with batch as t:
                x = torch.from_dlpack(t)
                ...  # train step

Variants reuse a base via :func:`dataclasses.replace`::

    big = dataclasses.replace(cfg, batch_size=64)
"""

from __future__ import annotations

import itertools
import logging
import os
import threading
import warnings
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import IntEnum
from types import TracebackType
from typing import TYPE_CHECKING, Any, NoReturn

from damacy import _native

if TYPE_CHECKING:  # avoid runtime import; only used for type hints
    from typing import Self

__all__ = [
    "Batch",
    "BatchInfo",
    "BudgetExceeded",
    "Config",
    "DamacyError",
    "DecodeError",
    "Dtype",
    "DtypeMismatch",
    "InvalidArgument",
    "Metric",
    "NativeCudaError",
    "NotFound",
    "NumaStrategy",
    "OutOfMemory",
    "Pipeline",
    "PoolStarved",
    "RankMismatch",
    "Sample",
    "ShutdownError",
    "Stats",
    "Status",
    "StorageError",
    "TryAgain",
    "set_log_level",
    "set_log_quiet",
]

__version__: str = _native.__version__


# ---- enums --------------------------------------------------------------


class Dtype(IntEnum):
    """Destination dtype for assembled batches. Sources may differ; the
    assemble kernel casts each element to this type."""

    F32 = _native.DTYPE_F32
    BF16 = _native.DTYPE_BF16

    @classmethod
    def coerce(cls, value: str | int | Dtype) -> Dtype:
        """Accept enum / int / one of {"f32", "float32", "bf16", "bfloat16"}.

        ```pycon
        >>> Dtype.coerce("f32") is Dtype.F32
        True
        >>> Dtype.coerce("BFloat16") is Dtype.BF16
        True
        >>> Dtype.coerce(Dtype.F32) is Dtype.F32
        True
        >>> Dtype.coerce("nope")
        Traceback (most recent call last):
            ...
        ValueError: unknown dtype: 'nope'

        ```
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        s = str(value).lower()
        if s in ("f32", "float32"):
            return cls.F32
        if s in ("bf16", "bfloat16"):
            return cls.BF16
        raise ValueError(f"unknown dtype: {value!r}")


class NumaStrategy(IntEnum):
    """How to pin pinned-host slabs and worker threads to a NUMA node.

    ``AUTO`` resolves the GPU's host-NUMA node from the CUDA driver
    (sysfs fallback) at create time. ``DISABLED`` is a full no-op.
    ``PIN_TO`` uses the explicit :attr:`Config.numa_node`.

    All three are silent no-ops when ``libnuma.so.1`` cannot be
    loaded at runtime; ``AUTO`` is also a no-op on single-node hosts
    where pinning has nothing to bind against.

    ```pycon
    >>> NumaStrategy.coerce("auto") is NumaStrategy.AUTO
    True
    >>> NumaStrategy.coerce("pin_to") is NumaStrategy.PIN_TO
    True
    >>> NumaStrategy.coerce(NumaStrategy.DISABLED) is NumaStrategy.DISABLED
    True
    >>> NumaStrategy.coerce("nope")
    Traceback (most recent call last):
        ...
    ValueError: unknown numa strategy: 'nope'

    ```
    """

    AUTO = _native.NUMA_AUTO
    DISABLED = _native.NUMA_DISABLED
    PIN_TO = _native.NUMA_PIN_TO

    @classmethod
    def coerce(cls, value: str | int | NumaStrategy) -> NumaStrategy:
        """Accept enum / int / one of ``"auto"`` / ``"disabled"`` / ``"pin_to"``."""
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        s = str(value).lower().replace("-", "_")
        if s == "auto":
            return cls.AUTO
        if s == "disabled":
            return cls.DISABLED
        if s == "pin_to":
            return cls.PIN_TO
        raise ValueError(f"unknown numa strategy: {value!r}")


def _gds_to_native(v: bool | None) -> int:
    if v is None:
        return _native.GDS_AUTO
    return _native.GDS_ON if v else _native.GDS_OFF


class Status(IntEnum):
    """Mirrors ``enum damacy_status``."""

    OK = _native.STATUS_OK
    AGAIN = _native.STATUS_AGAIN
    INVAL = _native.STATUS_INVAL
    NOTFOUND = _native.STATUS_NOTFOUND
    DTYPE = _native.STATUS_DTYPE
    RANK = _native.STATUS_RANK
    IO = _native.STATUS_IO
    DECODE = _native.STATUS_DECODE
    CUDA = _native.STATUS_CUDA
    OOM = _native.STATUS_OOM
    BUDGET = _native.STATUS_BUDGET
    SHUTDOWN = _native.STATUS_SHUTDOWN


# ---- exceptions ---------------------------------------------------------


class DamacyError(_native.DamacyError):
    """Base class for all damacy errors. ``.status`` is a :class:`Status`;
    ``.what`` names the failing stage (e.g. ``"create"``, ``"pop"``)."""

    status: Status  # type: ignore[assignment]


class TryAgain(DamacyError):
    """Non-blocking call would block (lookahead queue full)."""


class InvalidArgument(DamacyError):
    """Bad arguments / configuration rejected at create or push."""


class NotFound(DamacyError):
    """Sample uri did not resolve to a zarr."""


class DtypeMismatch(DamacyError):
    """Source dtype has no cast path to the configured destination dtype."""


class RankMismatch(DamacyError):
    """Sample rank is incompatible with the resolved zarr rank."""


class StorageError(DamacyError):
    """Read or open failure on a shard file. Named ``StorageError`` rather
    than ``IOError`` to avoid shadowing :class:`builtins.IOError` for
    callers that ``from damacy import *``."""


class DecodeError(DamacyError):
    """Codec parse or decompression failure."""


class NativeCudaError(DamacyError):
    """CUDA driver/runtime call failed."""


class OutOfMemory(DamacyError):
    """Host allocation failed (malloc/calloc returned NULL). Distinct
    from :class:`BudgetExceeded` — the OS denied memory, the
    configured cap was not the limiting factor."""


class BudgetExceeded(DamacyError):
    """A configured cap is too small to satisfy the request. Most
    commonly: ``Config.max_gpu_memory_bytes`` cannot fit the
    requested batch geometry, or a chunk's uncompressed size exceeds
    ``Config.max_chunk_uncompressed_bytes``. Raise the relevant cap
    and retry."""


class ShutdownError(DamacyError):
    """Pipeline destroyed or in failed state."""


class PoolStarved(DamacyError):
    """Raised when :meth:`Pipeline.pop` waits longer than
    ``Config.pop_timeout_s`` for the next batch.

    The usual cause is your loop holding on to tensors from previous
    batches — for example by stashing them in a list — which keeps
    damacy from reusing that memory. Drop those references before
    the next ``pop()``, or call ``.clone()`` if you need to keep
    them."""

    status = Status.SHUTDOWN


# Status -> exception class. Picked up by _wrap_native_error; OK is absent
# because no error is raised on success.
_STATUS_TO_EXC: dict[int, type[DamacyError]] = {
    _native.STATUS_AGAIN: TryAgain,
    _native.STATUS_INVAL: InvalidArgument,
    _native.STATUS_NOTFOUND: NotFound,
    _native.STATUS_DTYPE: DtypeMismatch,
    _native.STATUS_RANK: RankMismatch,
    _native.STATUS_IO: StorageError,
    _native.STATUS_DECODE: DecodeError,
    _native.STATUS_CUDA: NativeCudaError,
    _native.STATUS_OOM: OutOfMemory,
    _native.STATUS_BUDGET: BudgetExceeded,
    _native.STATUS_SHUTDOWN: ShutdownError,
}


def _reraise_typed(exc: _native.DamacyError) -> NoReturn:
    """Re-raise a native DamacyError as the matching :class:`DamacyError`
    subclass. Always raises."""
    status = getattr(exc, "status", None)
    cls = _STATUS_TO_EXC.get(int(status) if status is not None else -1, DamacyError)
    typed = cls(*exc.args)
    typed.status = Status(status) if status is not None else Status.SHUTDOWN
    typed.what = getattr(exc, "what", "")
    raise typed from exc


# ---- value types --------------------------------------------------------


def _normalise_axis(x: object, axis: int) -> tuple[int, int]:
    """Coerce one AABB axis spec to a canonical ``(start, stop)`` tuple.
    Slices accept ``start=None`` (defaults to 0) and require an integer
    ``stop``; ``step`` must be 1 or omitted. Bare ints are *rejected*
    — in NumPy/zarr indexing an int means "point selection", not
    "extent from 0", and silently disagreeing with that convention
    would be a foot-gun for ``np.s_[...]`` users.

    The input is typed as ``object`` so the final ``raise TypeError``
    is a real runtime guard, not unreachable code (callers pass
    ``slice | tuple[int, int]`` per axis but the protocol can't stop a
    rogue value)."""
    if isinstance(x, slice):
        if x.step not in (None, 1):
            raise ValueError(
                f"aabb axis {axis}: slice step must be 1 or omitted (got step={x.step})"
            )
        if x.stop is None:
            raise ValueError(f"aabb axis {axis}: slice stop is required (got {x!r})")
        return (0 if x.start is None else int(x.start), int(x.stop))
    if isinstance(x, tuple) and len(x) == 2:
        return (int(x[0]), int(x[1]))
    raise TypeError(
        f"aabb axis {axis}: expected slice or (start, stop) tuple; "
        f"got {type(x).__name__}"
    )


@dataclass(init=False, frozen=True, slots=True)
class Sample:
    """One sample request. ``aabb`` is a per-axis half-open interval
    list in level-0 voxel indices, in the zarr's stored axis order.

    Each axis may be a ``(start, stop)`` 2-tuple or a Python ``slice``;
    the tuple of slices that ``numpy.s_[...]`` produces is accepted
    directly. The stored form is always
    ``tuple[tuple[int, int], ...]`` regardless of how it was spelled,
    so equivalent inputs hash and compare equal:

    ```pycon
    >>> a = Sample(uri="cell.zarr", aabb=[(0, 64), (0, 256), (0, 256)])
    >>> b = Sample(
    ...     uri="cell.zarr",
    ...     aabb=[slice(0, 64), slice(0, 256), slice(0, 256)],
    ... )
    >>> c = Sample(
    ...     uri="cell.zarr",
    ...     aabb=[slice(None, 64), slice(None, 256), slice(None, 256)],
    ... )
    >>> a == b == c
    True
    >>> hash(a) == hash(b) == hash(c)
    True
    >>> a.aabb
    ((0, 64), (0, 256), (0, 256))

    ```

    Bare ints in ``aabb`` are rejected so the behaviour stays
    consistent with NumPy/zarr indexing semantics
    (``np.s_[64]`` means "point 64", not "extent (0, 64)"):

    ```pycon
    >>> Sample(uri="cell.zarr", aabb=[64, 256, 256])
    Traceback (most recent call last):
        ...
    TypeError: aabb axis 0: expected slice or (start, stop) tuple; got int

    ```

    Slice validation rejects strided slices and unbounded stops:

    ```pycon
    >>> Sample(uri="cell.zarr", aabb=[slice(0, 64, 2), slice(0, 256), slice(0, 256)])
    Traceback (most recent call last):
        ...
    ValueError: aabb axis 0: slice step must be 1 or omitted (got step=2)
    >>> Sample(uri="cell.zarr", aabb=[slice(0, None), slice(0, 256), slice(0, 256)])
    Traceback (most recent call last):
        ...
    ValueError: aabb axis 0: slice stop is required (got slice(0, None, None))

    ```
    """

    uri: str
    aabb: tuple[tuple[int, int], ...]

    def __init__(
        self,
        uri: str,
        aabb: Iterable[slice | tuple[int, int]],
    ) -> None:
        # Custom __init__ rather than __post_init__ so pyright sees the
        # polymorphic input shape on construction *and* the precise
        # canonical shape on field reads. dataclass(init=False) keeps
        # the auto-generated __eq__ / __hash__ / __repr__.
        object.__setattr__(self, "uri", uri)
        object.__setattr__(
            self,
            "aabb",
            tuple(_normalise_axis(x, axis=i) for i, x in enumerate(aabb)),
        )

    def _to_native(self) -> dict[str, Any]:
        return {"uri": self.uri, "aabb": list(self.aabb)}


@dataclass(init=False, frozen=True, slots=True)
class Config:
    """All resource caps and pipeline shape, fixed at create time.

    Build variants with :func:`dataclasses.replace`:

    ```pycon
    >>> import dataclasses
    >>> base = Config(batch_size=8, sample_shape=(8, 16),
    ...               max_gpu_memory_bytes=1 << 30)
    >>> base.dtype is Dtype.F32
    True
    >>> dataclasses.replace(base, batch_size=64).batch_size
    64

    ```

    Validation runs in ``__init__`` so invalid configs fail before we
    touch CUDA. The constructor accepts :class:`Dtype`, an int, or one
    of ``"f32"`` / ``"float32"`` / ``"bf16"`` / ``"bfloat16"`` for the
    ``dtype`` argument; the stored field is always a :class:`Dtype`.

    ```pycon
    >>> Config(batch_size=0, sample_shape=(8, 16), max_gpu_memory_bytes=1 << 30)
    Traceback (most recent call last):
        ...
    ValueError: batch_size must be >= 1 (got 0)

    ```

    Attributes:
        batch_size: Samples per batch (>= 1).
        max_gpu_memory_bytes: Primary GPU budget knob. Hard cap on
            GPU memory allocated for wave-resident buffers, decoder
            scratch, per-wave fanout SOAs, and batch-output pools.
            Required — no default. A value too small for the
            requested batch geometry raises :class:`BudgetExceeded`
            from ``Pipeline(cfg)``. Internal sizing (host slab,
            dev decompress arena, nvcomp temp) is derived from this
            value; the create-time resolver also reserves the
            worst-case observe-and-grow footprint so grows inside a
            successfully-created instance never trip the cap.
        dtype: Destination dtype for assembled batches.
        lookahead_batches: User-side push-queue depth (>= 2).
        n_io_threads: IO worker threads (>= 1).
        n_array_meta_cache: LRU cap for zarr-metadata entries.
        n_shard_index_cache: LRU cap for shard-index entries.
        n_chunk_layout_cache: LRU cap for per-array blosc1 chunk-layout entries.
        max_chunk_uncompressed_bytes: Largest uncompressed chunk size
            the pipeline accepts; 0 selects the C default (512 KB).
        max_read_op_bytes: Cap on the size of a single coalesced
            read issued to storage. 0 selects the C default. Tune
            against your storage tier: small values keep the queue
            deep and the read pattern fine-grained; large values
            amortize per-syscall overhead at the cost of latency
            spikes.
        device: CUDA device index to bind. ``None`` (default) captures
            the current ``CUcontext`` on the calling thread; pass an
            int (e.g. ``local_rank``) to retain that device's primary
            context internally — recommended under torchrun / MPI.
        pop_timeout_s: How long :meth:`Pipeline.pop` waits for the
            next batch before raising :class:`PoolStarved`. Defaults
            to 30 seconds; pass ``None`` to wait forever.
        enable_gds: GPUDirect Storage opt-in. ``True`` forces cuFile
            reads of compressed bytes straight to device memory,
            bypassing host-staging slabs; ``False`` forces the
            host-staging path. ``None`` (default) defers to env
            ``DAMACY_GDS_ENABLE=1``. Explicit ``True``/``False`` wins
            over the env var. Requires ``libcufile.so.0`` and a
            successful ``cuFileDriverOpen`` at create time.
        numa_strategy: How to pin pinned-host slabs and worker
            threads to a host-NUMA node. :attr:`NumaStrategy.AUTO`
            (default) resolves the GPU's host-NUMA node from the
            CUDA driver. :attr:`NumaStrategy.PIN_TO` uses
            :attr:`numa_node`. :attr:`NumaStrategy.DISABLED` is a
            full no-op. All three are silent no-ops when
            ``libnuma.so.1`` is unavailable at runtime; ``AUTO`` is
            also a no-op on single-node hosts.
        numa_node: Explicit host-NUMA node when
            ``numa_strategy=NumaStrategy.PIN_TO``. Must be ``>= 0``
            in that mode; must be ``-1`` (default) otherwise — the
            constructor rejects a node hint paired with a non-PIN_TO
            strategy rather than silently dropping it.
    """

    batch_size: int
    dtype: Dtype
    lookahead_batches: int
    max_chunk_uncompressed_bytes: int
    max_read_op_bytes: int
    max_gpu_memory_bytes: int
    host_buffer_waves: int
    max_chunks_per_wave: int
    max_substreams_per_chunk: int
    n_io_threads: int
    n_array_meta_cache: int
    n_shard_index_cache: int
    n_chunk_layout_cache: int
    sample_shape: tuple[int, ...]
    device: int | None
    pop_timeout_s: float | None
    enable_gds: bool | None
    numa_strategy: NumaStrategy
    numa_node: int

    def __init__(
        self,
        *,
        batch_size: int,
        sample_shape: Sequence[int],
        max_gpu_memory_bytes: int,
        dtype: Dtype | str | int = Dtype.F32,
        lookahead_batches: int = 2,
        max_chunk_uncompressed_bytes: int = 0,
        max_read_op_bytes: int = 0,
        host_buffer_waves: int = 0,
        max_chunks_per_wave: int = 0,
        max_substreams_per_chunk: int = 0,
        n_io_threads: int = 4,
        n_array_meta_cache: int = 64,
        n_shard_index_cache: int = 256,
        n_chunk_layout_cache: int = 64,
        device: int | None = None,
        pop_timeout_s: float | None = 30.0,
        enable_gds: bool | None = None,
        numa_strategy: NumaStrategy | str | int = NumaStrategy.AUTO,
        numa_node: int = -1,
    ) -> None:
        # Custom __init__ rather than __post_init__ so the constructor
        # signature accepts the polymorphic dtype input while reads of
        # `cfg.dtype` always type as Dtype. dataclass(init=False) keeps
        # the auto-generated __eq__ / __hash__ / __repr__.
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 (got {batch_size})")
        if lookahead_batches < 2:
            raise ValueError(
                f"lookahead_batches must be >= 2 (got {lookahead_batches})"
            )
        if n_io_threads < 1:
            raise ValueError(f"n_io_threads must be >= 1 (got {n_io_threads})")
        if max_chunk_uncompressed_bytes < 0:
            raise ValueError("max_chunk_uncompressed_bytes must be >= 0")
        if max_read_op_bytes < 0:
            raise ValueError("max_read_op_bytes must be >= 0")
        if max_gpu_memory_bytes < 1:
            raise ValueError(
                f"max_gpu_memory_bytes must be >= 1 (got {max_gpu_memory_bytes})"
            )
        if host_buffer_waves < 0:
            raise ValueError("host_buffer_waves must be >= 0")
        if max_chunks_per_wave < 0:
            raise ValueError("max_chunks_per_wave must be >= 0")
        if max_chunks_per_wave > 0xFFFF:
            raise ValueError(
                f"max_chunks_per_wave must be <= 0xFFFF (got {max_chunks_per_wave})"
            )
        if max_substreams_per_chunk < 0:
            raise ValueError("max_substreams_per_chunk must be >= 0")
        if max_substreams_per_chunk > 0xFFFF:
            raise ValueError(
                f"max_substreams_per_chunk must be <= 0xFFFF (got {max_substreams_per_chunk})"
            )
        if pop_timeout_s is not None and pop_timeout_s <= 0:
            raise ValueError(f"pop_timeout_s must be > 0 or None (got {pop_timeout_s})")
        ns = NumaStrategy.coerce(numa_strategy)
        if ns is NumaStrategy.PIN_TO:
            if numa_node < 0:
                raise ValueError(
                    f"numa_node must be >= 0 when numa_strategy=PIN_TO (got {numa_node})"
                )
        elif numa_node != -1:
            raise ValueError(
                f"numa_node must be -1 when numa_strategy={ns.name} (got {numa_node})"
            )
        shape_t = tuple(int(x) for x in sample_shape)
        if not shape_t:
            raise ValueError("sample_shape must be non-empty")
        if any(d <= 0 for d in shape_t):
            raise ValueError(f"sample_shape entries must be > 0 (got {shape_t})")
        set_ = object.__setattr__  # frozen=True forbids `self.x = ...`
        set_(self, "batch_size", batch_size)
        set_(self, "dtype", Dtype.coerce(dtype))
        set_(self, "lookahead_batches", lookahead_batches)
        set_(self, "max_chunk_uncompressed_bytes", max_chunk_uncompressed_bytes)
        set_(self, "max_read_op_bytes", max_read_op_bytes)
        set_(self, "max_gpu_memory_bytes", max_gpu_memory_bytes)
        set_(self, "host_buffer_waves", host_buffer_waves)
        set_(self, "max_chunks_per_wave", max_chunks_per_wave)
        set_(self, "max_substreams_per_chunk", max_substreams_per_chunk)
        set_(self, "n_io_threads", n_io_threads)
        set_(self, "n_array_meta_cache", n_array_meta_cache)
        set_(self, "n_shard_index_cache", n_shard_index_cache)
        set_(self, "n_chunk_layout_cache", n_chunk_layout_cache)
        set_(self, "sample_shape", shape_t)
        set_(self, "device", device)
        set_(self, "pop_timeout_s", pop_timeout_s)
        set_(self, "enable_gds", None if enable_gds is None else bool(enable_gds))
        set_(self, "numa_strategy", ns)
        set_(self, "numa_node", int(numa_node))


@dataclass(frozen=True, slots=True)
class BatchInfo:
    """Snapshot of the on-device batch geometry."""

    device_ptr: int
    shape: tuple[int, ...]
    dtype: Dtype
    ready_stream: int
    batch_id: int

    @classmethod
    def _from_native(cls, info: dict[str, Any]) -> BatchInfo:
        return cls(
            device_ptr=info["device_ptr"],
            shape=tuple(info["shape"]),
            dtype=Dtype.coerce(info["dtype"]),
            ready_stream=info["ready_stream"],
            batch_id=info["batch_id"],
        )


@dataclass(frozen=True, slots=True)
class Metric:
    """One pipeline-stage metric. ``ms`` is cumulative; ``best_ms`` is the
    best single observation (large sentinel when no samples yet)."""

    name: str
    ms: float
    best_ms: float
    input_bytes: float
    output_bytes: float
    count: int

    @classmethod
    def _from_native(cls, m: dict[str, Any]) -> Metric:
        return cls(
            name=m["name"],
            ms=m["ms"],
            best_ms=m["best_ms"],
            input_bytes=m["input_bytes"],
            output_bytes=m["output_bytes"],
            count=m["count"],
        )


@dataclass(frozen=True, slots=True)
class Stats:
    """Cumulative pipeline metrics. Reset with :meth:`Pipeline.stats_reset`."""

    plan: Metric
    io: Metric
    h2d: Metric
    decode: Metric
    post_decode: Metric
    decode_gap: Metric
    assemble: Metric
    bind_wait: Metric
    pop_wait: Metric
    flush_wait: Metric
    array_meta_hits: int
    array_meta_misses: int
    shard_index_hits: int
    shard_index_misses: int
    chunk_layout_hits: int
    chunk_layout_misses: int
    batches_emitted: int
    batches_truncated: int
    waves_emitted: int
    chunks_planned: int
    chunks_to_load: int
    chunks_dispatched: int
    reads_issued: int
    worker_steps: int
    gpu_bytes_committed: int

    @classmethod
    def _from_native(cls, st: dict[str, Any]) -> Stats:
        m = Metric._from_native
        return cls(
            plan=m(st["plan"]),
            io=m(st["io"]),
            h2d=m(st["h2d"]),
            decode=m(st["decode"]),
            post_decode=m(st["post_decode"]),
            decode_gap=m(st["decode_gap"]),
            assemble=m(st["assemble"]),
            bind_wait=m(st["bind_wait"]),
            pop_wait=m(st["pop_wait"]),
            flush_wait=m(st["flush_wait"]),
            array_meta_hits=st["array_meta_hits"],
            array_meta_misses=st["array_meta_misses"],
            shard_index_hits=st["shard_index_hits"],
            shard_index_misses=st["shard_index_misses"],
            chunk_layout_hits=st["chunk_layout_hits"],
            chunk_layout_misses=st["chunk_layout_misses"],
            batches_emitted=st["batches_emitted"],
            batches_truncated=st["batches_truncated"],
            waves_emitted=st["waves_emitted"],
            chunks_planned=st["chunks_planned"],
            chunks_to_load=st["chunks_to_load"],
            chunks_dispatched=st["chunks_dispatched"],
            reads_issued=st["reads_issued"],
            worker_steps=st["worker_steps"],
            gpu_bytes_committed=st["gpu_bytes_committed"],
        )


# ---- log helpers --------------------------------------------------------


def set_log_level(level: int) -> None:
    """Set the threshold for the C-side stderr sink (TRACE=0..FATAL=5).
    The Python sink (``logging.getLogger("damacy")``) is independent."""
    _native.set_log_level(level)


def set_log_quiet(quiet: bool) -> None:
    """Toggle the C-side stderr sink. The Python ``logging`` sink keeps
    firing regardless."""
    _native.set_log_quiet(quiet)


# Log-level constants re-exported for convenience.
LOG_TRACE: int = _native.LOG_TRACE
LOG_DEBUG: int = _native.LOG_DEBUG
LOG_INFO: int = _native.LOG_INFO
LOG_WARN: int = _native.LOG_WARN
LOG_ERROR: int = _native.LOG_ERROR
LOG_FATAL: int = _native.LOG_FATAL


# ---- Batch --------------------------------------------------------------


def _coerce_cuda_event_handle(event: object) -> int | None:
    """Resolve a user-supplied stream/event hint to a raw CUevent handle.

    Accepts:
      * ``None`` — caller wants the immediate-release path.
      * An ``int`` — already a CUevent handle.
      * ``torch.cuda.Event`` — read its ``.cuda_event`` attribute.
      * ``torch.cuda.Stream`` — call ``.record_event()`` and re-coerce.
      * ``cupy.cuda.Event`` — read its ``.ptr`` attribute (an int).
      * ``cupy.cuda.Stream`` — call ``.record()`` and re-coerce.

    Returns the integer handle, or ``None`` if ``event is None``.

    Raises ``TypeError`` for anything else. JAX does not expose CUevent
    handles through its public API; jax users need to drop to ``cuda``
    via dlpack and provide their own event.

    Stream paths record on the user's stream at call time, so the
    resulting event captures the stream's then-current position.
    """
    if event is None:
        return None
    if isinstance(event, int):
        return event
    # torch.cuda.Stream
    if hasattr(event, "record_event"):
        rec = event.record_event()  # type: ignore[union-attr]
        return _coerce_cuda_event_handle(rec)
    # cupy.cuda.Stream — .record() returns an Event
    if hasattr(event, "record") and callable(event.record):  # type: ignore[union-attr]
        rec = event.record()  # type: ignore[union-attr]
        return _coerce_cuda_event_handle(rec)
    # torch.cuda.Event
    handle = getattr(event, "cuda_event", None)
    if handle is not None:
        return int(handle)
    # cupy.cuda.Event — handle is in .ptr (also a generic fallback)
    ptr = getattr(event, "ptr", None)
    if isinstance(ptr, int):
        return ptr
    raise TypeError(
        f"event must be a CUevent handle (int), a torch.cuda or cupy.cuda "
        f"Event/Stream; got {type(event).__name__}"
    )


class Batch:
    """A batch of samples on the device, ready for consumption.

    Use as a context manager to release the slot back to the pool::

        with d.pop() as batch:
            x = torch.from_dlpack(batch)

    The DLPack capsule (``batch.__dlpack__()``) keeps the underlying
    storage alive as long as the consumer holds it; releasing the
    Batch object while a tensor still views it is safe.

    **Deferred release.** If the consumer kicks off an async D2D copy on
    a side stream, the default ``with`` block forces a host-side
    ``cuStreamSynchronize`` on the producer stream before the slot is
    reused. To avoid that block, call :meth:`release` explicitly with
    the consumer's stream or event — damacy will stream-wait on it
    before re-assembling into the slot's buffer::

        batch = d.pop()
        tensor = torch.empty_like(...)  # on side_stream
        with torch.cuda.stream(side_stream):
            tensor.copy_(torch.from_dlpack(batch))
        batch.release(event=side_stream)  # no host sync
    """

    __slots__ = ("_native",)

    def __init__(self, native_batch: _native.Batch) -> None:
        self._native = native_batch

    @property
    def info(self) -> BatchInfo:
        """Snapshot of the on-device batch geometry. Raises after release."""
        return BatchInfo._from_native(self._native.info)

    def release(
        self,
        *,
        event: object | None = None,
    ) -> None:
        """Return the slot to the pool. Idempotent.

        Args:
            event: If ``None`` (default), the slot is freed immediately;
                damacy may reuse the buffer right away, so callers must
                have host-synced any work that reads it. Otherwise the
                slot reuse waits on the supplied CUDA event before
                damacy's assemble kernel writes the buffer again — the
                host returns at once. Accepted forms:

                * ``int`` — raw CUevent handle.
                * ``torch.cuda.Event`` — its ``.cuda_event`` is read.
                * ``torch.cuda.Stream`` — ``record_event()`` is called
                  on it; the recorded event captures the stream's
                  current position.

        Raises:
            DamacyError: If the deferred-release CUDA call fails. On
                failure the slot is still released back to the pool;
                damacy logs and re-raises rather than silently leaking.
        """
        handle = _coerce_cuda_event_handle(event)
        if handle is None:
            self._native.release()
        else:
            self._native.release(handle)

    # DLPack passthrough — torch.from_dlpack(batch) etc. just works.

    def __dlpack__(
        self,
        *,
        stream: int | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> Any:
        return self._native.__dlpack__(
            stream=stream,
            max_version=max_version,
            dl_device=dl_device,
            copy=copy,
        )

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._native.__dlpack_device__()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb  # protocol-required, not consumed
        self.release()

    def __repr__(self) -> str:
        try:
            i = self.info
            return (
                f"Batch(batch_id={i.batch_id}, shape={i.shape}, dtype={i.dtype.name})"
            )
        except Exception:
            return "Batch(<released>)"


# Pairs of (LOCAL_RANK, bound) we've already warned about in this
# process. A loop that constructs many Pipelines under the same
# misconfiguration shouldn't generate one warning per construction.
# The lock guards against a check-then-add race when several threads
# construct pipelines concurrently; the dedup is best-effort either
# way, but with the lock it's deterministic.
_warned_local_rank_pairs: set[tuple[int, int]] = set()
_warned_local_rank_lock = threading.Lock()
_warned_multi_gpu_pairs: set[tuple[int, int]] = set()
_warned_multi_gpu_lock = threading.Lock()


def _warn_if_local_rank_disagrees(cfg_device: int | None, bound: int) -> bool:
    """Warn when ``LOCAL_RANK`` is set but the bound device disagrees;
    silent unless the user looks like they're under torchrun. Skipped
    when the user passed ``Config.device`` explicitly — they've already
    declared their intent and the native cross-check has run. Each
    (LOCAL_RANK, bound) pair warns at most once per process. Returns
    True when a warning is emitted (callers chain other heuristics off
    this to avoid stacking warnings on the same construction)."""
    if cfg_device is not None:
        return False
    raw = os.environ.get("LOCAL_RANK")
    if raw is None:
        return False
    try:
        local_rank = int(raw)
    except ValueError:
        return False
    if local_rank == bound:
        return False
    key = (local_rank, bound)
    with _warned_local_rank_lock:
        if key in _warned_local_rank_pairs:
            return True
        _warned_local_rank_pairs.add(key)
    warnings.warn(
        f"damacy.Pipeline bound to CUDA device {bound} but LOCAL_RANK={local_rank}. "
        f"Did you forget torch.cuda.set_device({local_rank}) before constructing "
        f"the pipeline, or pass Config(device={local_rank}) to bind explicitly?",
        stacklevel=3,
    )
    return True


def _warn_if_multi_gpu_implicit(cfg_device: int | None, bound: int) -> None:
    """On a multi-GPU host with no ``Config.device``, warn that the
    binding came from whatever CUcontext happened to be current. Covers
    the launchers that don't export ``LOCAL_RANK`` (raw ``srun``, manual
    ``CUDA_VISIBLE_DEVICES``) where the bug pattern is every rank silently
    binding to GPU 0. Caller suppresses this when the LOCAL_RANK warning
    already fired — same root cause, one message is enough."""
    if cfg_device is not None:
        return
    count = _native.cuda_device_count()
    if count <= 1:
        return
    key = (count, bound)
    with _warned_multi_gpu_lock:
        if key in _warned_multi_gpu_pairs:
            return
        _warned_multi_gpu_pairs.add(key)
    warnings.warn(
        f"damacy.Pipeline bound to CUDA device {bound} of {count}, "
        f"but you didn't say which device to use. Pass "
        f"`Config(device=...)` to pick one, or call "
        f"`torch.cuda.set_device(...)` before constructing the Pipeline.",
        stacklevel=3,
    )


# ---- Pipeline -----------------------------------------------------------


class Pipeline:
    """Streaming GPU data pipeline. Drive :meth:`push`, :meth:`pop`,
    :meth:`flush`. Stages are plan → host I/O → H2D copy → on-device
    decompress → assemble; output batches are double-buffered (B=2)
    and waves are double-buffered internally.

    A CUcontext must be current on the calling thread when this is
    constructed; PyTorch sets one up implicitly. For bare-Python use,
    call :func:`damacy._native.cuda_init_primary` once first.

    Constructed from a :class:`Config`::

        cfg = damacy.Config(batch_size=8, ...)
        with damacy.Pipeline(cfg) as p:
            ...

    Resource caps are fixed at construction; nothing grows after that.
    """

    __slots__ = (
        "_closed",
        "_config",
        "_native",
        "_pending",
        "_pending_buf",
        "_pop_done",
        "_pop_err",
        "_pop_lock",
        "_pop_result",
        "_pop_thread",
    )

    def __init__(self, config: Config) -> None:
        try:
            self._native = _native.Pipeline(
                batch_size=config.batch_size,
                lookahead_batches=config.lookahead_batches,
                dtype=int(config.dtype),  # already coerced by Config.__init__
                max_chunk_uncompressed_bytes=config.max_chunk_uncompressed_bytes,
                max_read_op_bytes=config.max_read_op_bytes,
                max_gpu_memory_bytes=config.max_gpu_memory_bytes,
                host_buffer_waves=config.host_buffer_waves,
                max_chunks_per_wave=config.max_chunks_per_wave,
                max_substreams_per_chunk=config.max_substreams_per_chunk,
                n_io_threads=config.n_io_threads,
                n_array_meta_cache=config.n_array_meta_cache,
                n_shard_index_cache=config.n_shard_index_cache,
                n_chunk_layout_cache=config.n_chunk_layout_cache,
                sample_shape=tuple(config.sample_shape),
                device=-1 if config.device is None else int(config.device),
                enable_gds=_gds_to_native(config.enable_gds),
                numa_strategy=int(config.numa_strategy),
                numa_node=config.numa_node,
            )
        except _native.DamacyError as exc:
            _reraise_typed(exc)
        self._closed = False
        self._config = config
        # User-side queue of pending sample iterators. push() appends
        # here and best-effort drains; pop()/flush() top up before
        # touching native. This makes push() consume-everything from
        # the user's perspective and lets generators flow naturally.
        # _pending_buf is the head iterator's already-pulled-but-not-yet-
        # pushed samples; held flat to avoid wrapping `it` in successive
        # itertools.chain() layers under sustained backpressure.
        self._pending: deque[Iterator[Sample]] = deque()
        self._pending_buf: list[Sample] = []
        # damacy_pop has no timed variant; on timeout the worker stays
        # parked inside it and the next pop() adopts the same thread.
        self._pop_lock = threading.Lock()
        self._pop_done = threading.Event()
        self._pop_thread: threading.Thread | None = None
        self._pop_result: _native.Batch | None = None
        self._pop_err: BaseException | None = None
        bound = self._native.device
        if not _warn_if_local_rank_disagrees(config.device, bound):
            _warn_if_multi_gpu_implicit(config.device, bound)

    @property
    def device(self) -> int:
        """CUDA device index this pipeline is bound to."""
        self._check_open()
        return int(self._native.device)

    @property
    def config(self) -> Config:
        """The :class:`Config` this loader was built from."""
        return self._config

    # ---- lifecycle ---------------------------------------------------

    def _check_open(self) -> None:
        """Raise :class:`ShutdownError` if close() has already run.
        Called at the top of every method that touches the native handle."""
        if self._closed:
            exc = ShutdownError("damacy: pipeline has been closed")
            exc.status = Status.SHUTDOWN
            exc.what = "closed"
            raise exc

    def close(self) -> None:
        """Release the underlying handle. Idempotent. Subsequent calls
        on the pipeline raise :class:`ShutdownError`."""
        if not self._closed:
            self._closed = True
            del self._native
            t = self._pop_thread
            if t is not None:
                t.join()  # damacy_destroy already woke it with SHUTDOWN
                self._pop_thread = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb  # protocol-required, not consumed
        # Drain any in-flight work so a clean shutdown doesn't drop a
        # partial last batch silently. Pending samples that don't fit
        # are discarded — the user is on the hook for popping
        # everything they pushed before exiting. close() runs in
        # finally so the native handle releases even if flush raises
        # something we don't catch.
        try:
            if not self._closed:
                self.flush()
        except DamacyError:
            pass  # don't mask the user's exception
        finally:
            self.close()

    # ---- pipeline ----------------------------------------------------

    def push(self, samples: Iterable[Sample]) -> None:
        """Queue samples for processing. Accepts any iterable (list,
        generator, infinite generator, …); large or unbounded sources
        are pulled lazily as :meth:`pop` frees space.

        Fatal errors from the C-side validator (``NotFound``,
        ``DtypeMismatch``, ``RankMismatch``, …) raise the matching
        :class:`DamacyError` subclass; the offending iterator is
        discarded but samples accepted by earlier ``push`` calls are
        unaffected.
        """
        self._check_open()
        self._pending.append(iter(samples))
        self._drain_pending()

    def _drain_pending(self) -> None:
        """Move samples from pending iterators into the native lookahead
        until the lookahead is full or all pending iterators are
        exhausted. Best-effort; safe to call any time.

        Unconsumed samples sit in ``_pending_buf`` (always sourced from
        a single head iterator), not re-wrapped onto ``self._pending[0]``
        — successive backpressure events leave the buffer flat instead
        of nesting ``itertools.chain`` layers."""
        cap = self._config.lookahead_batches * self._config.batch_size
        while True:
            # Top up buffer from the head iterator. Buffer is only ever
            # filled from one iterator at a time, so on push failure we
            # know exactly which iterator to drop.
            if not self._pending_buf and self._pending:
                taken = list(itertools.islice(self._pending[0], cap))
                if not taken:
                    self._pending.popleft()
                    continue
                self._pending_buf = taken
            if not self._pending_buf:
                return
            try:
                r = self._native.push([s._to_native() for s in self._pending_buf])
            except _native.DamacyError as exc:
                # Drop the failed iterator and any leftover items it
                # had produced; other pending iterators keep their
                # place and will retry on next drain.
                self._pending_buf = []
                if self._pending:
                    self._pending.popleft()
                _reraise_typed(exc)
            consumed = int(r["consumed"])
            if consumed < len(self._pending_buf):
                # Native is full — keep the unconsumed tail in the
                # buffer; the next drain (after a pop) resumes here.
                del self._pending_buf[:consumed]
                return
            self._pending_buf = []

    def pop(self) -> Batch:
        """Block until the next batch is on-device-ready. Returns a
        :class:`Batch` you can hand to ``torch.from_dlpack`` (or any
        DLPack consumer) — preferably inside a ``with`` block.

        Raises :class:`PoolStarved` if no batch arrives within
        ``Config.pop_timeout_s`` seconds (default 30). Usually that
        means tensors from previous batches are still being held —
        drop them, or ``.clone()`` if you need to keep them."""
        self._check_open()
        self._drain_pending()
        timeout = self._config.pop_timeout_s
        if timeout is None:
            try:
                return Batch(self._native.pop())
            except _native.DamacyError as exc:
                _reraise_typed(exc)
        return self._pop_with_timeout(timeout)

    def _pop_with_timeout(self, timeout: float) -> Batch:
        with self._pop_lock:
            if self._pop_thread is None:
                self._pop_done.clear()
                self._pop_result = None
                self._pop_err = None
                t = threading.Thread(
                    target=self._pop_worker,
                    name="damacy-pop",
                    daemon=True,
                )
                self._pop_thread = t
                t.start()
        if not self._pop_done.wait(timeout):
            raise PoolStarved(
                "Pipeline.pop() timed out waiting for the next batch. "
                "This usually means tensors from previous batches are "
                "still being held — drop those references before the "
                "next pop(), or .clone() if you need to keep them."
            )
        with self._pop_lock:
            err = self._pop_err
            result = self._pop_result
            self._pop_thread = None
            self._pop_result = None
            self._pop_err = None
            self._pop_done.clear()
        if err is not None:
            if isinstance(err, _native.DamacyError):
                _reraise_typed(err)
            raise err
        assert result is not None
        return Batch(result)

    def _pop_worker(self) -> None:
        try:
            self._pop_result = self._native.pop()
        except BaseException as exc:
            self._pop_err = exc
        finally:
            self._pop_done.set()

    def flush(self) -> None:
        """Drain pending samples into the pipeline (best-effort) and
        ready any partial last batch for pop. Idempotent. Pending
        samples that don't fit before flush are dropped — pop until
        :attr:`pending` reads False if you want every queued sample to
        emit as a batch."""
        self._check_open()
        self._drain_pending()
        try:
            self._native.flush()
        except _native.DamacyError as exc:
            _reraise_typed(exc)

    @property
    def pending(self) -> bool:
        """True if push() has accepted samples that haven't yet entered
        the native lookahead. Becomes False as :meth:`pop` frees space."""
        return bool(self._pending) or bool(self._pending_buf)

    def batches(self, n: int) -> Iterator[Batch]:
        """Pop *n* batches as an iterator. Each call to :meth:`pop`
        blocks until that batch is on-device-ready.

        Pair with a ``with`` block so the slot is released::

            for batch in d.batches(8):
                with batch as t:
                    x = torch.from_dlpack(t)
                    ...
        """
        for _ in range(n):
            yield self.pop()

    # ---- stats -------------------------------------------------------

    def stats(self) -> Stats:
        """Cumulative pipeline metrics as a :class:`Stats` snapshot.

        The snapshot is taken at call time; pipeline counters keep
        accumulating in the background. Per-stage :class:`Metric`
        fields carry cumulative milliseconds, the best single
        observation, input/output byte totals, and a sample count.
        Cache hit/miss counters and lifetime totals (batches emitted,
        waves emitted, chunks dispatched, …) round out the snapshot.

        ``gpu_bytes_committed`` reflects the live GPU footprint
        counted against ``Config.max_gpu_memory_bytes``; it grows from
        wave-init to first pop (lazy batch-output sizing) and stays
        flat afterward.

        Use :meth:`stats_reset` to zero the cumulative timing
        counters. ``gpu_bytes_committed`` is not reset — it reflects
        the live commitment, not a delta.

        Raises:
            ShutdownError: If the pipeline has been closed.
        """
        self._check_open()
        return Stats._from_native(self._native.stats())

    def stats_reset(self) -> None:
        """Zero the cumulative timing counters and per-stage rolling
        totals. Cache hit/miss counters and ``gpu_bytes_committed`` are
        left alone — they reflect live state, not deltas.

        Raises:
            ShutdownError: If the pipeline has been closed.
        """
        self._check_open()
        self._native.stats_reset()


# ---- logging-bridge defaults --------------------------------------------

# Avoid "no handler for damacy" warnings in apps that don't configure
# logging; users opt in by attaching their own handler / level.
logging.getLogger(__name__).addHandler(logging.NullHandler())
