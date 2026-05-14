"""High-throughput streaming loader for sharded NGFF zarr stores.

The native extension is loaded eagerly at import so the in-process log
sink is registered before any C threads spin up — log records produced
by C threads are routed to ``logging.getLogger("damacy")`` via a
daemon-thread drain that holds the GIL only while delivering messages.

Typical use::

    import damacy, torch

    cfg = damacy.Config(
        batch_size=8,
        max_gpu_memory_bytes=1 << 30,  # 1 GiB GPU budget
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
from collections.abc import Iterable, Iterator
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
    "Config",
    "DamacyError",
    "DecodeError",
    "Dtype",
    "DtypeMismatch",
    "InvalidArgument",
    "Metric",
    "NativeCudaError",
    "NotFound",
    "OutOfMemory",
    "Pipeline",
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
    """Configured GPU memory cap would be exceeded."""


class ShutdownError(DamacyError):
    """Pipeline destroyed or in failed state."""


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
    >>> base = Config(batch_size=8, max_gpu_memory_bytes=1 << 30)
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
    >>> Config(batch_size=0)
    Traceback (most recent call last):
        ...
    ValueError: batch_size must be >= 1 (got 0)

    ```

    Attributes:
        batch_size: Samples per batch (>= 1).
        max_gpu_memory_bytes: Primary GPU budget knob. Hard cap on
            GPU memory allocated for wave-resident buffers, decoder
            scratch, per-wave fanout SOAs, and batch-output pools. 0
            selects the default (~1 GB). Internal sizing (host slab,
            dev decompress arena, nvcomp temp) is derived from this
            value; the create-time resolver also reserves the
            worst-case observe-and-grow footprint so grows inside a
            successfully-created instance never trip the cap.
        dtype: Destination dtype for assembled batches.
        lookahead_batches: User-side push-queue depth (>= 2).
        n_io_threads: IO worker threads (>= 1).
        n_compute_threads: Background workers for blosc1 chunk-header
            parsing (>= 0). 0 runs parsing serially on the calling
            thread; > 0 spawns a fork-join pool. Total parallelism is
            ``n_compute_threads + 1`` (caller participates as tid 0).
        n_zarrs_meta_cache: LRU cap for zarr-metadata entries.
        n_shards_meta_cache: LRU cap for shard-index entries.
        max_chunk_uncompressed_bytes: Largest uncompressed chunk size
            the pipeline accepts; 0 selects the C default (512 KB).
            Values exceeding :data:`MAX_CHUNK_UNCOMPRESSED_BYTES` are
            rejected at create.
        device: CUDA device index to bind. ``None`` (default) captures
            the current ``CUcontext`` on the calling thread; pass an
            int (e.g. ``local_rank``) to retain that device's primary
            context internally — recommended under torchrun / MPI.
    """

    batch_size: int
    dtype: Dtype
    lookahead_batches: int
    n_io_threads: int
    n_compute_threads: int
    n_zarrs_meta_cache: int
    n_shards_meta_cache: int
    max_chunk_uncompressed_bytes: int
    max_gpu_memory_bytes: int
    host_buffer_waves: int
    device: int | None

    def __init__(
        self,
        *,
        batch_size: int,
        max_gpu_memory_bytes: int = 0,
        dtype: Dtype | str | int = Dtype.F32,
        lookahead_batches: int = 2,
        n_io_threads: int = 4,
        n_compute_threads: int = 0,
        n_zarrs_meta_cache: int = 64,
        n_shards_meta_cache: int = 256,
        max_chunk_uncompressed_bytes: int = 0,
        host_buffer_waves: int = 0,
        device: int | None = None,
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
        if n_compute_threads < 0:
            raise ValueError(
                f"n_compute_threads must be >= 0 (got {n_compute_threads})"
            )
        if max_chunk_uncompressed_bytes < 0:
            raise ValueError("max_chunk_uncompressed_bytes must be >= 0")
        if max_gpu_memory_bytes < 0:
            raise ValueError("max_gpu_memory_bytes must be >= 0")
        if host_buffer_waves < 0:
            raise ValueError("host_buffer_waves must be >= 0")
        set_ = object.__setattr__  # frozen=True forbids `self.x = ...`
        set_(self, "batch_size", batch_size)
        set_(self, "dtype", Dtype.coerce(dtype))
        set_(self, "lookahead_batches", lookahead_batches)
        set_(self, "n_io_threads", n_io_threads)
        set_(self, "n_compute_threads", n_compute_threads)
        set_(self, "n_zarrs_meta_cache", n_zarrs_meta_cache)
        set_(self, "n_shards_meta_cache", n_shards_meta_cache)
        set_(self, "max_chunk_uncompressed_bytes", max_chunk_uncompressed_bytes)
        set_(self, "max_gpu_memory_bytes", max_gpu_memory_bytes)
        set_(self, "host_buffer_waves", host_buffer_waves)
        set_(self, "device", device)


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
    decompress_parse: Metric
    assemble: Metric
    bind_wait: Metric
    pop_wait: Metric
    flush_wait: Metric
    zarr_meta_hits: int
    zarr_meta_misses: int
    shard_idx_hits: int
    shard_idx_misses: int
    batches_emitted: int
    batches_truncated: int
    waves_emitted: int
    chunks_dispatched: int
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
            decompress_parse=m(st["decompress_parse"]),
            assemble=m(st["assemble"]),
            bind_wait=m(st["bind_wait"]),
            pop_wait=m(st["pop_wait"]),
            flush_wait=m(st["flush_wait"]),
            zarr_meta_hits=st["zarr_meta_hits"],
            zarr_meta_misses=st["zarr_meta_misses"],
            shard_idx_hits=st["shard_idx_hits"],
            shard_idx_misses=st["shard_idx_misses"],
            batches_emitted=st["batches_emitted"],
            batches_truncated=st["batches_truncated"],
            waves_emitted=st["waves_emitted"],
            chunks_dispatched=st["chunks_dispatched"],
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
      * Anything with a ``cuda_event`` attribute (torch.cuda.Event) — read it.
      * Anything with a ``record_event()`` method (torch.cuda.Stream) — call it
        and re-coerce the returned object.

    Returns the integer handle, or ``None`` if ``event is None``.

    Raises ``TypeError`` for anything else. The Stream path records on
    the user's stream at call time, so the resulting event captures the
    stream's then-current position — exactly what the user wants when
    asking "don't reuse until my stream finishes the work I just queued".
    """
    if event is None:
        return None
    if isinstance(event, int):
        return event
    # torch.cuda.Stream
    if hasattr(event, "record_event"):
        rec = event.record_event()  # type: ignore[union-attr]
        return _coerce_cuda_event_handle(rec)
    # torch.cuda.Event
    handle = getattr(event, "cuda_event", None)
    if handle is not None:
        return int(handle)
    raise TypeError(
        f"event must be a CUevent handle (int), a torch.cuda.Event, or a "
        f"torch.cuda.Stream; got {type(event).__name__}"
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


def _warn_if_local_rank_disagrees(cfg_device: int | None, bound: int) -> None:
    """Warn when ``LOCAL_RANK`` is set but the bound device disagrees;
    silent unless the user looks like they're under torchrun. Skipped
    when the user passed ``Config.device`` explicitly — they've already
    declared their intent and the native cross-check has run. Each
    (LOCAL_RANK, bound) pair warns at most once per process."""
    if cfg_device is not None:
        return
    raw = os.environ.get("LOCAL_RANK")
    if raw is None:
        return
    try:
        local_rank = int(raw)
    except ValueError:
        return
    if local_rank == bound:
        return
    key = (local_rank, bound)
    with _warned_local_rank_lock:
        if key in _warned_local_rank_pairs:
            return
        _warned_local_rank_pairs.add(key)
    warnings.warn(
        f"damacy.Pipeline bound to CUDA device {bound} but LOCAL_RANK={local_rank}. "
        f"Did you forget torch.cuda.set_device({local_rank}) before constructing "
        f"the pipeline, or pass Config(device={local_rank}) to bind explicitly?",
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

    __slots__ = ("_closed", "_config", "_native", "_pending", "_pending_buf")

    def __init__(self, config: Config) -> None:
        try:
            self._native = _native.Pipeline(
                batch_size=config.batch_size,
                lookahead_batches=config.lookahead_batches,
                n_io_threads=config.n_io_threads,
                n_compute_threads=config.n_compute_threads,
                n_zarrs_meta_cache=config.n_zarrs_meta_cache,
                n_shards_meta_cache=config.n_shards_meta_cache,
                dtype=int(config.dtype),  # already coerced by Config.__init__
                max_chunk_uncompressed_bytes=config.max_chunk_uncompressed_bytes,
                max_gpu_memory_bytes=config.max_gpu_memory_bytes,
                host_buffer_waves=config.host_buffer_waves,
                device=-1 if config.device is None else int(config.device),
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
        _warn_if_local_rank_disagrees(config.device, self._native.device)

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
        DLPack consumer) — preferably inside a ``with`` block."""
        self._check_open()
        # Top up native from the pending queue so the planner has work.
        self._drain_pending()
        try:
            return Batch(self._native.pop())
        except _native.DamacyError as exc:
            _reraise_typed(exc)

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
