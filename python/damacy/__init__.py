"""High-throughput streaming loader for sharded NGFF zarr stores.

The native extension is loaded eagerly at import so the in-process log
sink is registered before any C threads spin up — log records produced
by C threads are routed to ``logging.getLogger("damacy")`` via a
daemon-thread drain that holds the GIL only while delivering messages.

Typical use::

    import damacy, torch

    cfg = damacy.Config(
        batch_size=8,
        host_buffer_bytes=1 << 30,
        device_buffer_bytes=1 << 30,
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

    Bare ints in ``aabb`` are rejected so the behaviour stays
    consistent with NumPy/zarr indexing semantics
    (``np.s_[64]`` means "point 64", not "extent (0, 64)"):

    >>> Sample(uri="cell.zarr", aabb=[64, 256, 256])
    Traceback (most recent call last):
        ...
    TypeError: aabb axis 0: expected slice or (start, stop) tuple; got int

    Slice validation rejects strided slices and unbounded stops:

    >>> Sample(uri="cell.zarr", aabb=[slice(0, 64, 2), slice(0, 256), slice(0, 256)])
    Traceback (most recent call last):
        ...
    ValueError: aabb axis 0: slice step must be 1 or omitted (got step=2)
    >>> Sample(uri="cell.zarr", aabb=[slice(0, None), slice(0, 256), slice(0, 256)])
    Traceback (most recent call last):
        ...
    ValueError: aabb axis 0: slice stop is required (got slice(0, None, None))
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


@dataclass(frozen=True, slots=True, kw_only=True)
class Config:
    """All resource caps and pipeline shape, fixed at create time.

    Build variants with :func:`dataclasses.replace`:

    >>> import dataclasses
    >>> base = Config(batch_size=8,
    ...               host_buffer_bytes=1 << 30, device_buffer_bytes=1 << 30)
    >>> base.dtype is Dtype.F32
    True
    >>> dataclasses.replace(base, batch_size=64).batch_size
    64

    Validation runs in ``__post_init__`` so invalid configs fail before
    we touch CUDA. The dtype field accepts :class:`Dtype`, an int, or
    one of ``"f32"`` / ``"float32"`` / ``"bf16"`` / ``"bfloat16"``.

    >>> Config(batch_size=0,
    ...        host_buffer_bytes=1, device_buffer_bytes=1)
    Traceback (most recent call last):
        ...
    ValueError: batch_size must be >= 1 (got 0)

    Attributes:
        batch_size: Samples per batch (>= 1).
        host_buffer_bytes: Pinned-host staging budget; sized for IO bw.
        device_buffer_bytes: Device decompress-scratch budget.
        dtype: Destination dtype for assembled batches.
        lookahead_batches: User-side push-queue depth (>= 2).
        n_io_threads: IO worker threads (>= 1).
        n_zarrs_meta_cache: LRU cap for zarr-metadata entries.
        n_shards_meta_cache: LRU cap for shard-index entries.
        max_chunk_uncompressed_bytes: Largest uncompressed chunk size
            the pipeline accepts; 0 selects the C default (512 KB).
            Values exceeding :data:`MAX_CHUNK_UNCOMPRESSED_BYTES` are
            rejected at create.
        max_gpu_memory_bytes: Hard cap on GPU memory allocated for
            wave-resident buffers and batch-output pools. 0 = no cap.
        max_bytes_per_element: Largest source dtype size (bytes) the
            pipeline will accept; 0 = the codec ceiling.
        device: CUDA device index to bind. ``None`` (default) captures
            the current ``CUcontext`` on the calling thread; pass an
            int (e.g. ``local_rank``) to retain that device's primary
            context internally — recommended under torchrun / MPI.
    """

    batch_size: int
    host_buffer_bytes: int
    device_buffer_bytes: int
    dtype: Dtype | str | int = Dtype.F32
    lookahead_batches: int = 2
    n_io_threads: int = 4
    n_zarrs_meta_cache: int = 64
    n_shards_meta_cache: int = 256
    max_chunk_uncompressed_bytes: int = 0
    max_gpu_memory_bytes: int = 0
    max_bytes_per_element: int = 0
    device: int | None = None

    def __post_init__(self) -> None:
        # Coerce dtype eagerly so reading .dtype always yields a Dtype.
        # frozen=True forbids `self.dtype = ...`, hence object.__setattr__.
        object.__setattr__(self, "dtype", Dtype.coerce(self.dtype))
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 (got {self.batch_size})")
        if self.lookahead_batches < 2:
            raise ValueError(
                f"lookahead_batches must be >= 2 (got {self.lookahead_batches})"
            )
        if self.n_io_threads < 1:
            raise ValueError(f"n_io_threads must be >= 1 (got {self.n_io_threads})")
        if self.host_buffer_bytes <= 0 or self.device_buffer_bytes <= 0:
            raise ValueError("host/device_buffer_bytes must be positive")
        if self.max_chunk_uncompressed_bytes < 0:
            raise ValueError("max_chunk_uncompressed_bytes must be >= 0")


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
    decompress: Metric
    assemble: Metric
    pop_wait_io: Metric
    pop_wait_compute: Metric
    flush_wait: Metric
    zarr_meta_hits: int
    zarr_meta_misses: int
    shard_idx_hits: int
    shard_idx_misses: int
    batches_emitted: int
    batches_truncated: int
    waves_emitted: int
    gpu_bytes_committed: int

    @classmethod
    def _from_native(cls, st: dict[str, Any]) -> Stats:
        m = Metric._from_native
        return cls(
            plan=m(st["plan"]),
            io=m(st["io"]),
            h2d=m(st["h2d"]),
            decompress=m(st["decompress"]),
            assemble=m(st["assemble"]),
            pop_wait_io=m(st["pop_wait_io"]),
            pop_wait_compute=m(st["pop_wait_compute"]),
            flush_wait=m(st["flush_wait"]),
            zarr_meta_hits=st["zarr_meta_hits"],
            zarr_meta_misses=st["zarr_meta_misses"],
            shard_idx_hits=st["shard_idx_hits"],
            shard_idx_misses=st["shard_idx_misses"],
            batches_emitted=st["batches_emitted"],
            batches_truncated=st["batches_truncated"],
            waves_emitted=st["waves_emitted"],
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


class Batch:
    """A batch of samples on the device, ready for consumption.

    Use as a context manager to release the slot back to the pool::

        with d.pop() as batch:
            x = torch.from_dlpack(batch)

    The DLPack capsule (``batch.__dlpack__()``) keeps the underlying
    storage alive as long as the consumer holds it; releasing the
    Batch object while a tensor still views it is safe.
    """

    __slots__ = ("_native",)

    def __init__(self, native_batch: _native.Batch) -> None:
        self._native = native_batch

    @property
    def info(self) -> BatchInfo:
        return BatchInfo._from_native(self._native.info)

    def release(self) -> None:
        """Return the slot to the pool. Idempotent."""
        self._native.release()

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
_warned_local_rank_pairs: set[tuple[int, int]] = set()


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

    __slots__ = ("_closed", "_config", "_native", "_pending")

    def __init__(self, config: Config) -> None:
        try:
            self._native = _native.Pipeline(
                batch_size=config.batch_size,
                lookahead_batches=config.lookahead_batches,
                n_io_threads=config.n_io_threads,
                host_buffer_bytes=config.host_buffer_bytes,
                device_buffer_bytes=config.device_buffer_bytes,
                n_zarrs_meta_cache=config.n_zarrs_meta_cache,
                n_shards_meta_cache=config.n_shards_meta_cache,
                dtype=int(config.dtype),  # already coerced in __post_init__
                max_chunk_uncompressed_bytes=config.max_chunk_uncompressed_bytes,
                max_gpu_memory_bytes=config.max_gpu_memory_bytes,
                max_bytes_per_element=config.max_bytes_per_element,
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
        self._pending: deque[Iterator[Sample]] = deque()
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
        exhausted. Best-effort; safe to call any time."""
        cap = self._config.lookahead_batches * self._config.batch_size
        while self._pending:
            it = self._pending[0]
            chunk = list(itertools.islice(it, cap))
            if not chunk:
                self._pending.popleft()
                continue
            try:
                r = self._native.push([s._to_native() for s in chunk])
            except _native.DamacyError as exc:
                # Drop the failed iterator; other pending iterators
                # keep their place and will retry on next drain.
                self._pending.popleft()
                _reraise_typed(exc)
            consumed = int(r["consumed"])
            if consumed < len(chunk):
                # Native is full — put the unconsumed back at the head
                # so the next drain (after a pop) resumes from here.
                self._pending[0] = itertools.chain(chunk[consumed:], it)
                return

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
        return bool(self._pending)

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
        self._check_open()
        return Stats._from_native(self._native.stats())

    def stats_reset(self) -> None:
        self._check_open()
        self._native.stats_reset()


# ---- logging-bridge defaults --------------------------------------------

# Avoid "no handler for damacy" warnings in apps that don't configure
# logging; users opt in by attaching their own handler / level.
logging.getLogger(__name__).addHandler(logging.NullHandler())
