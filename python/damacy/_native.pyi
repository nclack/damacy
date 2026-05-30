# Type stubs for the C extension. Hand-written to mirror python/damacy/_api.c
# and python/damacy/_native.c. The user-facing surface lives in damacy/__init__.py;
# treat this module as internal.
from __future__ import annotations

from typing import Any, Final

__version__: Final[str]

# ---- log-level constants (mirror damacy_log.h) --------------------------

LOG_TRACE: Final[int]
LOG_DEBUG: Final[int]
LOG_INFO: Final[int]
LOG_WARN: Final[int]
LOG_ERROR: Final[int]
LOG_FATAL: Final[int]

# ---- damacy_status integers ---------------------------------------------

STATUS_OK: Final[int]
STATUS_AGAIN: Final[int]
STATUS_INVAL: Final[int]
STATUS_NOTFOUND: Final[int]
STATUS_DTYPE: Final[int]
STATUS_RANK: Final[int]
STATUS_IO: Final[int]
STATUS_DECODE: Final[int]
STATUS_CUDA: Final[int]
STATUS_OOM: Final[int]
STATUS_BUDGET: Final[int]
STATUS_SHUTDOWN: Final[int]

# ---- damacy_dtype integers ----------------------------------------------

DTYPE_F32: Final[int]
DTYPE_BF16: Final[int]

# ---- damacy_numa_strategy integers --------------------------------------

NUMA_AUTO: Final[int]
NUMA_DISABLED: Final[int]
NUMA_PIN_TO: Final[int]

# ---- damacy_gds_mode integers -------------------------------------------

GDS_AUTO: Final[int]
GDS_ON: Final[int]
GDS_OFF: Final[int]

# ---- exceptions ---------------------------------------------------------

class DamacyError(RuntimeError):
    """Raised by the native pipeline. .status is one of STATUS_*; .what
    names the failing stage (e.g. "create", "pop")."""

    status: int
    what: str

# ---- log-sink helpers ---------------------------------------------------

def set_log_level(level: int, /) -> None: ...
def set_log_quiet(quiet: bool, /) -> None: ...
def cuda_init_primary(device: int = 0) -> None: ...
def cuda_device_count() -> int: ...
def _log_emit(level: int, msg: str, /) -> None: ...
def _log_emit_from_thread(level: int, msg: str, /) -> None: ...

# ---- core types ---------------------------------------------------------

class Batch:
    """Native batch handle. Yielded by Pipeline.pop()."""

    @property
    def info(self) -> dict[str, Any]:
        """Snapshot of damacy_batch_info as a dict with keys
        device_ptr (int), shape (tuple[int, ...]), dtype (str),
        ready_stream (int), batch_id (int)."""

    def release(self, event: int | None = None) -> None:
        """Return the batch slot to the pool. Idempotent.

        With no argument or ``event=None`` the slot is freed
        immediately. With ``event`` set to an integer CUevent handle,
        damacy queues a ``cuStreamWaitEvent`` against its internal
        stream_post before reusing the slot's buffer; the host returns
        at once instead of synchronizing."""

    def __dlpack__(
        self,
        *,
        stream: int | None = ...,
        max_version: tuple[int, int] | None = ...,
        dl_device: tuple[int, int] | None = ...,
        copy: bool | None = ...,
    ) -> Any:
        """DLPack capsule export. ``stream`` is honored per the
        protocol (consumer's stream waits on damacy's ``ready_stream``).
        ``copy=True`` raises :class:`BufferError` — the batch is on the
        device damacy assembled it on, no implicit copy is performed.

        ``max_version`` selects the wire format:

        - ``None`` (default) → v0 ``"dltensor"`` capsule. PyTorch 2.8
          and other legacy consumers expect this.
        - ``(1, 0)`` or higher → v1.0 ``"dltensor_versioned"`` capsule.
          CuPy and array-API-spec consumers ask for this.

        ``dl_device`` is accepted for protocol compatibility but ignored;
        the producer always emits on the assembling device."""

    def __dlpack_device__(self) -> tuple[int, int]:
        """Returns (kDLCUDA=2, ordinal)."""

class Pipeline:
    """Native streaming-pipeline handle.

    Phase 5: ``max_gpu_memory_bytes`` is the primary budget knob. The
    create-time resolver picks per-wave geometry that fits inside the
    budget *including* worst-case observe-and-grow headroom for the
    shared decoder scratch and per-wave fanout SOAs, so grows inside a
    successfully-created instance never trip the cap.

    Internal sizing derives from ``max_gpu_memory_bytes``.
    """

    def __init__(
        self,
        samples_per_batch: int,
        lookahead_batches: int,
        dtype: str | int,
        max_chunk_uncompressed_bytes: int,
        max_gpu_memory_bytes: int,
        n_io_threads: int,
        n_array_meta_cache: int,
        n_shard_index_cache: int,
        n_chunk_layout_cache: int,
        sample_shape: tuple[int, ...],
        host_buffer_waves: int = 0,
        max_chunks_per_wave: int = 0,
        max_substreams_per_chunk: int = 0,
        max_read_op_bytes: int = 0,
        device: int = -1,
        enable_gds: int = GDS_AUTO,
        numa_strategy: int = NUMA_AUTO,
        numa_node: int = -1,
        bypass_decode: bool = False,
    ) -> None: ...
    @property
    def device(self) -> int: ...
    def push(
        self, samples: list[dict[str, Any]] | tuple[dict[str, Any], ...]
    ) -> dict[str, Any]:
        """Push a sequence of {uri, aabb} dicts.
        Returns {consumed: int, status: int}. Raises DamacyError for
        anything other than OK / AGAIN."""

    def pop(self) -> Batch:
        """Block until the next batch is on-device-ready. Raises
        DamacyError on failure."""

    def flush(self) -> None: ...
    def stats(self) -> dict[str, Any]: ...
    def stats_reset(self) -> None: ...
