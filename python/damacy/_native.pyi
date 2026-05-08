# Type stubs for the C extension. Hand-written to mirror python/damacy/_api.c
# and python/damacy/_native.c. The user-facing surface lives in damacy/__init__.py;
# treat this module as internal.
from __future__ import annotations

from typing import Any, Final

__version__: Final[str]

# ---- compile-time ceilings ----------------------------------------------

MAX_CHUNK_UNCOMPRESSED_BYTES: Final[int]

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
STATUS_SHUTDOWN: Final[int]

# ---- damacy_dtype integers ----------------------------------------------

DTYPE_F32: Final[int]
DTYPE_BF16: Final[int]

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

    def release(self) -> None:
        """Return the batch slot to the pool. Idempotent."""

    def __dlpack__(
        self,
        *,
        stream: int | None = ...,
        max_version: tuple[int, int] | None = ...,
        dl_device: tuple[int, int] | None = ...,
        copy: bool | None = ...,
    ) -> Any:
        """DLPack v1 capsule export. Honors stream=... per the protocol."""

    def __dlpack_device__(self) -> tuple[int, int]:
        """Returns (kDLCUDA=2, ordinal)."""

class Pipeline:
    """Native streaming-pipeline handle."""

    def __init__(
        self,
        batch_size: int,
        lookahead_batches: int,
        n_io_threads: int,
        host_buffer_bytes: int,
        device_buffer_bytes: int,
        n_zarrs_meta_cache: int,
        n_shards_meta_cache: int,
        dtype: str | int,
        max_chunk_uncompressed_bytes: int,
        max_gpu_memory_bytes: int = 0,
        max_bytes_per_element: int = 0,
        device: int = -1,
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
