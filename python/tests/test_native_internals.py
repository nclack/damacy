"""Tests for ``damacy._native`` internals not exposed by the public wrapper.

The user-facing surface lives in ``damacy``; everything here pokes at the
C extension directly. Kept narrow on purpose — these tests survive only
as long as the corresponding C-extension symbols do.
"""

from __future__ import annotations

import logging
import time

import pytest
from damacy import _native

# Touches the C extension; needs a primary CUDA context.
pytestmark = pytest.mark.usefixtures("cuda_ctx")


def test_module_constants():
    assert _native.LOG_TRACE == 0
    assert _native.LOG_FATAL == 5


def test_status_constants_present():
    # Every enumerator from damacy_status_str must be a module-level int.
    for name in (
        "STATUS_OK",
        "STATUS_AGAIN",
        "STATUS_INVAL",
        "STATUS_NOTFOUND",
        "STATUS_DTYPE",
        "STATUS_RANK",
        "STATUS_IO",
        "STATUS_DECODE",
        "STATUS_CUDA",
        "STATUS_OOM",
        "STATUS_BUDGET",
        "STATUS_SHUTDOWN",
    ):
        assert isinstance(getattr(_native, name), int), name


def test_log_sink_routes_to_python_logger(caplog):
    # Drives a synthetic record from a freshly-spawned C thread through
    # the lock-free ring → drain thread → logging.getLogger("damacy").
    with caplog.at_level(logging.ERROR, logger="damacy"):
        _native._log_emit_from_thread(_native.LOG_ERROR, "pytest-smoke")
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if any("pytest-smoke" in r.getMessage() for r in caplog.records):
                break
            time.sleep(0.02)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("pytest-smoke" in m for m in msgs), msgs


def test_native_damacy_error_carries_status_and_what():
    # Cheapest INVAL trigger: max_gpu_memory_bytes=0.
    with pytest.raises(_native.DamacyError) as excinfo:
        _native.Pipeline(
            samples_per_batch=1,
            lookahead_batches=2,
            n_io_threads=1,
            n_array_meta_cache=4,
            n_shard_index_cache=4,
            n_chunk_layout_cache=4,
            dtype="f32",
            max_chunk_uncompressed_bytes=512 << 10,
            max_gpu_memory_bytes=0,
            sample_shape=(8, 16),
        )
    assert excinfo.value.status == _native.STATUS_INVAL
    assert excinfo.value.what == "create"
    # DamacyError subclasses RuntimeError so legacy callers still work.
    assert isinstance(excinfo.value, RuntimeError)
