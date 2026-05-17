"""Deferred-release path (issue #35).

The deterministic race coverage for ``Batch.release(event=...)`` lives in
the C-level ``test_release_event_blocks_assemble`` (tests/test_damacy.c)
which uses ``cudaLaunchHostFunc`` to gate the side-stream D2H behind a
100 ms host sleep, then asserts the captured bytes belong to the
released batch. The Python wrapper is a thin handle-coercion shim, so
the tests here cover only that surface.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import damacy
import pytest
from damacy import Config, Pipeline, Sample


def _make_zarr(tmp_path: Path, write_zarr_script: Path, name: str) -> str:
    if not shutil.which("uv"):
        pytest.skip("uv not on PATH")
    out = tmp_path / name
    cmd = [
        "uv", "run", "--script", str(write_zarr_script),
        "--out", str(out),
        "--shape", "8,16", "--inner", "8,16", "--shard", "8,16",
        "--dtype", "uint16", "--offset", "0",
        "--codec", "blosc-zstd",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        pytest.skip(f"write_zarr.py failed (rc={r.returncode})")
    return str(out)


@pytest.fixture
def one_zarr(tmp_path: Path, write_zarr_script: Path) -> str:
    return _make_zarr(tmp_path, write_zarr_script, "a")


def _mk_cfg() -> Config:
    return Config(
        batch_size=1,
        dtype="f32",
        lookahead_batches=2,
        n_io_threads=1,
        n_zarrs_meta_cache=4,
        n_shards_meta_cache=4,
        sample_shape=(8, 16),
    )


def test_release_event_none_falls_back_to_immediate(one_zarr):
    """``release(event=None)`` is the same as ``release()``."""
    with Pipeline(_mk_cfg()) as d:
        d.push([Sample(uri=one_zarr, aabb=[(0, 8), (0, 16)])])
        b = d.pop()
        b.release(event=None)
        b.release()  # idempotent


def test_release_event_rejects_bad_type(one_zarr):
    with Pipeline(_mk_cfg()) as d:
        d.push([Sample(uri=one_zarr, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch, pytest.raises(TypeError, match="event must be"):
            batch.release(event=b"nope")  # type: ignore[arg-type]


def test_coerce_cuda_event_handle_accepts_ints():
    assert damacy._coerce_cuda_event_handle(None) is None
    assert damacy._coerce_cuda_event_handle(0xDEADBEEF) == 0xDEADBEEF


def test_coerce_cuda_event_handle_reads_event_like():
    class FakeEvent:
        cuda_event = 42

    assert damacy._coerce_cuda_event_handle(FakeEvent()) == 42


def test_coerce_cuda_event_handle_records_stream_like():
    class FakeEvent:
        cuda_event = 1234

    class FakeStream:
        def record_event(self):
            return FakeEvent()

    assert damacy._coerce_cuda_event_handle(FakeStream()) == 1234


def test_coerce_cuda_event_handle_reads_cupy_event_ptr():
    """CuPy events expose their handle through `.ptr`."""

    class FakeCupyEvent:
        ptr = 0xCAFE

    assert damacy._coerce_cuda_event_handle(FakeCupyEvent()) == 0xCAFE


def test_coerce_cuda_event_handle_records_cupy_stream():
    """CuPy streams use `.record()` (not `.record_event()`)."""

    class FakeCupyEvent:
        ptr = 0xBEEF

    class FakeCupyStream:
        def record(self):
            return FakeCupyEvent()

    assert damacy._coerce_cuda_event_handle(FakeCupyStream()) == 0xBEEF


def test_coerce_cuda_event_handle_rejects_unknown():
    with pytest.raises(TypeError, match="event must be"):
        damacy._coerce_cuda_event_handle(object())
