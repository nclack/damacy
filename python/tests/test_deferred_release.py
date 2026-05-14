"""Deferred-release path (issue #35).

Verifies that ``Batch.release(event=...)`` stream-waits on the user's
event before damacy reuses the slot's buffer. The crucial property is
that an async D2D/D2H copy on a side stream observes the OLD batch's
content even after the host has returned from ``release`` and a new
batch has been popped into the same slot.

The CUDA driver is poked via ``ctypes`` rather than torch — keeps the
test free of optional deps while still exercising real side streams.
"""

from __future__ import annotations

import ctypes
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import damacy
import pytest
from damacy import Config, Pipeline, Sample

# ---- thin CUDA driver bindings -----------------------------------------

# The driver API names we touch. Function pointers cache after the
# first call; ctypes resolves by symbol on the shared lib.
_cu = ctypes.CDLL("libcuda.so")


def _check(name: str, rc: int) -> None:
    if rc != 0:
        # Map a few common codes for readability; full table is in cuda.h.
        raise RuntimeError(f"{name} failed: CUresult={rc}")


# CUstream, CUevent, CUdeviceptr are all opaque uintptr_t-sized handles.
def stream_create() -> int:
    s = ctypes.c_void_p()
    _check("cuStreamCreate", _cu.cuStreamCreate(ctypes.byref(s), 0))
    return int(s.value) if s.value else 0


def stream_destroy(s: int) -> None:
    _check("cuStreamDestroy_v2", _cu.cuStreamDestroy_v2(ctypes.c_void_p(s)))


def stream_sync(s: int) -> None:
    _check("cuStreamSynchronize", _cu.cuStreamSynchronize(ctypes.c_void_p(s)))


def event_create() -> int:
    e = ctypes.c_void_p()
    # CU_EVENT_DISABLE_TIMING = 2
    _check("cuEventCreate", _cu.cuEventCreate(ctypes.byref(e), 2))
    return int(e.value) if e.value else 0


def event_record(ev: int, stream: int) -> None:
    _check(
        "cuEventRecord", _cu.cuEventRecord(ctypes.c_void_p(ev), ctypes.c_void_p(stream))
    )


def event_destroy(ev: int) -> None:
    _check("cuEventDestroy_v2", _cu.cuEventDestroy_v2(ctypes.c_void_p(ev)))


def memcpy_dtoh_async(host_ptr: int, dev_ptr: int, n_bytes: int, stream: int) -> None:
    _cu.cuMemcpyDtoHAsync_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    _check(
        "cuMemcpyDtoHAsync_v2",
        _cu.cuMemcpyDtoHAsync_v2(
            ctypes.c_void_p(host_ptr),
            ctypes.c_size_t(dev_ptr),
            ctypes.c_size_t(n_bytes),
            ctypes.c_void_p(stream),
        ),
    )


def mem_alloc_host(n_bytes: int) -> tuple[int, ctypes.Array]:
    """Pinned-host allocation via cuMemAllocHost. Returns (ptr, buffer)
    where ``buffer`` must outlive ``ptr`` for the bytes to stay mapped."""
    buf = ctypes.c_void_p()
    _check(
        "cuMemAllocHost_v2",
        _cu.cuMemAllocHost_v2(ctypes.byref(buf), ctypes.c_size_t(n_bytes)),
    )
    return int(buf.value) if buf.value else 0, buf  # type: ignore[return-value]


def mem_free_host(ptr: int) -> None:
    _check("cuMemFreeHost", _cu.cuMemFreeHost(ctypes.c_void_p(ptr)))


# ---- fixtures: two zarrs with distinct content -------------------------


def _make_zarr(tmp_path: Path, write_zarr_script: Path, name: str, offset: int) -> str:
    if not shutil.which("uv"):
        pytest.skip("uv not on PATH")
    out = tmp_path / name
    cmd = [
        "uv", "run", "--script", str(write_zarr_script),
        "--out", str(out),
        "--shape", "8,16", "--inner", "8,16", "--shard", "8,16",
        "--dtype", "uint16", "--offset", str(offset),
        "--codec", "blosc-zstd",
    ]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        pytest.skip(f"write_zarr.py failed (rc={r.returncode})")
    return str(out)


@pytest.fixture
def two_zarrs(tmp_path: Path, write_zarr_script: Path) -> tuple[str, str]:
    """A pair of distinct u16 zarrs: zarr_a content = arange,
    zarr_b content = arange + 10_000 (mod 2**16)."""
    a = _make_zarr(tmp_path, write_zarr_script, "a", offset=0)
    b = _make_zarr(tmp_path, write_zarr_script, "b", offset=10_000)
    return a, b


# ---- expected content --------------------------------------------------

# tiny zarr is 8x16 u16; batch_size=1 → f32 output is shape (1, 8, 16),
# 512 bytes. Source values are ``(i + offset) mod 2**16``; the assemble
# kernel reorders elements within each inner chunk, so we don't predict
# the exact layout — we check membership against the source value set
# instead. This is enough to distinguish "buffer carries zarr A" from
# "buffer carries zarr B" since the offsets are far apart.
_N_ELEMS = 8 * 16
_BATCH_BYTES = _N_ELEMS * 4  # f32


def _expected_value_set(offset: int) -> set[int]:
    mod = 1 << 16
    return {(i + offset) % mod for i in range(_N_ELEMS)}


def _read_pinned_f32(ptr: int, n_elems: int) -> list[float]:
    raw = ctypes.string_at(ptr, n_elems * 4)
    return list(struct.unpack(f"<{n_elems}f", raw))


def _matches_zarr(values: list[float], offset: int) -> bool:
    return {int(v) for v in values} == _expected_value_set(offset)


# ---- the smoke test ----------------------------------------------------


def test_release_with_event_accepts_int_handle(two_zarrs):
    """Smoke: deferred release with a raw CUevent int handle returns
    without raising, and subsequent pops still work."""
    a, b = two_zarrs
    cfg = Config(
        batch_size=1,
        dtype="f32",
        lookahead_batches=2,
        n_io_threads=1,
        n_zarrs_meta_cache=4,
        n_shards_meta_cache=4,
    )
    with Pipeline(cfg) as d:
        d.push([Sample(uri=a, aabb=[(0, 8), (0, 16)])])
        d.push([Sample(uri=b, aabb=[(0, 8), (0, 16)])])

        side = stream_create()
        try:
            host_ptr, _buf = mem_alloc_host(_BATCH_BYTES)
            try:
                # First batch.
                batch1 = d.pop()
                memcpy_dtoh_async(host_ptr, batch1.info.device_ptr, _BATCH_BYTES, side)
                ev = event_create()
                event_record(ev, side)
                batch1.release(event=ev)
                event_destroy(ev)

                # Second batch — damacy should not stomp host_ptr's source
                # buffer before the side-stream copy retires.
                with d.pop() as batch2:
                    assert batch2.info.batch_id == 1

                # Sync the side stream; host_ptr should hold batch1 (zarr A,
                # values 0..127), not batch2 (zarr B, values 10000..10127).
                stream_sync(side)
                got = _read_pinned_f32(host_ptr, _N_ELEMS)
                assert _matches_zarr(got, offset=0), (
                    f"captured buffer doesn't match zarr A's value set; "
                    f"first 8 values: {got[:8]}"
                )
            finally:
                mem_free_host(host_ptr)
        finally:
            stream_destroy(side)


def test_release_with_event_no_stomp_across_many_iterations(two_zarrs):
    """Stress the deferred-release path: alternate zarrs (a, b, a, b, …)
    and verify every captured batch matches its expected content."""
    a, b = two_zarrs
    cfg = Config(
        batch_size=1,
        dtype="f32",
        lookahead_batches=2,
        n_io_threads=1,
        n_zarrs_meta_cache=4,
        n_shards_meta_cache=4,
    )
    n_iter = 8
    samples = [
        Sample(uri=a if i % 2 == 0 else b, aabb=[(0, 8), (0, 16)])
        for i in range(n_iter)
    ]
    expected_offsets = [0 if i % 2 == 0 else 10_000 for i in range(n_iter)]

    with Pipeline(cfg) as d:
        d.push(samples)
        side = stream_create()
        host_ptrs: list[tuple[int, object]] = [
            mem_alloc_host(_BATCH_BYTES) for _ in range(n_iter)
        ]
        try:
            for i in range(n_iter):
                batch = d.pop()
                memcpy_dtoh_async(
                    host_ptrs[i][0], batch.info.device_ptr, _BATCH_BYTES, side
                )
                ev = event_create()
                event_record(ev, side)
                batch.release(event=ev)
                event_destroy(ev)

            stream_sync(side)
            for i, off in enumerate(expected_offsets):
                got = _read_pinned_f32(host_ptrs[i][0], _N_ELEMS)
                assert _matches_zarr(got, offset=off), (
                    f"iter {i}: captured buffer doesn't match offset={off}; "
                    f"first 8: {got[:8]}"
                )
        finally:
            for ptr, _ in host_ptrs:
                mem_free_host(ptr)
            stream_destroy(side)


def test_release_event_none_falls_back_to_immediate(two_zarrs):
    """``release(event=None)`` is the same as ``release()``."""
    a, _ = two_zarrs
    cfg = Config(
        batch_size=1,
        dtype="f32",
        lookahead_batches=2,
        n_io_threads=1,
        n_zarrs_meta_cache=4,
        n_shards_meta_cache=4,
    )
    with Pipeline(cfg) as d:
        d.push([Sample(uri=a, aabb=[(0, 8), (0, 16)])])
        b = d.pop()
        b.release(event=None)
        b.release()  # idempotent


def test_release_event_rejects_bad_type(two_zarrs):
    a, _ = two_zarrs
    cfg = Config(
        batch_size=1,
        dtype="f32",
        lookahead_batches=2,
        n_io_threads=1,
        n_zarrs_meta_cache=4,
        n_shards_meta_cache=4,
    )
    with Pipeline(cfg) as d:
        d.push([Sample(uri=a, aabb=[(0, 8), (0, 16)])])
        with d.pop() as batch, pytest.raises(TypeError, match="event must be"):
            # bytes is neither int nor a stream/event-like
            batch.release(event=b"nope")  # type: ignore[arg-type]


def test_coerce_cuda_event_handle_accepts_ints():
    """Unit-level: the resolver passes ints straight through."""
    assert damacy._coerce_cuda_event_handle(None) is None
    assert damacy._coerce_cuda_event_handle(0xDEADBEEF) == 0xDEADBEEF


def test_coerce_cuda_event_handle_reads_event_like():
    """Anything with a ``cuda_event`` attribute is unwrapped."""

    class FakeEvent:
        cuda_event = 42

    assert damacy._coerce_cuda_event_handle(FakeEvent()) == 42


def test_coerce_cuda_event_handle_records_stream_like():
    """Stream-like objects: ``record_event()`` is called; the result is
    re-coerced."""

    class FakeEvent:
        cuda_event = 1234

    class FakeStream:
        def record_event(self):
            return FakeEvent()

    assert damacy._coerce_cuda_event_handle(FakeStream()) == 1234


def test_coerce_cuda_event_handle_rejects_unknown():
    with pytest.raises(TypeError, match="event must be"):
        damacy._coerce_cuda_event_handle(object())
