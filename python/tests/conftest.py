# Pytest fixtures for damacy._native bindings tests.
#
# Reuses tests/write_zarr.py (a uv-run-script PEP 723 fixture writer)
# to materialise small NGFF zarr stores under tmp_path. The same script
# powers the C tests via tests/fixture.c, so the fixture surface is
# already battle-tested. WRITE_ZARR_SCRIPT comes from CMake (set on the
# python_pytest test); falls back to a repo-relative path when pytest
# is invoked outside ctest.
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _write_zarr_script() -> Path:
    env = os.environ.get("WRITE_ZARR_SCRIPT")
    if env:
        return Path(env)
    # Fall back to <repo_root>/tests/write_zarr.py when running ad-hoc.
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "tests" / "write_zarr.py"
        if cand.is_file():
            return cand
    raise RuntimeError("WRITE_ZARR_SCRIPT not set and tests/write_zarr.py not found")


def _have_uv() -> bool:
    return shutil.which("uv") is not None


@pytest.fixture(scope="session", autouse=True)
def _cuda_ctx() -> None:
    """Make device 0's primary CUcontext current for the test process.

    damacy_create requires a current CUcontext on the calling thread.
    PyTorch sets one up implicitly; bare pytest doesn't, so we do it
    once per session. Skips the suite if no CUDA driver is reachable.
    """
    from damacy import _native

    try:
        _native.cuda_init_primary()
    except RuntimeError as exc:
        pytest.skip(f"no CUDA available for binding tests: {exc}")


@pytest.fixture(scope="session")
def write_zarr_script() -> Path:
    return _write_zarr_script()


@pytest.fixture
def tiny_zarr(tmp_path: Path, write_zarr_script: Path) -> tuple[Path, str]:
    """A 2D u16 zarr (8x16, single 8x16 shard, 4x8 inner chunks, blosc-lz4).

    Returns (store_root, uri). uri is the directory name relative to root.
    """
    if not _have_uv():
        pytest.skip("uv not on PATH; needed to materialise the zarr fixture")
    out = tmp_path / "foo"
    cmd = [
        "uv",
        "run",
        "--script",
        str(write_zarr_script),
        "--out",
        str(out),
        "--shape",
        "8,16",
        "--inner",
        "4,8",
        "--shard",
        "8,16",
        "--dtype",
        "uint16",
        "--offset",
        "0",
        "--codec",
        "blosc-lz4",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        pytest.skip(f"write_zarr.py failed (rc={r.returncode}); skipping")
    return tmp_path, "foo"


@pytest.fixture
def tiny_zarr_no_cast(tmp_path: Path, write_zarr_script: Path) -> tuple[Path, str]:
    """Same shape as tiny_zarr but int64 — has no cast path to f32/bf16
    (post-#16: only u8/u16/i16/u32/i32/f16/f32 sources are supported).
    Used to drive the DAMACY_DTYPE error path."""
    if not _have_uv():
        pytest.skip("uv not on PATH; needed to materialise the zarr fixture")
    out = tmp_path / "bar"
    cmd = [
        "uv",
        "run",
        "--script",
        str(write_zarr_script),
        "--out",
        str(out),
        "--shape",
        "8,16",
        "--inner",
        "4,8",
        "--shard",
        "8,16",
        "--dtype",
        "int64",
        "--codec",
        "blosc-lz4",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        pytest.skip(f"write_zarr.py failed (rc={r.returncode}); skipping")
    return tmp_path, "bar"
