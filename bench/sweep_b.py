#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich>=13", "typer>=0.12"]
# ///
"""Sweep DAMACY_BATCH_SLOTS (the B in issue #79).

B is a compile-time symbol (src/damacy_limits.h), so each value gets its
own build directory. The script configures and builds build-bN once per
unique B, then drives the existing bench/run.py against the matching
binary by symlinking build -> build-bN (run.py looks for
build/bench/damacy_bench).

  uv run bench/sweep_b.py bench/scenarios/noisy-nfs.json --values 2,3,4
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "bench" / "run.py"

app = typer.Typer(add_completion=False, help=__doc__)
console = Console(stderr=True)


def build_for_b(b: int) -> Path:
    """Configure + build a damacy_bench with DAMACY_BATCH_SLOTS=b."""
    build_dir = REPO_ROOT / f"build-b{b}"
    cflags = f"-DDAMACY_BATCH_SLOTS={b}"
    console.print(f"[bold]configure[/bold] build-b{b} (CFLAGS={cflags})")
    rc = subprocess.run(
        [
            "cmake",
            "-B",
            str(build_dir),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
            f"-DCMAKE_C_FLAGS={cflags}",
            f"-DCMAKE_CXX_FLAGS={cflags}",
        ],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"cmake configure failed for B={b}")
    console.print(f"[bold]build[/bold] build-b{b}/bench/damacy_bench")
    rc = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "damacy_bench"],
        cwd=REPO_ROOT,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"cmake build failed for B={b}")
    return build_dir


def run_with_build(scenario_path: Path, build_dir: Path, runs_dir: Path) -> dict:
    runs_dir.mkdir(parents=True, exist_ok=True)
    bench_bin = build_dir / "bench" / "damacy_bench"
    env = os.environ.copy()
    env["DAMACY_BENCH_BIN"] = str(bench_bin)
    proc = subprocess.run(
        [
            "uv",
            "run",
            str(RUN_SCRIPT),
            str(scenario_path),
            "--runs-dir",
            str(runs_dir),
            "--no-report",
            "--warm",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"run.py exited {proc.returncode}")
    hits = sorted(runs_dir.glob("*/results.json"))
    if not hits:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"no results.json under {runs_dir}")
    return json.loads(hits[-1].read_text())


@app.command()
def main(
    scenario: Path = typer.Argument(..., help="scenario JSON"),
    values: str = typer.Option("2,3,4", help="comma-separated B values"),
) -> None:
    bs = [int(v.strip()) for v in values.split(",")]
    builds = {b: build_for_b(b) for b in sorted(set(bs))}

    results: list[tuple[int, dict]] = []
    with tempfile.TemporaryDirectory() as td:
        for i, b in enumerate(bs):
            console.print(f"[bold cyan]── B = {b} ──[/bold cyan]")
            data = run_with_build(scenario, builds[b], Path(td) / f"runs_{i}")
            results.append((b, data))

    t = Table(title=f"B sweep: {scenario.name}", title_style="bold cyan")
    t.add_column("B", justify="right")
    t.add_column("wall s", justify="right")
    t.add_column("ttfb ms", justify="right")
    t.add_column("throughput MB/s", justify="right")
    t.add_column("io GB/s", justify="right")
    t.add_column("consumer_block ms", justify="right")
    t.add_column("pop_wait ms_total", justify="right")
    t.add_column("gpu MB", justify="right")
    for b, d in results:
        stages = {s["name"]: s for s in d["stages"]}
        io = stages["io"]
        pop_wait = stages["pop_wait"]
        c = d["counters"]
        wall_ms = d["timings_ms"]["wall"]
        gb_in = io["input_bytes"] / 1e9
        io_gb_s = (gb_in * 1000.0) / io["ms_total"] if io["ms_total"] > 0 else 0.0
        t.add_row(
            str(b),
            f"{wall_ms / 1000.0:.2f}",
            f"{d['timings_ms']['time_to_first_batch']:.0f}",
            f"{d['derived']['throughput_mb_s']:.1f}",
            f"{io_gb_s:.2f}",
            f"{d['timings_ms']['consumer_block']:.0f}",
            f"{pop_wait['ms_total']:.0f}",
            f"{c['gpu_bytes_committed'] / 1e6:.0f}",
        )
    Console().print(t)


if __name__ == "__main__":
    app()
