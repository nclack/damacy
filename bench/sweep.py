#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["rich>=13", "typer>=0.12"]
# ///
"""Parameter sweep around bench/run.py.

Patches one dotted-path field in a base scenario, runs bench/run.py for
each value, and prints a comparison table.

  uv run bench/sweep.py bench/scenarios/default.json \\
      --param pipeline.n_io_threads --values 1,2,4,8,16,32
"""

from __future__ import annotations

import json
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


def patch(scenario: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    d = scenario
    for p in parts[:-1]:
        d = d[p]
    d[parts[-1]] = value


def parse_value(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def run_one(scenario_path: Path, runs_dir: Path) -> dict:
    runs_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "uv",
            "run",
            str(RUN_SCRIPT),
            str(scenario_path),
            "--runs-dir",
            str(runs_dir),
            "--no-report",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"run.py exited {proc.returncode}")
    # run.py writes <runs_dir>/<timestamp>/results.json; we own a fresh
    # runs_dir per iteration so exactly one results.json lands here.
    hits = sorted(runs_dir.glob("*/results.json"))
    if not hits:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"no results.json under {runs_dir}")
    return json.loads(hits[-1].read_text())


@app.command()
def main(
    base: Path = typer.Argument(..., help="base scenario JSON"),
    param: str = typer.Option(
        ..., help="dotted-path field, e.g. pipeline.n_io_threads"
    ),
    values: str = typer.Option(..., help="comma-separated values"),
) -> None:
    base_data = json.loads(base.read_text())
    parsed = [parse_value(v.strip()) for v in values.split(",")]

    results: list[tuple[object, dict]] = []
    with tempfile.TemporaryDirectory() as td:
        for i, v in enumerate(parsed):
            sc = json.loads(json.dumps(base_data))  # deep copy
            patch(sc, param, v)
            sc["name"] = f"{sc.get('name', 'sweep')}-{param.split('.')[-1]}{v}"
            tmp = Path(td) / f"sweep_{v}.json"
            tmp.write_text(json.dumps(sc, indent=2))
            console.print(f"[bold cyan]── {param} = {v} ──[/bold cyan]")
            data = run_one(tmp, Path(td) / f"runs_{i}")
            results.append((v, data))

    t = Table(title=f"sweep: {param}", title_style="bold cyan")
    t.add_column(param, justify="right")
    t.add_column("io GB_in", justify="right")
    t.add_column("io ms", justify="right")
    t.add_column("io GB/s", justify="right")
    t.add_column("input GB/s", justify="right")
    t.add_column("reads", justify="right")
    t.add_column("wall s", justify="right")
    t.add_column("throughput GB/s", justify="right")
    for v, d in results:
        stages = {s["name"]: s for s in d["stages"]}
        io = stages["io"]
        input_transfer = stages["input_transfer"]
        c = d["counters"]
        wall_ms = d["timings_ms"]["wall"]
        gb_in = io["input_bytes"] / 1e9
        io_gb_s = (gb_in * 1000.0) / io["ms_total"] if io["ms_total"] > 0 else 0.0
        input_gb_in = input_transfer["input_bytes"] / 1e9
        input_gb_s = (
            (input_gb_in * 1000.0) / input_transfer["ms_total"]
            if input_transfer["ms_total"] > 0
            else 0.0
        )
        t.add_row(
            str(v),
            f"{gb_in:.2f}",
            f"{io['ms_total']:.0f}",
            f"{io_gb_s:.2f}",
            f"{input_gb_s:.2f}",
            f"{c['reads_issued']:,}",
            f"{wall_ms / 1000.0:.2f}",
            f"{d['derived']['throughput_mb_s'] / 1e3:.2f}",
        )
    Console().print(t)


if __name__ == "__main__":
    app()
