#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2", "rich>=13", "typer>=0.12"]
# ///
"""Scenario-driven damacy_bench runner.

Reads a scenario JSON (validated via pydantic), ensures every zarr it
references exists (calls bench/gen_dataset.py per missing zarr with a
rich progress bar), drops the page cache, runs damacy_bench, archives
results under bench/runs/<scenario>/<ts>/, and prints a rich report.

  uv run bench/run.py bench/scenarios/default.json
  uv run bench/run.py bench/scenarios/default.json --regen
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import typer
from pydantic import ValidationError
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GEN_SCRIPT = REPO_ROOT / "bench" / "gen_dataset.py"
BENCH_BIN = REPO_ROOT / "build" / "bench" / "damacy_bench"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import make_console  # noqa: E402
from report import render as render_report
from scenario import (
    NUMPY_DTYPE,
    Results,
    Scenario,
    format_subdir,
    zarr_subdir_fmt,
)

DEFAULT_SCENARIO = REPO_ROOT / "bench" / "scenarios" / "default.json"

app = typer.Typer(add_completion=False, help=__doc__)
console = make_console(stderr=True)


def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


def gen_one_zarr(out: Path, sc: Scenario, seed: int, codec: str, dtype: str) -> int:
    ds = sc.dataset
    csv = lambda xs: ",".join(str(x) for x in xs)
    cmd = [
        "uv",
        "run",
        str(GEN_SCRIPT),
        "--out",
        str(out),
        "--shape",
        csv(ds.zarr_shape),
        "--inner",
        csv(ds.chunk_shape),
        "--shard",
        csv(ds.shard_shape),
        "--dtype",
        NUMPY_DTYPE[dtype],
        "--codec",
        codec,
        "--clevel",
        str(ds.clevel),
        "--entropy",
        str(ds.entropy),
        "--seed",
        str(seed),
    ]
    return subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode


def ensure_zarrs(sc: Scenario, regen: bool) -> None:
    ds = sc.dataset
    store_root = resolve_path(ds.store_root)
    sub_fmt = zarr_subdir_fmt(ds.uri_fmt, ds.array_path)

    pending: list[tuple[int, Path]] = []
    for i in range(ds.n_zarrs):
        zarr_root = store_root / format_subdir(sub_fmt, i)
        marker = zarr_root / ds.array_path / "zarr.json"
        if regen and zarr_root.exists():
            console.print(f"[yellow]rm -rf[/yellow] {zarr_root}")
            shutil.rmtree(zarr_root)
        if not marker.exists():
            pending.append((i, zarr_root))

    if not pending:
        console.print(
            f"[green]✓[/green] all {ds.n_zarrs} zarrs present at {store_root}"
        )
        return

    console.print(
        f"generating [bold]{len(pending)}[/bold] zarr(s) "
        f"under [cyan]{store_root}[/cyan]"
    )
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("gen_dataset", total=len(pending))
        for i, zarr_root in pending:
            zarr_root.parent.mkdir(parents=True, exist_ok=True)
            codec = ds.codec_for(i)
            dtype = ds.dtype_for(i)
            prog.update(
                task,
                description=(
                    f"[cyan]{zarr_root.name}[/cyan] [dim]({codec}, {dtype})[/dim]"
                ),
            )
            rc = gen_one_zarr(zarr_root, sc, ds.seed + i, codec, dtype)
            if rc != 0:
                raise typer.Exit(rc)
            prog.advance(task)


def drop_page_cache(roots: list[Path]) -> tuple[int, int]:
    """posix_fadvise(DONTNEED) every regular file under each root.
    Best-effort; returns (n_files, n_bytes_hinted)."""
    if not hasattr(os, "posix_fadvise"):
        console.print(
            "[yellow]warn[/yellow] posix_fadvise unavailable; cache not dropped"
        )
        return (0, 0)
    os.sync()
    n, sz = 0, 0
    files: list[Path] = []
    for root in roots:
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file())
    with Progress(
        SpinnerColumn(),
        TextColumn("dropping page cache"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as prog:
        task = prog.add_task("drop", total=len(files))
        for p in files:
            try:
                fd = os.open(str(p), os.O_RDONLY)
            except OSError:
                prog.advance(task)
                continue
            try:
                st = os.fstat(fd)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
                sz += st.st_size
            except OSError:
                pass
            finally:
                os.close(fd)
                prog.advance(task)
    return n, sz


def run_bench(scenario_path: Path) -> tuple[int, str]:
    if not BENCH_BIN.exists():
        raise typer.Exit(
            f"error: {BENCH_BIN} not found; run `cmake --build build` first"
        )
    console.print(f"[dim]+ {BENCH_BIN} {scenario_path}[/dim]")
    proc = subprocess.run(
        [str(BENCH_BIN), str(scenario_path)], stdout=subprocess.PIPE, text=True
    )
    return proc.returncode, proc.stdout


@app.command()
def main(
    scenario_path: Path = typer.Argument(
        DEFAULT_SCENARIO,
        metavar="SCENARIO",
        exists=True,
        dir_okay=False,
        readable=True,
        help=f"path to scenario.json (default: {DEFAULT_SCENARIO.relative_to(REPO_ROOT)})",
    ),
    regen: bool = typer.Option(
        False,
        "--regen",
        help="delete and regenerate every zarr in the scenario before running",
    ),
    runs_dir: Path | None = typer.Option(
        None,
        "--runs-dir",
        help="parent dir for the timestamped run output "
        "(default: bench/runs/<scenario_name>/<ts>)",
    ),
    no_report: bool = typer.Option(
        False, "--no-report", help="skip pretty-printing the report"
    ),
    warm: bool = typer.Option(
        False,
        "--warm",
        help="skip the page-cache drop before running "
        "(default: drop pages so the bench measures cold IO)",
    ),
):
    try:
        sc = Scenario.model_validate_json(scenario_path.read_text())
    except ValidationError as e:
        console.print(f"[red]invalid scenario {scenario_path}:[/red]")
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            console.print(f"  [yellow]{loc}[/yellow]: {err['msg']}")
        raise typer.Exit(1)

    ensure_zarrs(sc, regen)

    if not warm:
        store_root = resolve_path(sc.dataset.store_root)
        n, sz = drop_page_cache([store_root])
        console.print(
            f"[green]✓[/green] page cache dropped: {n} files, {sz / 1e9:.2f} GB hinted"
        )

    rc, stdout = run_bench(scenario_path)
    if rc != 0:
        console.print(f"[red]damacy_bench exited {rc}[/red]")
        raise typer.Exit(rc)
    if not stdout.strip():
        console.print("[red]damacy_bench produced no stdout[/red]")
        raise typer.Exit(1)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = (
        (runs_dir / ts) if runs_dir else (REPO_ROOT / "bench" / "runs" / sc.name / ts)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "results.json"
    results_path.write_text(stdout)
    (out_dir / "scenario.json").write_text(scenario_path.read_text())
    console.print(f"[green]results:[/green] {results_path}")

    if not no_report:
        results = Results.model_validate_json(stdout)
        render_report(results, out=make_console())


if __name__ == "__main__":
    app()
