#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2", "rich>=13", "typer>=0.12"]
# ///
"""Pretty-print a damacy_bench results.json.

uv run bench/report.py results.json
cat results.json | uv run bench/report.py -
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenario import Results

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


def make_console(stderr: bool = False) -> Console:
    # Force a wide width when stdout/stderr isn't a tty so piping the
    # report (e.g. through `tail` or into a file) doesn't squash the
    # stages table to single-character columns.
    c = Console(stderr=stderr)
    if not c.is_terminal:
        c = Console(stderr=stderr, width=140, force_terminal=False)
    return c


console = make_console()


def _fmt_gb(b: float) -> str:
    return f"{b / 1e9:.2f}" if b > 0 else "-"


def _fmt_gbps(b: float, secs: float) -> str:
    if b <= 0 or secs <= 0:
        return "-"
    return f"{(b / 1e9) / secs:.2f}"


def _fmt_ms(ms: float) -> str:
    if ms >= 10000:
        return f"{ms / 1000:.1f} s"
    if ms >= 100:
        return f"{ms:.0f}"
    return f"{ms:.2f}"


def _stage_table(r: Results) -> Table:
    t = Table(
        title="pipeline stages",
        title_style="bold cyan",
        show_lines=False,
        header_style="bold",
    )
    t.add_column("stage", style="cyan", no_wrap=True)
    t.add_column("unit", style="dim")
    t.add_column("GB/s_in", justify="right")
    t.add_column("ms_total", justify="right")
    t.add_column("ms_avg", justify="right")
    t.add_column("ms_best", justify="right")
    t.add_column("count", justify="right")
    t.add_column("GB_in", justify="right")
    t.add_column("GB_out", justify="right")

    # color stages by responsibility
    color = {
        "plan": "white",
        "io": "yellow",
        "h2d": "magenta",
        "decompress": "green",
        "decompress.parse": "green",
        "assemble": "green",
        "pop_wait_io": "red",
        "pop_wait_compute": "red",
        "flush_wait": "red",
    }

    for s in r.stages:
        secs = s.ms_total / 1e3
        c = color.get(s.name, "white")
        t.add_row(
            Text(s.name, style=c),
            s.unit,
            _fmt_gbps(s.input_bytes, secs),
            _fmt_ms(s.ms_total),
            _fmt_ms(s.ms_avg) if s.count else "-",
            _fmt_ms(s.ms_best) if s.count else "-",
            f"{s.count:,}",
            _fmt_gb(s.input_bytes),
            _fmt_gb(s.output_bytes),
        )
    return t


def _counters_table(r: Results) -> Table:
    t = Table(
        title="counters",
        title_style="bold cyan",
        show_header=False,
        show_lines=False,
        box=None,
        padding=(0, 2),
    )
    t.add_column(style="dim")
    t.add_column(justify="right")
    c = r.counters
    rows = [
        ("samples_pushed", f"{c.samples_pushed:,}"),
        ("batches_emitted", f"{c.batches_emitted:,}"),
        ("batches_truncated", f"{c.batches_truncated:,}"),
        ("waves_emitted", f"{c.waves_emitted:,}"),
        ("chunks_dispatched", f"{c.chunks_dispatched:,}"),
        ("distinct_zarrs", f"{c.distinct_zarrs}"),
        ("distinct_shards", f"{c.distinct_shards}"),
        ("zarr_meta hits/misses", f"{c.zarr_meta_hits:,} / {c.zarr_meta_misses:,}"),
        ("shard_idx hits/misses", f"{c.shard_idx_hits:,} / {c.shard_idx_misses:,}"),
        ("gpu_bytes_committed", f"{c.gpu_bytes_committed / 1e6:,.1f} MB"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    return t


def _summary_table(r: Results) -> Table:
    t = Table(
        title="summary",
        title_style="bold cyan",
        show_header=False,
        show_lines=False,
        box=None,
        padding=(0, 2),
    )
    t.add_column(style="dim")
    t.add_column(justify="right")
    tm = r.timings_ms
    d = r.derived
    rows = [
        ("init", _fmt_ms(tm.init) + " ms"),
        ("time_to_first_batch", _fmt_ms(tm.time_to_first_batch) + " ms"),
        ("wall (steady-state)", f"{tm.wall / 1e3:.2f} s"),
        (
            "throughput",
            f"{d.throughput_mb_s / 1e3:.2f} GB/s  [dim](sample volume / wall)[/dim]",
        ),
        (
            "stage_concurrency",
            f"{d.stage_concurrency:.2f}  [dim](sum stage ms / wall ms)[/dim]",
        ),
        ("chunks/batch", f"{d.chunks_per_batch:.0f}"),
        ("chunks/wave", f"{d.chunks_per_wave:.0f}"),
        ("bytes/sample", f"{d.bytes_per_sample / 1e6:.2f} MB"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    return t


def _scenario_panel(r: Results) -> Panel:
    sc = r.scenario
    body = (
        f"[bold]{sc.name}[/bold]\n"
        f"[dim]dataset[/dim]  n_zarrs={sc.dataset.n_zarrs} "
        f"shape={sc.dataset.zarr_shape} "
        f"chunk={sc.dataset.chunk_shape} shard={sc.dataset.shard_shape} "
        f"src_dtypes={sc.dataset.dtypes}\n"
        f"[dim]sampling[/dim] n_batches={sc.sampling.n_batches} "
        f"(warmup={sc.sampling.n_warmup_batches}) "
        f"batch_size={sc.sampling.batch_size} "
        f"sample_shape={sc.sampling.sample_shape}\n"
        f"[dim]pipeline[/dim] dst_dtype={sc.pipeline.dtype} "
        f"lookahead={sc.pipeline.lookahead_batches} "
        f"io_threads={sc.pipeline.n_io_threads} "
        f"max_gpu_mb={sc.pipeline.max_gpu_memory_mb or 'default'}"
    )
    return Panel(body, title="scenario", border_style="cyan")


def render(r: Results, out: Console | None = None) -> None:
    out = out or console
    out.print(_scenario_panel(r))
    out.print(_stage_table(r))
    out.print(_summary_table(r))
    out.print(_counters_table(r))


@app.command()
def main(
    path: str = typer.Argument(..., help="results.json path, or '-' for stdin"),
):
    src = sys.stdin.read() if path == "-" else Path(path).read_text()
    r = Results.model_validate_json(src)
    render(r)


if __name__ == "__main__":
    app()
