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
    title = Text.assemble(
        ("pipeline stages  ", "bold cyan"),
        ("[", "dim"),
        ("italic", "italic dim"),
        ("=cpu ", "dim"),
        ("bold", "bold dim"),
        ("=gpu", "dim"),
        ("]", "dim"),
    )
    t = Table(
        title=title,
        show_lines=False,
        header_style="bold",
    )
    t.add_column("stage", style="cyan", no_wrap=True)
    t.add_column("unit", style="dim", no_wrap=True)
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
        "input_transfer": "magenta",
        "decode": "green",
        "post_decode": "green",
        "decode_gap": "red",
        "decompress.parse": "green",
        "assemble": "green",
        "pop_wait": "red",
        "flush_wait": "red",
    }

    # encode where each stage's time is spent in the unit column:
    # italic = CPU wall time, bold = GPU event time, dim = wait/stall.
    unit_style = {
        "plan": "italic dim",
        "io": "italic dim",
        "input_transfer": "bold dim",
        "decompress.parse": "italic dim",
        "decode": "bold dim",
        "post_decode": "bold dim",
        "decode_gap": "dim",
        "assemble": "bold dim",
        "pop_wait": "dim",
        "flush_wait": "dim",
    }

    for s in r.stages:
        secs = s.ms_total / 1e3
        c = color.get(s.name, "white")
        u = unit_style.get(s.name, "dim")
        t.add_row(
            Text(s.name, style=c),
            Text(s.unit, style=u),
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
        ("chunks_planned", f"{c.chunks_planned:,}"),
        (
            "filter→fuse",
            f"{c.chunks_planned:,} → {c.chunks_to_load:,} → {c.reads_issued:,}",
        ),
        ("distinct_zarrs", f"{c.distinct_zarrs}"),
        ("distinct_shards", f"{c.distinct_shards}"),
        ("array_meta hits/misses", f"{c.array_meta_hits:,} / {c.array_meta_misses:,}"),
        (
            "shard_index hits/misses",
            f"{c.shard_index_hits:,} / {c.shard_index_misses:,}",
        ),
        (
            "chunk_layout hits/misses",
            f"{c.chunk_layout_hits:,} / {c.chunk_layout_misses:,}",
        ),
        (
            "metadata latency (injected) ops",
            f"{c.metadata_latency_ops:,} "
            f"[dim](stat/submit "
            f"{c.metadata_latency_stat_ops:,}/"
            f"{c.metadata_latency_submit_ops:,})[/dim]",
        ),
        (
            "metadata latency (injected) concurrency",
            f"max={c.metadata_latency_max_active:,} "
            f"active={c.metadata_latency_active:,}",
        ),
        (
            "metadata latency (injected) sleep",
            f"total={c.metadata_latency_total_sleep_ns / 1e9:,.2f}s "
            f"max={c.metadata_latency_max_sleep_ns / 1e9:,.2f}s",
        ),
        (
            "metadata backend reads",
            f"jobs={c.metadata_backend_read_jobs:,} "
            f"max_active={c.metadata_backend_read_max_active:,} "
            f"active={c.metadata_backend_read_active:,}",
        ),
        ("gpu_bytes_committed", f"{c.gpu_bytes_committed / 1e6:,.1f} MB"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    return t


def _fmt_lat_ns(ns: float) -> str:
    if ns <= 0:
        return "-"
    if ns >= 1e9:
        return f"{ns / 1e9:.2f} s"
    if ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.2f} us"
    return f"{ns:.0f} ns"


def _meta_op_latency_table(r: Results) -> Table | None:
    ops = [o for o in r.counters.metadata_op_latency if o.count]
    if not ops:
        return None
    t = Table(
        title="metadata op latency (measured)",
        title_style="bold cyan",
        show_lines=False,
        header_style="bold",
        padding=(0, 2),
    )
    t.add_column("op", style="cyan", no_wrap=True)
    t.add_column("count", justify="right")
    t.add_column("avg", justify="right")
    t.add_column("p50", justify="right")
    t.add_column("p90", justify="right")
    t.add_column("p99", justify="right")
    t.add_column("max", justify="right")
    for o in ops:
        t.add_row(
            o.op,
            f"{o.count:,}",
            _fmt_lat_ns(o.avg_ns()),
            _fmt_lat_ns(o.percentile_ns(0.50)),
            _fmt_lat_ns(o.percentile_ns(0.90)),
            _fmt_lat_ns(o.percentile_ns(0.99)),
            _fmt_lat_ns(float(o.max_ns)),
        )
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
    hold_ms = r.scenario.consumer.hold_ms
    n_batches = r.scenario.sampling.n_batches
    block_per_batch = tm.consumer_block / n_batches if n_batches > 0 else 0.0
    push_per_batch = tm.consumer_push / n_batches if n_batches > 0 else 0.0
    pop_wait_per_batch = tm.consumer_pop_wait / n_batches if n_batches > 0 else 0.0
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
        (
            "consumer hold/batch",
            f"{hold_ms:.1f} ms"
            if hold_ms > 0
            else "[dim]0 (no backpressure sim)[/dim]",
        ),
        (
            "consumer_block/batch",
            f"{_fmt_ms(block_per_batch)} ms  [dim](wait for next batch)[/dim]",
        ),
        (
            "  push/batch",
            f"{_fmt_ms(push_per_batch)} ms  [dim](inside damacy_push)[/dim]",
        ),
        (
            "  pop_wait/batch",
            f"{_fmt_ms(pop_wait_per_batch)} ms  [dim](inside damacy_pop)[/dim]",
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
    lat = sc.metadata_latency
    latency_line = ""
    if (
        lat.baseline_ns
        or lat.lognormal_mu_ln_ns != 0.0
        or lat.lognormal_sigma_ln_ns != 0.0
    ):
        latency_line = (
            f"\n[dim]metadata latency[/dim] baseline_ns={lat.baseline_ns} "
            f"lognormal_mu_ln_ns={lat.lognormal_mu_ln_ns:g} "
            f"lognormal_sigma_ln_ns={lat.lognormal_sigma_ln_ns:g} "
            f"cap_ns={lat.cap_ns} seed={lat.seed}"
        )
    ds = sc.dataset
    if ds.uris is not None:
        dataset_line = (
            f"[dim]dataset[/dim]  uris={len(ds.uris)} real arrays  "
            f"store_root={ds.store_root}"
        )
    else:
        dataset_line = (
            f"[dim]dataset[/dim]  n_zarrs={ds.n_zarrs} "
            f"shape={ds.zarr_shape} "
            f"chunk={ds.chunk_shape} shard={ds.shard_shape} "
            f"src_dtypes={ds.dtypes}"
        )
    read_op_line = (
        f" max_read_op_kb={sc.pipeline.max_read_op_kb}"
        if sc.pipeline.max_read_op_kb is not None
        else ""
    )
    body = (
        f"[bold]{sc.name}[/bold]\n"
        f"{dataset_line}\n"
        f"[dim]sampling[/dim] n_batches={sc.sampling.n_batches} "
        f"(warmup={sc.sampling.n_warmup_batches}) "
        f"samples_per_batch={sc.sampling.samples_per_batch} "
        f"sample_shape={sc.sampling.sample_shape}\n"
        f"[dim]pipeline[/dim] dst_dtype={sc.pipeline.dtype} "
        f"lookahead={sc.pipeline.lookahead_samples} "
        f"io_threads={sc.pipeline.n_io_threads} "
        f"metadata_io_concurrency={sc.pipeline.metadata_io_concurrency} "
        f"max_gpu_mb={sc.pipeline.max_gpu_memory_mb or 'default'}"
        f"{read_op_line}"
        f"{latency_line}"
    )
    return Panel(body, title="scenario", border_style="cyan")


def render(r: Results, out: Console | None = None) -> None:
    out = out or console
    out.print(_scenario_panel(r))
    out.print(_stage_table(r))
    out.print(_summary_table(r))
    out.print(_counters_table(r))
    meta = _meta_op_latency_table(r)
    if meta is not None:
        out.print(meta)


@app.command()
def main(
    path: str = typer.Argument(..., help="results.json path, or '-' for stdin"),
):
    src = sys.stdin.read() if path == "-" else Path(path).read_text()
    r = Results.model_validate_json(src)
    render(r)


if __name__ == "__main__":
    app()
