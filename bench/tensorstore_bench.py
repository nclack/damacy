#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2", "tensorstore"]
# ///
"""CPU read+decode benchmark for damacy bench scenarios, via tensorstore.

Reads the same scenario JSON that bench/run.py feeds damacy, resolves the same
array directories, and samples the same TCZYX patches, so tensorstore's CPU
read+decode throughput is directly comparable to damacy's GPU numbers on
identical data, chunking, and patch sampling.

Both scenario modes are handled the way bench/run.py resolves them:
  synthetic (dataset.uris is None): arrays at
    store_root / format_subdir(zarr_subdir_fmt(uri_fmt, array_path), i) / array_path
  uris mode: store_root / uri for each uri.

Patch sampling mirrors bench/main.c: an xorshift64* RNG seeded from
sampling.seed draws the array index first, then a start per axis in
[0, dim - patch_dim]. Warmup draws come off the same stream before the timed
draws, so with the same seed the sampled positions line up with damacy's.

  # login/CPU node smoke test (tiny, warm cache):
  uv run bench/tensorstore_bench.py bench/scenarios/smoke.json --threads 1,2

  # full cold-cache concurrency sweep (compute node via srun):
  uv run bench/tensorstore_bench.py bench/scenarios/throughput.json --drop-cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import tensorstore as ts

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scenario import (  # noqa: E402
    Scenario,
    format_subdir,
    zarr_subdir_fmt,
)

DEFAULT_THREADS = [1, 2, 4, 8, 16, 32]


class Xorshift64Star:
    """Matches bench/main.c's rng: same sequence given the same seed, so the
    sampled patch positions line up with damacy's."""

    def __init__(self, seed: int) -> None:
        self.s = seed if seed else 0xDEADBEEF

    def next(self) -> int:
        x = self.s & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 12
        x = (x ^ (x << 25)) & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 27
        self.s = x
        return (x * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF

    def range(self, n: int) -> int:
        return self.next() % n if n else 0


def resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


def array_dirs(sc: Scenario) -> list[Path]:
    """Absolute dir of each array the scenario references, resolved the same
    way bench/run.py does for both scenario modes."""
    ds = sc.dataset
    store_root = resolve_path(ds.store_root)
    if ds.uris is not None:
        return [store_root / u for u in ds.uris]
    assert ds.uri_fmt is not None
    sub_fmt = zarr_subdir_fmt(ds.uri_fmt, ds.array_path)
    return [
        store_root / format_subdir(sub_fmt, i) / ds.array_path
        for i in range(ds.n_zarrs)
    ]


def concurrency_context(limit: int) -> ts.Context:
    return ts.Context(
        {
            "data_copy_concurrency": {"limit": limit},
            "file_io_concurrency": {"limit": limit},
        }
    )


def open_array(path: Path, context: ts.Context | None = None):
    return ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(path)},
        },
        context=context,
    ).result()


def open_arrays(dirs: list[Path], rank: int, sample_shape: list[int],
                context: ts.Context | None = None):
    """Open each dir, keep those of matching rank and large enough for the
    patch on every axis. Mirrors damacy's filtering so the array count feeding
    the RNG lines up. Returns (paths, arrays, shapes); reports the skip count."""
    paths, arrays, shapes = [], [], []
    skipped = 0
    for d in dirs:
        arr = open_array(d, context)
        shape = tuple(arr.shape)
        ok = len(shape) == rank and all(
            shape[k] >= sample_shape[k] for k in range(rank)
        )
        if not ok:
            skipped += 1
            continue
        paths.append(d)
        arrays.append(arr)
        shapes.append(shape)
    if skipped:
        print(f"skipped {skipped} array(s) of wrong rank or too small for the patch")
    return paths, arrays, shapes


def sample_positions(rng: Xorshift64Star, shapes: list[tuple],
                     sample_shape: list[int], n: int) -> list[tuple[int, tuple]]:
    """Draw n (array_index, start_per_axis) the same way bench/main.c does:
    array first, then a start per axis in [0, dim - patch_dim]."""
    rank = len(sample_shape)
    out = []
    for _ in range(n):
        z = rng.range(len(shapes))
        shape = shapes[z]
        begs = tuple(rng.range(shape[d] - sample_shape[d] + 1) for d in range(rank))
        out.append((z, begs))
    return out


def read_patches(arrays: list, positions: list[tuple[int, tuple]],
                 sample_shape: list[int], threads: int) -> None:
    """Issue patch reads with up to `threads` reads in flight at once."""
    rank = len(sample_shape)

    def submit(z: int, begs: tuple):
        arr = arrays[z]
        idx = tuple(slice(begs[d], begs[d] + sample_shape[d]) for d in range(rank))
        return arr[idx].read()

    if threads <= 1:
        for z, begs in positions:
            submit(z, begs).result()
        return

    in_flight: list = []
    i = 0
    while i < len(positions) or in_flight:
        while i < len(positions) and len(in_flight) < threads:
            z, begs = positions[i]
            in_flight.append(submit(z, begs))
            i += 1
        in_flight.pop(0).result()


def patch_bytes(arrays: list, sample_shape: list[int]) -> int:
    n = arrays[0].dtype.numpy_dtype.itemsize
    for d in sample_shape:
        n *= d
    return n


def drop_page_cache(roots: list[Path]) -> tuple[int, int]:
    """posix_fadvise(DONTNEED) every regular file under each root, mirroring
    bench/run.py's drop_page_cache. Best-effort; returns (n_files, n_bytes)."""
    if not hasattr(os, "posix_fadvise"):
        print("warn: posix_fadvise unavailable; cache not dropped")
        return (0, 0)
    os.sync()
    n, sz = 0, 0
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            try:
                fd = os.open(str(p), os.O_RDONLY)
            except OSError:
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
    return n, sz


def run_one(arrays: list, positions: list[tuple[int, tuple]],
            sample_shape: list[int], threads: int, pbytes: int,
            warmup: list[tuple[int, tuple]]) -> dict:
    if warmup:
        read_patches(arrays, warmup, sample_shape, threads)
    t0 = time.perf_counter()
    read_patches(arrays, positions, sample_shape, threads)
    wall = time.perf_counter() - t0
    n = len(positions)
    return {
        "threads": threads,
        "samples": n,
        "wall_s": wall,
        "samples_s": n / wall if wall else 0.0,
        "gb_s": (n * pbytes) / wall / 1e9 if wall else 0.0,
    }


def summarize(name: str, rows: list[dict], pbytes: int, n_arrays: int,
              sample_shape: list[int], best: dict | None) -> str:
    cols = [("threads", 8), ("samp/s", 10), ("GB/s", 8), ("wall_s", 8)]
    head = "  ".join(f"{c:<{w}}" for c, w in cols)
    lines = [
        f"scenario: {name}   arrays: {n_arrays}   patch: "
        f"{','.join(str(x) for x in sample_shape)}   "
        f"patch_bytes: {pbytes}",
        f"best: {best.get('threads', '') if best else ''} threads  "
        f"{(best or {}).get('samples_s', 0):.1f} samp/s  "
        f"{(best or {}).get('gb_s', 0):.3f} GB/s",
        "",
        head,
        "  ".join("-" * w for _, w in cols),
    ]
    for r in rows:
        vals = [str(r["threads"]), f"{r['samples_s']:.1f}",
                f"{r['gb_s']:.3f}", f"{r['wall_s']:.2f}"]
        lines.append("  ".join(f"{v:<{w}}" for v, (_, w) in zip(vals, cols)))
    return "\n".join(lines)


def read_damacy_throughput(path: Path) -> dict:
    """Pull GB/s + samp/s out of a damacy bench results.json for side-by-side."""
    r = json.loads(path.read_text())
    derived = r.get("derived", {})
    timings = r.get("timings_ms", {})
    counters = r.get("counters", {})
    gb_s = derived.get("throughput_mb_s", 0.0) / 1000.0
    wall_s = timings.get("wall", 0.0) / 1000.0
    pushed = counters.get("samples_pushed", 0)
    return {
        "gb_s": gb_s,
        "samples_s": pushed / wall_s if wall_s else 0.0,
        "samples": pushed,
        "wall_s": wall_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("scenario", help="damacy scenario JSON (path)")
    ap.add_argument("--threads", default=",".join(str(t) for t in DEFAULT_THREADS),
                    help="comma-separated concurrency sweep (default 1,2,4,8,16,32)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap arrays opened/sampled (for a tiny smoke test)")
    ap.add_argument("--n-batches", type=int, default=None,
                    help="override sampling.n_batches (smoke test)")
    ap.add_argument("--samples-per-batch", type=int, default=None,
                    help="override sampling.samples_per_batch (smoke test)")
    ap.add_argument("--drop-cache", action="store_true",
                    help="posix_fadvise(DONTNEED) the sampled files before each "
                    "thread-count run (cold cache; default off)")
    ap.add_argument("--compare-with", default=None,
                    help="a damacy bench results.json; prints a side-by-side line "
                    "of damacy GPU vs tensorstore best-thread throughput")
    ap.add_argument("--out", default=None,
                    help="JSON summary path (default: <scenario>.tensorstore.json)")
    a = ap.parse_args()

    scenario_path = Path(a.scenario)
    sc = Scenario.model_validate_json(scenario_path.read_text())
    name = sc.name
    samp = sc.sampling
    sample_shape = list(samp.sample_shape)
    rank = len(sample_shape)
    n_batches = a.n_batches if a.n_batches is not None else samp.n_batches
    spb = (a.samples_per_batch if a.samples_per_batch is not None
           else samp.samples_per_batch)
    n_warmup = samp.n_warmup_batches
    seed = samp.seed
    thread_counts = [int(t) for t in a.threads.split(",") if t.strip()]

    dirs = array_dirs(sc)
    if a.limit is not None:
        dirs = dirs[:a.limit]

    paths, arrays, shapes = open_arrays(dirs, rank, sample_shape)
    if not arrays:
        raise SystemExit("no arrays matched the patch rank/shape")
    pbytes = patch_bytes(arrays, sample_shape)

    n_samples = n_batches * spb
    n_warmup_samples = n_warmup * spb
    print(f"opened {len(arrays)} array(s); {n_warmup_samples} warmup + "
          f"{n_samples} timed patches of {pbytes} bytes each")

    rows: list[dict] = []
    for threads in thread_counts:
        ctx = concurrency_context(threads)
        run_arrays = [open_array(p, ctx) for p in paths]
        rng = Xorshift64Star(seed)
        warmup = sample_positions(rng, shapes, sample_shape, n_warmup_samples)
        positions = sample_positions(rng, shapes, sample_shape, n_samples)
        if a.drop_cache:
            n, sz = drop_page_cache(paths)
            print(f"page cache dropped: {n} files, {sz / 1e9:.2f} GB hinted")
        rows.append(run_one(run_arrays, positions, sample_shape, threads, pbytes, warmup))

    best = max(rows, key=lambda r: r["gb_s"]) if rows else None
    table = summarize(name, rows, pbytes, len(arrays), sample_shape, best)
    print("\n" + table + "\n")

    compare = None
    if a.compare_with and best is not None:
        damacy = read_damacy_throughput(Path(a.compare_with))
        compare = {"damacy": damacy, "tensorstore_best": best}
        print(
            f"head-to-head ({name}): "
            f"damacy GPU {damacy['gb_s']:.3f} GB/s "
            f"({damacy['samples_s']:.1f} samp/s)  vs  "
            f"tensorstore CPU best {best['gb_s']:.3f} GB/s "
            f"({best['samples_s']:.1f} samp/s, {best['threads']} threads)\n"
        )

    out_path = Path(a.out) if a.out else scenario_path.with_suffix(".tensorstore.json")
    summary = {
        "backend": "tensorstore",
        "scenario": name,
        "scenario_path": str(scenario_path),
        "n_arrays": len(arrays),
        "sample_shape": sample_shape,
        "patch_bytes": pbytes,
        "n_warmup_samples": n_warmup_samples,
        "n_timed_samples": n_samples,
        "drop_cache": a.drop_cache,
        "rows": rows,
        "best": best,
        "compare": compare,
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
