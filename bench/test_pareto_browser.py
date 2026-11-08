# /// script
# requires-python = ">=3.11"
# dependencies = ["playwright==1.62.0"]
# ///
"""Check plotted values, filtering, exports, and layout in Chromium."""

import argparse
import copy
import csv
import io
import json
import math
import os
import shutil
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from playwright.sync_api import sync_playwright


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def end_headers(self):
        if self.path.split("?", 1)[0] == "/pareto/data/index.json":
            self.send_header("Cache-Control", "max-age=600")
        super().end_headers()


def verify_study_refresh(browser, url, index_path):
    original = index_path.read_text()
    index = json.loads(original)
    previous = {**index, "studies": index["studies"][1:]}
    modified = index_path.stat().st_mtime - 60
    context = browser.new_context()
    try:
        index_path.write_text(json.dumps(previous))
        os.utime(index_path, (modified, modified))
        page = context.new_page()
        page.goto(url + f"#study={previous['studies'][0]['id']}")
        page.wait_for_selector("#workspace", state="visible")
        assert page.locator("#study").input_value() == previous["studies"][0]["id"]

        index_path.write_text(original)
        os.utime(index_path, (modified + 2, modified + 2))
        cached = page.evaluate("async () => (await fetch('data/index.json')).json()")
        assert cached == previous

        page.goto("about:blank")
        latest = index["studies"][0]["id"]
        page.goto(url + f"?revisit=1#study={latest}")
        page.wait_for_selector("#workspace", state="visible")
        assert page.locator("#study option").evaluate_all(
            "options => options.map(option => option.value)"
        ) == [entry["id"] for entry in index["studies"]]
        assert page.locator("#study").input_value() == latest
        assert page.locator("circle.point").count() > 0
    finally:
        index_path.write_text(original)
        context.close()


def fixture():
    study = {
        "version": 1,
        "id": "0" * 20,
        "label": "Browser fixture — not measured · long machine description · 2000-01-01",
        "pilot": True,
        "context": {
            "machine": {
                "host": "test fixture",
                "cpu_affinity": list(range(8)),
                "gpu": "Test GPU",
                "local_storage": {"devices": [{"model": "Test NVMe"}]},
            },
            "storage": "Test files",
            "cache_policy": "cold",
            "source": {"revision": "1" * 40},
            "binary_sha256": "2" * 64,
        },
        "pipeline": {"dtype": "f32", "decode_workers": 8, "n_io_threads": 8},
        "inputs": [
            {
                "id": name,
                "name": name,
                "source": {
                    "url": "https://example.invalid/",
                    "attribution": "Test fixture",
                    "license": "Test fixture",
                },
            }
            for name in ("image-a", "image-b")
        ],
        "rows": [],
        "observations": [],
    }
    for input_id in ("image-a", "image-b"):
        for backend in ("cpu", "cuda"):
            for workload in ("random-256", "whole-planes"):
                settings = [
                    ("none", None, 0.9, 10),
                    ("zstd", None, 3.123456789, 7),
                    ("blosc-zstd", 16384, 2.5, 6),
                    ("blosc-zstd", 262144, 4, 5),
                ]
                for codec, block, fold, rate in settings:
                    rate *= (1 if backend == "cpu" else 1.5) * (
                        1 if workload == "random-256" else 2
                    )
                    key = f"{input_id}-{backend}-{workload}-{codec}-{block}"
                    value = lambda median: {
                        "median": median,
                        "min": median * 0.95,
                        "max": median * 1.05,
                        "n": 3,
                    }
                    study["rows"].append(
                        {
                            "id": key,
                            "condition_id": f"{input_id}-{backend}-{workload}",
                            "input_id": input_id,
                            "backend": backend,
                            "workload": workload,
                            "sampling": {"sample_shape": [1, 256, 256]},
                            "compression_fold": fold,
                            "setting": {
                                "codec": codec,
                                "chunk_bytes": 262144,
                                "block_bytes": block,
                                "chunk_shape": [1, 256, 512],
                                "shard_shape": [1, 1024, 1024],
                                "actual_block_bytes": [block] if block else [],
                                "level": 0 if codec == "none" else 3,
                                "dtype": "uint16",
                                "stored_bytes": 100000,
                                "logical_bytes": 1000000,
                                "input_sha256": "3" * 64,
                            },
                            "metrics": {
                                "useful_gib_s": value(rate),
                                "returned_gib_s": value(2 * rate),
                                "read_amplification": value(0.5),
                                "decode_amplification": value(2),
                                "first_batch_ms": value(3),
                                "read_iops": value(rate * 1024),
                                "read_gib_s": value(rate / 2),
                                "mean_read_bytes": value(512 * 1024),
                                "reads_per_useful_gib": value(1024),
                                "device_iops": value(rate * 2048),
                                "device_read_gib_s": value(rate),
                                "device_mean_read_bytes": value(512 * 1024),
                                "device_read_amplification": value(1),
                                "idle_device_iops": value(0),
                                "idle_device_gib_s": value(0),
                                "cache_preparation_seconds": value(0.1),
                            },
                            "reference_spread_percent": 2.1,
                            "observation_ids": [key],
                        }
                    )
                    study["observations"].append(
                        {
                            "task": {"id": key},
                            "metrics": {
                                "useful_gib_s": rate,
                                "read_gib_s": rate / 2,
                                "device_read_gib_s": rate,
                            },
                            "result": {"stages": [{"name": "decode", "ms_total": 100}]},
                        }
                    )
    return study


def verify_scales(page, view="pareto"):
    assert page.locator(".panel svg").evaluate_all("""svgs => svgs.every(svg => {
      const title = svg.querySelector('text.axis-label[transform]').getBoundingClientRect();
      return [...svg.querySelectorAll('.y-axis .tick text')].every(text => {
        const tick = text.getBoundingClientRect();
        return tick.left >= title.right || tick.right <= title.left
          || tick.top >= title.bottom || tick.bottom <= title.top;
      });
    })""")
    sizes = page.locator(".panel svg").evaluate_all("""svgs => svgs.map(svg => ({
      width: svg.viewBox.baseVal.width, height: svg.viewBox.baseVal.height,
      renderedWidth: svg.getBoundingClientRect().width,
      renderedHeight: svg.getBoundingClientRect().height
    }))""")
    for size in sizes[1:]:
        assert all(
            math.isclose(size[key], sizes[0][key], abs_tol=1e-3) for key in size
        ), sizes
    panels = page.locator(".panel svg").evaluate_all("""svgs => svgs.map(svg =>
      [...svg.querySelectorAll('circle.point')].map(point => ({
        x: +point.getAttribute('cx'), y: +point.getAttribute('cy'),
        fold: point.__data__.compression_fold,
        rate: point.__data__.metrics.useful_gib_s.median,
        read_iops: point.__data__.metrics.read_iops?.median,
        read_gib_s: point.__data__.metrics.read_gib_s?.median,
        device_iops: point.__data__.metrics.device_iops?.median,
        device_read_gib_s: point.__data__.metrics.device_read_gib_s?.median,
        crop_size: point.__data__.crop_size,
        samples_per_second: point.__data__.query_summary ? point.__data__.metrics.useful_gib_s.median
          * point.__data__.query_summary.samples * 2 ** 30 / point.__data__.query_summary.useful_bytes : null,
        metrics: {...point.__data__.metrics, samples_per_second: point.__data__.query_summary ? Object.fromEntries(
          ['min', 'median', 'max'].map(key => [key, point.__data__.metrics.useful_gib_s[key]
            * point.__data__.query_summary.samples * 2 ** 30 / point.__data__.query_summary.useful_bytes])) : null},
        ranges: [...point.parentNode.querySelectorAll('path.range')].map(path => {
          const box = path.getBBox();
          return {x: box.x, y: box.y, width: box.width, height: box.height,
            after: !!(point.compareDocumentPosition(path) & Node.DOCUMENT_POSITION_FOLLOWING),
            pointer: getComputedStyle(path).pointerEvents};
        })
      })))""")
    axes = {
        "pareto": (("fold", math.log, "x", 1), ("rate", float, "y", -1)),
        "device": (
            ("device_iops", float, "x", 1),
            ("device_read_gib_s", float, "y", -1),
        ),
        "crop": (("crop_size", float, "x", 1), ("samples_per_second", float, "y", -1)),
    }
    for points in panels:
        for key, transform, coordinate, sign in axes[view]:
            low, high = (
                min(points, key=lambda p: p[key]),
                max(points, key=lambda p: p[key]),
            )
            if low[key] == high[key]:
                continue
            slope = (high[coordinate] - low[coordinate]) / (
                transform(high[key]) - transform(low[key])
            )
            assert slope * sign > 0
            for point in points:
                expected = low[coordinate] + slope * (
                    transform(point[key]) - transform(low[key])
                )
                assert math.isclose(point[coordinate], expected, abs_tol=1e-6)
                if key in {"fold", "crop_size"}:
                    continue
                metric = point["metrics"]["useful_gib_s" if key == "rate" else key]
                limits = sorted(
                    low[coordinate]
                    + slope * (transform(metric[end]) - transform(low[key]))
                    for end in ("min", "max")
                )
                bar = point["ranges"][0 if coordinate == "y" else 1]
                extent = bar["height" if coordinate == "y" else "width"]
                assert math.isclose(bar[coordinate], limits[0], abs_tol=1e-3)
                assert math.isclose(bar[coordinate] + extent, limits[1], abs_tol=1e-3)
                assert bar["after"] and bar["pointer"] == "none"


def verify_table(page, rows, view):
    headers = page.locator("#columns th").all_text_contents()
    cells = page.locator("#rows tr").evaluate_all("""rows => rows.map(row => ({
      id: row.querySelector('button').dataset.setting,
      values: [...row.children].map(cell => ({text: cell.firstChild.textContent, range: cell.querySelector('.observed-range')?.textContent}))
    }))""")
    by_id = {row["id"]: row for row in rows}
    metrics = [("Image throughput (GiB/s)", "useful_gib_s")]
    assert not any(
        word in heading
        for heading in headers
        for word in ("Encoded", "Application", "NVMe", "Useful", "Device")
    )
    if view == "device":
        metrics += [
            ("IOPS (reads/s)", "device_iops"),
            ("Storage bandwidth (GiB/s)", "device_read_gib_s"),
        ]
    assert {row["id"] for row in cells} == set(by_id)
    for row in cells:
        original = by_id[row["id"]]
        assert (
            int(row["values"][headers.index("Repeats")]["text"])
            == original["metrics"]["useful_gib_s"]["n"]
        )
        for heading, key in metrics:
            cell = row["values"][headers.index(heading)]
            metric = original["metrics"].get(key)
            if metric is None:
                assert cell["text"] == "Not recorded"
                continue
            median, limits = cell["text"], cell["range"]
            values = [median, *limits.split("\N{EN DASH}")]
            for value, name in zip(values, ("median", "min", "max")):
                assert math.isclose(
                    float(value.replace(",", "")), metric[name], rel_tol=5e-4
                )


def verify_mark_contrast(page):
    assert page.locator("circle.point").evaluate_all(r"""points => {
      const luminance = color => {
        const rgb = color.match(/[\d.]+/g).slice(0, 3).map(Number).map(v => {
          v /= 255;
          return v <= .04045 ? v / 12.92 : ((v + .055) / 1.055) ** 2.4;
        });
        return rgb[0] * .2126 + rgb[1] * .7152 + rgb[2] * .0722;
      };
      return points.every(point => {
        const stroke = luminance(getComputedStyle(point).stroke);
        const background = luminance(getComputedStyle(point.closest('.panel')).backgroundColor);
        const contrast = (Math.max(stroke, background) + .05) / (Math.min(stroke, background) + .05);
        return contrast >= 3 && getComputedStyle(point).opacity === '1'
          && getComputedStyle(point.parentNode).opacity === '1';
      });
    }""")


def verify_study(page, document, output):
    if document.get("baked"):
        verify_crop_study(page, document, output)
        return
    for asset in document.get("cohorts", document["inputs"]):
        page.select_option("#input", asset["id"])
        for workload in sorted(
            {
                row["workload"]
                for row in document["rows"]
                if row["input_id"] == asset["id"]
            }
        ):
            page.select_option("#workload", workload)
            expected = [
                row
                for row in document["rows"]
                if row["input_id"] == asset["id"] and row["workload"] == workload
            ]
            assert page.locator("circle.point").count() == len(expected)
            assert page.locator(".panel").count() == len(
                {row["condition_id"] for row in expected}
            )
            verify_scales(page)
            verify_table(page, expected, "pareto")
            if document.get("cohorts"):
                assert (
                    "Every timed pass" in page.locator("#cache-condition").inner_text()
                )
                assert (
                    f"{len(asset['input_ids'])} source inputs"
                    in page.locator("#context").inner_text()
                )
                point = page.locator("circle.point").first
                point.focus()
                point.press("Enter")
                assert point.get_attribute("aria-pressed") == "true"
                detail = page.locator("#detail-content").inner_text()
                assert "Expanded shape" in detail and "Query coverage" in detail
                assert "Leave-one-out sensitivity" in detail
                assert "Speed vs uncompressed" in detail
                selected = point.evaluate("point => point.__data__.id")
                selected_row = next(row for row in expected if row["id"] == selected)
                shown = page.locator(".sensitivity-comparison").evaluate_all("""groups => groups.map(group => ({
                  input: group.dataset.input, values: [...group.querySelectorAll('dd')].map(value => value.textContent)
                }))""")
                assert len(shown) == len(document["cohorts"])
                assert shown[0]["input"] == "all-sources"
                for comparison in shown:
                    original = next(
                        row
                        for row in document["rows"]
                        if row["input_id"] == comparison["input"]
                        and row["backend"] == selected_row["backend"]
                        and row["workload"] == selected_row["workload"]
                        and all(
                            row["setting"].get(key) == selected_row["setting"].get(key)
                            for key in (
                                "codec",
                                "chunk_bytes",
                                "block_bytes",
                                "level",
                                "shuffle",
                            )
                        )
                    )
                    values = [
                        float(value.replace(",", "")[:-1])
                        for value in comparison["values"]
                    ]
                    assert math.isclose(
                        values[0], original["compression_fold"], rel_tol=5e-4
                    )
                    assert math.isclose(
                        values[1],
                        original["metrics"]["useful_gib_s"]["median"]
                        / original["reference_useful_gib_s"]["median"],
                        rel_tol=5e-4,
                    )
            for view, metric in (("device", "device_iops"),):
                page.select_option("#view", view)
                assert page.locator("circle.point").count() == sum(
                    bool(row["metrics"].get(metric)) for row in expected
                )
                assert page.locator("path.frontier").count() == 0
                assert page.locator("#frontier-filter").is_hidden()
                verify_scales(page, view)
                verify_table(page, expected, view)
            page.select_option("#view", "pareto")


def mixture_fixture():
    study = fixture()
    study["id"] = "1" * 20
    study["label"] = "Mixed-source browser fixture — not measured"
    study["context"]["cache_policy"] = "cold-pass"
    study["cohorts"] = [
        {
            "id": "all-sources",
            "name": "All sources",
            "input_ids": ["image-a", "image-b"],
            "omitted_dataset": None,
        },
        {
            "id": "without-b",
            "name": "Without image-b",
            "input_ids": ["image-a"],
            "omitted_dataset": "image-b",
        },
    ]
    study["volume"] = {
        "frame_shape": [2048, 2048],
        "shard_spatial_shape": [512, 512],
        "shard_bytes": 2**30,
    }
    study["weighting"] = "Equal sample counts per input; original source byte sizes"
    for row in study["rows"]:
        cohort = study["cohorts"][row["input_id"] == "image-b"]
        row["input_id"] = cohort["id"]
        if row["workload"] == "random-256":
            row["workload"] = "balanced-256"
        row["setting"]["dtype"] = "mixed"
        row["sampling"]["sample_shape"] = (
            [1, 2048, 2048] if row["workload"] == "whole-planes" else [1, 256, 256]
        )
        row["reference_useful_gib_s"] = {"median": 10, "min": 9, "max": 11, "n": 3}
        if (
            row["setting"]["codec"] == "blosc-zstd"
            and row["setting"]["block_bytes"] == 262144
        ):
            row["setting"]["codec"] = "blosc-lz4"
        row["setting"]["sources"] = [
            {
                "input_id": name,
                "source_shape": [6, 600, 600],
                "shape": [2048, 2048, 2048],
                "dtype": "uint16",
                "chunk_shape": [1, 256, 512],
                "shard_shape": [2048, 512, 512],
                "shard_files": 16,
                "shard_file_bytes": [2**29] * 16,
                "input_sha256": "f" * 64,
            }
            for name in cohort["input_ids"]
        ]
        row["query_summary"] = {
            "samples": 128,
            "useful_bytes": 2**24,
            "distinct_shards": 16 * len(cohort["input_ids"]),
            "shards_per_window": {
                "batch": {"min": 16, "max": 32},
                "lookahead": {"min": 16, "max": 32},
            },
            "source_samples": [
                {"input_id": name, "samples": 128 // len(cohort["input_ids"])}
                for name in cohort["input_ids"]
            ],
        }
        row["sampling"].update(query_id="fixture", query_sha256="f" * 64)
    return study


def crop_fixture():
    study = mixture_fixture()
    base_rows = [
        row
        for row in study["rows"]
        if row["input_id"] == "all-sources" and row["workload"] == "balanced-256"
    ]
    observations = {row["task"]["id"]: row for row in study["observations"]}
    study.update(
        id="2" * 20,
        label="Crop-size browser fixture — not measured",
        rows=[],
        observations=[],
    )
    study["baked"] = {
        "active_shards": 16,
        "shard_bytes": 2**30,
        "source_shards": {"image-a": 8, "image-b": 8},
    }
    study["batch_output_bytes"] = 36 * 2**20
    study["lookahead_output_bytes"] = 144 * 2**20
    for access in ("random", "baked"):
        for crop in (128, 192, 256):
            for original in base_rows:
                row = copy.deepcopy(original)
                row.update(
                    id=f"{access}-{crop}-{original['id']}",
                    access_pattern=access,
                    crop_size=crop,
                    workload=f"{access}-{crop}",
                    condition_id=f"{access}-{crop}-{row['backend']}",
                )
                row["sampling"]["sample_shape"] = [1, crop, crop]
                row["pipeline"] = {
                    "lookahead_samples": 144 * 2**20 // (crop * crop * 4)
                }
                row["query_summary"]["useful_bytes"] = (
                    row["query_summary"]["samples"] * crop * crop * 2
                )
                row["query_summary"]["samples_crossing_shards"] = (
                    0 if access == "baked" else 30
                )
                setting = row["setting"]
                setting["recipe_id"] = f"{setting['codec']}-{setting['block_bytes']}"
                setting["chunk_mode"] = "sample" if access == "baked" else "bytes"
                setting["block_mode"] = (
                    "chunk"
                    if setting["block_bytes"] == 262144
                    else "bytes"
                    if setting["block_bytes"]
                    else "none"
                )
                if setting["block_mode"] == "chunk":
                    setting["block_bytes"] = None
                for source in setting["sources"]:
                    source["chunk_bytes"] = 262144
                if access == "baked":
                    setting["chunk_bytes"] = None
                    row["query_summary"]["source_query_id"] = "f" * 20
                    row["query_summary"]["distinct_shards"] = 16
                    for source in setting["sources"]:
                        depth = 2**30 // (crop * crop * 2)
                        source.update(
                            baked={"sample_shape": [1, crop, crop]},
                            shape=[8 * depth, crop, crop],
                            shard_shape=[depth, crop, crop],
                            chunk_shape=[1, crop, crop],
                            chunk_bytes=crop * crop * 2,
                            shard_files=8,
                        )
                observation = copy.deepcopy(
                    observations[original["observation_ids"][0]]
                )
                observation["task"]["id"] = row["id"]
                row["observation_ids"] = [row["id"]]
                study["rows"].append(row)
                study["observations"].append(observation)
    for original in study["rows"].copy():
        row = copy.deepcopy(original)
        row.update(
            id=f"omission-{original['id']}",
            input_id="without-b",
            condition_id=f"omission-{original['condition_id']}",
        )
        row["setting"]["sources"] = row["setting"]["sources"][:1]
        row["query_summary"]["distinct_shards"] //= 2
        row["query_summary"]["samples"] //= 2
        row["query_summary"]["useful_bytes"] //= 2
        study["rows"].append(row)
    return study


def verify_crop_study(page, document, output):
    output = output / f"study-{document['id']}"
    output.mkdir(exist_ok=True)
    if page.locator("#input").is_visible():
        page.select_option("#input", "all-sources")
    for access in ("random", "baked"):
        page.select_option("#workload", access)
        page.select_option("#view", "pareto")
        for crop in (128, 192, 256):
            page.select_option("#crop", str(crop))
            expected = [
                row
                for row in document["rows"]
                if row["input_id"] == "all-sources"
                and row["access_pattern"] == access
                and row["crop_size"] == crop
            ]
            assert page.locator("circle.point").count() == len(expected)
            assert page.locator(".panel").count() == len(
                {row["backend"] for row in expected}
            )
            assert page.locator("path.frontier").evaluate_all(
                "paths => paths.every(path => new Set(path.__data__.map(row => row.crop_size)).size === 1)"
            )
            verify_scales(page)
            verify_table(page, expected, "pareto")
            page.select_option("#view", "device")
            if page.locator("svg.storage-work").count():
                from test_pareto_overview import verify_storage

                verify_storage(page, document, expected)
            else:
                verify_scales(page, "device")
            verify_table(page, expected, "device")
            page.select_option("#view", "pareto")
        point = page.locator("circle.point").first
        point.focus()
        point.press("Enter")
        assert point.get_attribute("aria-pressed") == "true"
        selected_id = point.evaluate("node => node.__data__.id")
        measurements = page.locator(".measurements")
        was_open = measurements.evaluate("node => node.open")
        if not was_open:
            measurements.locator("summary").click()
        page.locator(f'#rows button[data-setting="{selected_id}"]').click()
        assert point.get_attribute("aria-pressed") == "true"
        if not was_open:
            measurements.locator("summary").click()
        detail = page.locator("#detail-content").inner_text()
        assert "Samples/s" in detail
        if access == "baked":
            assert "One sample" in detail and "Baked shape" in detail
            assert "Download paired source coordinates" in detail
            assert (
                "16 shard files total per process"
                in page.locator("#context").inner_text()
            )
        page.select_option("#view", "crop")
        assert page.locator("#crop-label").is_hidden()
        assert page.locator("#frontier-filter").is_hidden()
        assert page.locator("path.frontier").count() == 0
        assert page.locator("circle.point").count() == sum(
            row["input_id"] == "all-sources" and row["access_pattern"] == access
            for row in document["rows"]
        )
        verify_scales(page, "crop")
        assert page.locator("path.crop-series").evaluate_all(
            "paths => paths.every(path => new Set(path.__data__.map(row => row.setting.recipe_id)).size === 1)"
        )
        with page.expect_download() as downloaded:
            page.locator("#csv").click()
        rows = list(
            csv.DictReader(io.StringIO(Path(downloaded.value.path()).read_text()))
        )
        expected_by_id = {row["id"]: row for row in document["rows"]}
        for row in rows:
            original = expected_by_id[row["setting_id"]]
            expected_rate = (
                original["metrics"]["useful_gib_s"]["median"]
                * original["query_summary"]["samples"]
                * 2**30
                / original["query_summary"]["useful_bytes"]
            )
            assert math.isclose(
                float(row["samples_per_second_median"]), expected_rate, rel_tol=1e-14
            )
            assert row["access_pattern"] == access
        for theme in ("light", "dark"):
            page.select_option("#appearance", theme)
            for width in (1440, 390, 320):
                page.set_viewport_size({"width": width, "height": 1100})
                page.evaluate(
                    "() => new Promise(resolve => requestAnimationFrame(resolve))"
                )
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth + 1"
                )
                verify_scales(page, "crop")
                verify_mark_contrast(page)
                page.screenshot(
                    path=str(output / f"crops-{access}-{theme}-{width}.png"),
                    full_page=True,
                )
        page.set_viewport_size({"width": 1440, "height": 1100})
    page.select_option("#view", "pareto")
    page.select_option("#appearance", "system")
    page.locator(".more-filters").evaluate("node => { node.open = true; }")
    page.select_option("#chunk", "sample")
    page.select_option("#workload", "random")
    assert page.locator("#chunk").input_value() == ""
    assert page.locator("circle.point").count() > 0
    page.select_option("#chunk", "262144")
    page.select_option("#workload", "baked")
    assert page.locator("#chunk").input_value() == ""
    assert page.locator("circle.point").count() > 0
    if any(cohort["id"] == "without-b" for cohort in document["cohorts"]):
        page.select_option("#input", "without-b")
        assert (
            "8 shard files total per process" in page.locator("#context").inner_text()
        )
        page.select_option("#input", "all-sources")
    page.locator(".more-filters").evaluate("node => { node.open = false; }")


def check(assets, output, actual):
    if actual:
        from test_pareto_overview import check_actual

        return check_actual(assets, output)
    with tempfile.TemporaryDirectory(prefix="damacy-browser-") as temporary:
        root = Path(temporary)
        shutil.copytree(assets, root / "pareto")
        report = root / "pareto"
        if not actual:
            document = fixture()
            index = {
                "version": 1,
                "studies": [
                    {
                        "id": document["id"],
                        "file": f"{document['id']}.json",
                        "label": document["label"],
                        "pilot": True,
                    }
                ],
            }
            mixed = mixture_fixture()
            index["studies"].append(
                {
                    "id": mixed["id"],
                    "file": f"{mixed['id']}.json",
                    "label": mixed["label"],
                    "pilot": True,
                }
            )
            (report / f"data/{mixed['id']}.json").write_text(json.dumps(mixed))
            crops = crop_fixture()
            index["studies"].insert(
                0,
                {
                    "id": crops["id"],
                    "file": f"{crops['id']}.json",
                    "label": crops["label"],
                    "pilot": True,
                },
            )
            (report / f"data/{crops['id']}.json").write_text(json.dumps(crops))
            cpu = copy.deepcopy(crops)
            cpu.update(id="3" * 20, label="CPU crop browser fixture — not measured")
            cpu["rows"] = [row for row in cpu["rows"] if row["backend"] == "cpu"]
            cpu["context"]["machine"]["gpu"] = None
            index["studies"].append(
                {
                    "id": cpu["id"],
                    "file": f"{cpu['id']}.json",
                    "label": cpu["label"],
                    "pilot": True,
                }
            )
            (report / f"data/{cpu['id']}.json").write_text(json.dumps(cpu))
            (report / "data/index.json").write_text(json.dumps(index))
            (report / f"data/{document['id']}.json").write_text(json.dumps(document))
        index = json.loads((report / "data/index.json").read_text())
        if not index["studies"]:
            raise ValueError("Browser measurement checks need a retained study")
        documents = {
            entry["id"]: json.loads((report / "data" / entry["file"]).read_text())
            for entry in index["studies"]
        }
        document = next(value for value in documents.values() if not value.get("baked"))
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(QuietHandler, directory=str(root))
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        output.mkdir(parents=True, exist_ok=True)
        errors = []
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(
                    headless=True, args=["--no-sandbox"]
                )
                url = f"http://127.0.0.1:{server.server_port}/pareto/"
                if not actual:
                    verify_study_refresh(browser, url, report / "data/index.json")
                page = browser.new_page(
                    viewport={"width": 1440, "height": 1100},
                    device_scale_factor=1,
                    color_scheme="light",
                    locale="en-US",
                )
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url + "?check=default")
                page.wait_for_function("document.getElementById('status').hidden")
                page.locator("#explorer").evaluate("node => { node.open = true; }")
                page.wait_for_selector("#workspace", state="visible")
                assert page.locator("#study").input_value() == index["studies"][0]["id"]
                assert page.locator("circle.point").count() > 0
                page.goto(url + f"#study={document['id']}&view=io")
                page.wait_for_selector("#workspace", state="visible")
                assert page.locator("#view").input_value() == "device"
                assert page.locator("#view option").all_text_contents() == [
                    "Image throughput and compression",
                    "Storage reads",
                ]
                page.select_option("#view", "pareto")
                for entry in index["studies"]:
                    current = documents[entry["id"]]
                    page.select_option("#study", entry["id"])
                    page.wait_for_selector("#workspace", state="visible")
                    verify_study(page, current, output)
                    if current.get("cohorts") and not current.get("baked"):
                        page.select_option("#input", "all-sources")
                        point = page.locator("circle.point").last
                        point.click()
                        assert point.get_attribute("aria-pressed") == "true"
                        for theme in ("light", "dark"):
                            page.select_option("#appearance", theme)
                            verify_mark_contrast(page)
                            page.screenshot(
                                path=str(output / f"mixture-{entry['id']}-{theme}.png"),
                                full_page=True,
                            )
                            page.locator(".primary-metrics").screenshot(
                                path=str(
                                    output / f"measurements-{entry['id']}-{theme}.png"
                                )
                            )
                            detail = page.locator("#detail")
                            detail.scroll_into_view_if_needed()
                            for section, heading in (
                                ("sensitivity", "Leave-one-out sensitivity"),
                                ("sources", "Input and source"),
                            ):
                                detail.get_by_role(
                                    "heading", name=heading, exact=True
                                ).evaluate(
                                    "node => node.scrollIntoView({block: 'start'})"
                                )
                                assert detail.evaluate("node => node.scrollTop > 0")
                                detail.screenshot(
                                    path=str(
                                        output
                                        / f"mixture-{entry['id']}-{section}-{theme}.png"
                                    )
                                )
                            detail.evaluate("node => { node.scrollTop = 0; }")
                            page.evaluate("window.scrollTo(0, 0)")
                        page.select_option("#appearance", "system")
                page.select_option("#study", document["id"])
                page.wait_for_selector("#workspace", state="visible")
                input_id = document.get("cohorts", document["inputs"])[0]["id"]
                workload = min(
                    row["workload"]
                    for row in document["rows"]
                    if row["input_id"] == input_id
                )
                page.select_option("#input", input_id)
                page.select_option("#workload", workload)
                expected = [
                    row
                    for row in document["rows"]
                    if row["input_id"] == input_id and row["workload"] == workload
                ]
                point = page.locator('circle.point[aria-label^="Uncompressed"]').first
                point.focus()
                point.press("Enter")
                assert (
                    "Image throughput" in page.locator("#detail-content").inner_text()
                )
                assert page.locator(".primary-metrics dt").all_text_contents() == [
                    "Image throughput",
                    "Compression ratio",
                    "Storage bandwidth",
                    "IOPS",
                ]
                diagnostics = page.locator(".diagnostics")
                assert diagnostics.get_attribute("open") is None
                assert (
                    "File read bandwidth"
                    not in page.locator("#detail-content").inner_text()
                )
                diagnostics.locator("summary").click()
                assert "File read bandwidth" in diagnostics.inner_text()
                assert "Revision" in diagnostics.inner_text()
                page.select_option("#view", "device")
                assert diagnostics.get_attribute("open") is not None
                page.set_viewport_size({"width": 1400, "height": 1100})
                page.evaluate(
                    "() => new Promise(resolve => requestAnimationFrame(resolve))"
                )
                assert diagnostics.get_attribute("open") is not None
                page.select_option("#view", "pareto")
                diagnostics.locator("summary").click()
                assert diagnostics.get_attribute("open") is None
                page.locator("#detail").evaluate("node => { node.scrollTop = 0; }")
                assert point.get_attribute("aria-pressed") == "true"
                assert page.locator('#rows button[aria-pressed="true"]').count() == 1
                assert "selected" in page.locator("#selection-status").inner_text()
                page.locator("#selection-details").click()
                assert page.evaluate("document.activeElement.id") == "detail-heading"
                verify_mark_contrast(page)
                with page.expect_download() as downloaded:
                    page.locator("#csv").click()
                exported = list(
                    csv.DictReader(
                        io.StringIO(Path(downloaded.value.path()).read_text())
                    )
                )
                assert len(exported) == len(expected)
                by_id = {row["id"]: row for row in expected}
                for row in exported:
                    original = by_id[row["setting_id"]]
                    assert (
                        float(row["image_throughput_gib_s_median"])
                        == original["metrics"]["useful_gib_s"]["median"]
                    )
                    assert (
                        float(row["compression_ratio"]) == original["compression_fold"]
                    )
                    for column, metric in (
                        ("iops", "device_iops"),
                        ("storage_bandwidth_gib_s", "device_read_gib_s"),
                    ):
                        value = original["metrics"].get(metric)
                        assert (
                            float(row[f"{column}_median"])
                            if row[f"{column}_median"]
                            else None
                        ) == (value["median"] if value else None)
                assert not any(
                    "encoded" in key or "read_iops" in key or "useful" in key
                    for key in exported[0]
                )
                page.locator(".more-filters summary").click()
                page.select_option("#block", "16384")
                assert page.locator("circle.point").count() == sum(
                    not row["setting"]["codec"].startswith("blosc-")
                    or row["setting"]["block_bytes"] == 16384
                    for row in expected
                )
                page.locator("#frontier-only").check()
                assert page.locator("circle.point").count() > 0
                page.select_option("#view", "device")
                assert page.locator("circle.point").count() == sum(
                    not row["setting"]["codec"].startswith("blosc-")
                    or row["setting"]["block_bytes"] == 16384
                    for row in expected
                )
                page.select_option("#view", "pareto")
                page.locator("#frontier-only").uncheck()
                page.select_option("#block", "")
                page.select_option("#codec", "zstd")
                assert page.locator("#selection-details").is_hidden()
                assert page.locator('circle.point[aria-pressed="true"]').count() == 0
                page.select_option("#view", "device")
                page.reload()
                page.wait_for_selector("#workspace", state="visible")
                assert page.locator("#codec").input_value() == "zstd"
                assert page.locator("#view").input_value() == "device"
                page.locator(".more-filters summary").click()
                page.select_option("#codec", "")
                assert page.locator("#appearance").input_value() == "system"
                page.emulate_media(color_scheme="dark")
                page.wait_for_function(
                    "document.documentElement.dataset.theme === 'dark'"
                )
                page.select_option("#appearance", "light")
                page.wait_for_function(
                    "document.documentElement.dataset.theme === 'light'"
                )
                page.reload()
                page.wait_for_selector("#workspace", state="visible")
                assert page.locator("#appearance").input_value() == "light"
                assert (
                    page.evaluate(
                        "getComputedStyle(document.documentElement).colorScheme"
                    )
                    == "light"
                )
                page.select_option("#appearance", "dark")
                page.emulate_media(color_scheme="light")
                assert page.evaluate("document.documentElement.dataset.theme") == "dark"
                page.select_option("#appearance", "system")
                page.wait_for_function(
                    "document.documentElement.dataset.theme === 'light'"
                )
                page.locator("#storage-details summary").click()
                assert "Machine" in page.locator("#storage-content").inner_text()
                assert (
                    "All panels in this study use this storage system"
                    in page.locator("#storage-content").inner_text()
                )
                page.locator("#storage-details").screenshot(
                    path=str(output / "storage-system.png")
                )
                page.locator("#storage-details summary").click()
                page.evaluate("window.scrollTo(0, 0)")
                for width in (1440, 768, 390, 320):
                    page.set_viewport_size({"width": width, "height": 1100})
                    page.evaluate(
                        "() => new Promise(resolve => requestAnimationFrame(resolve))"
                    )
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= innerWidth + 1"
                    ), (
                        width,
                        page.evaluate("""[...document.querySelectorAll('body *')]
                      .filter(node => node.getBoundingClientRect().right > innerWidth + 1)
                      .slice(0, 12).map(node => ({tag: node.tagName, id: node.id,
                        class: node.getAttribute('class'), right: node.getBoundingClientRect().right}))"""),
                    )
                    page.screenshot(
                        path=str(
                            output / f"{'actual' if actual else 'fixture'}-{width}.png"
                        ),
                        full_page=True,
                    )
                page.emulate_media(color_scheme="dark")
                page.wait_for_function(
                    "document.documentElement.dataset.theme === 'dark'"
                )
                page.set_viewport_size({"width": 1440, "height": 1100})
                page.screenshot(
                    path=str(output / f"{'actual' if actual else 'fixture'}-dark.png"),
                    full_page=True,
                )
                page.select_option("#view", "pareto")
                page.screenshot(path=str(output / "pareto-dark.png"), full_page=True)
                verify_mark_contrast(page)
                page.select_option("#view", "device")
                page.locator(".measurements summary").click()
                page.locator("#rows button").first.click()
                assert page.locator('circle.point[aria-pressed="true"]').count() == 1
                page.screenshot(
                    path=str(output / "device-table-dark.png"), full_page=True
                )
                page.set_viewport_size({"width": 390, "height": 850})
                page.locator("#selection-details").click()
                assert page.evaluate("document.activeElement.id") == "detail-heading"
                assert page.locator("#detail-heading").evaluate(
                    "node => { const box = node.getBoundingClientRect(); return box.top >= 0 && box.bottom <= innerHeight; }"
                )
                page.screenshot(
                    path=str(output / "device-selected-390.png"), full_page=True
                )
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= innerWidth + 1"
                ), page.evaluate("""[...document.querySelectorAll('body *')]
                  .filter(node => node.getBoundingClientRect().right > innerWidth + 1)
                  .slice(0, 8).map(node => ({tag: node.tagName, class: node.className,
                    right: node.getBoundingClientRect().right}))""")
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        (output / "validation.json").write_text(
            json.dumps(
                {
                    "actual_measurements": actual,
                    "settings": sum(len(value["rows"]) for value in documents.values()),
                    "default_study": index["studies"][0]["id"],
                    "studies_checked": len(index["studies"]),
                    "console_errors": errors,
                    "viewports": [1440, 768, 390, 320],
                },
                indent=2,
            )
            + "\n"
        )
        count = sum(len(value["rows"]) for value in documents.values())
        print(f"Browser checks passed for {count} settings: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/pareto",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actual", action="store_true")
    args = parser.parse_args()
    check(args.assets, args.output, args.actual)


if __name__ == "__main__":
    main()
