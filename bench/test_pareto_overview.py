"""Check the published comparisons against their retained observations."""

import json
import math
import shutil
import statistics
import tempfile
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright


def verify_storage(page, document, rows):
    observations = {value["task"]["id"]: value for value in document["observations"]}
    plotted = page.locator("svg.storage-work").evaluate_all(
        "nodes => nodes.map(node => ({...node.dataset, x: +node.querySelector('circle').getAttribute('cx')}))"
    )
    assert len(plotted) == len(rows) * 2
    by_id = {row["id"]: row for row in rows}
    for point in plotted:
        row = by_id[point["setting"]]
        values = []
        for identity in row["observation_ids"]:
            observation = observations[identity]
            samples = observation["metrics"]["passes"] * row["query_summary"]["samples"]
            counter = (
                "device_read_bytes" if point["metric"] == "bytes" else "device_reads"
            )
            divisor = 1024 if point["metric"] == "bytes" else 1
            values.append(
                observation["result"]["cold_reads"][counter] / samples / divisor
            )
        for name, value in (
            ("median", statistics.median(values)),
            ("min", min(values)),
            ("max", max(values)),
        ):
            assert math.isclose(float(point[name]), value, rel_tol=1e-12)
    for metric in ("bytes", "reads"):
        axes = page.locator(f'.storage-axis[data-metric="{metric}"]').evaluate_all(
            "axes => axes.map(axis => [...axis.querySelectorAll('.tick')].map(tick => [tick.textContent, tick.getAttribute('transform')]))"
        )
        assert all(axis == axes[0] for axis in axes)
        points = [point for point in plotted if point["metric"] == metric]
        low, high = min(points, key=lambda p: p["x"]), max(points, key=lambda p: p["x"])
        slope = (high["x"] - low["x"]) / (float(high["median"]) - float(low["median"]))
        assert slope > 0
        for point in points:
            expected = low["x"] + slope * (
                float(point["median"]) - float(low["median"])
            )
            assert math.isclose(point["x"], expected, abs_tol=1e-6)


def verify_overview(page, documents, output):
    assert page.locator("#overview").is_visible()
    assert (
        "4.3\N{EN DASH}10.4\N{MULTIPLICATION SIGN}"
        in page.locator("#layout-title").inner_text()
    )
    assert page.locator(".layout-pair").count() == 9
    rows = {
        (document["id"], row["id"]): (document, row)
        for document in documents.values()
        for row in document["rows"]
    }
    links = page.locator(".winner a").evaluate_all(
        "links => links.map(link => link.href)"
    )
    assert len(links) == 18
    for href in links:
        query = parse_qs(urlparse(href).fragment)
        document, row = rows[(query["study"][0], query["setting"][0])]
        assert document["id"] == query["study"][0]
        peers = [
            candidate
            for candidate in document["rows"]
            if candidate["backend"] == row["backend"]
            and candidate["crop_size"] == row["crop_size"]
            and candidate["access_pattern"] == row["access_pattern"]
        ]
        assert row["metrics"]["useful_gib_s"]["median"] == max(
            value["metrics"]["useful_gib_s"]["median"] for value in peers
        )
    assert page.locator("#parameter-comparison .parameter-pair").count() == 6
    options = page.locator("#overview-recipe option").evaluate_all(
        "options => options.map(option => option.value)"
    )
    assert len(options) == 5
    for option in options:
        page.select_option("#overview-recipe", option)
        curves = page.locator(".summary-curve").evaluate_all(
            "paths => paths.map(path => ({recipe: path.dataset.recipe, rows: path.__data__}))"
        )
        assert len(curves) == 6
        for curve in curves:
            assert {row["setting"]["recipe_id"] for row in curve["rows"]} == {
                curve["recipe"]
            }
            assert [row["crop_size"] for row in curve["rows"]] == [128, 192, 256]
        axes = page.locator(".summary-y-axis").evaluate_all(
            "axes => axes.map(axis => axis.textContent)"
        )
        assert axes[0] == axes[1]
        assert page.locator("#crop-comparison .summary-mark").count() == 18
        assert page.locator("#crop-comparison tbody tr").count() == 6
    page.reload()
    page.wait_for_selector("#overview", state="visible")
    assert page.locator("#overview-recipe").input_value() == options[-1]
    page.select_option("#overview-recipe", options[0])
    assert "83,300" in page.locator("#crop-finding").inner_text()
    assert "71,300" in page.locator("#crop-finding").inner_text()
    page.locator("#explorer").evaluate("node => { node.open = false; }")
    for theme in ("light", "dark"):
        page.select_option("#appearance", theme)
        for width in (1440, 768, 390, 320):
            page.set_viewport_size({"width": width, "height": 1100})
            page.evaluate(
                "() => new Promise(resolve => requestAnimationFrame(resolve))"
            )
            assert page.evaluate(
                "document.documentElement.scrollWidth <= innerWidth + 1"
            ), width
            assert page.locator(".summary-mark").evaluate_all(r"""nodes => {
              const luminance = color => {
                const values = color.match(/[\d.]+/g).slice(0, 3).map(Number).map(value => {
                  value /= 255;
                  return value <= .04045 ? value / 12.92 : ((value + .055) / 1.055) ** 2.4;
                });
                return values[0] * .2126 + values[1] * .7152 + values[2] * .0722;
              };
              return nodes.every(node => {
                const stroke = luminance(getComputedStyle(node).stroke);
                const background = luminance(getComputedStyle(node.closest('.comparison-panel')).backgroundColor);
                const contrast = (Math.max(stroke, background) + .05) / (Math.min(stroke, background) + .05);
                const box = node.getBoundingClientRect(), svg = node.closest('svg').getBoundingClientRect();
                return contrast >= 3 && box.left >= svg.left && box.right <= svg.right
                  && box.top >= svg.top && box.bottom <= svg.bottom;
              });
            }""")
            page.screenshot(
                path=str(output / f"overview-{theme}-{width}.png"), full_page=True
            )
            if width in (1440, 390):
                for section in ("layout", "crop", "parameter"):
                    page.locator(f"#{section}-comparison").screenshot(
                        path=str(output / f"{section}-{theme}-{width}.png")
                    )
    page.set_viewport_size({"width": 1440, "height": 1100})
    first = page.locator(".winner a").first
    setting = parse_qs(urlparse(first.get_attribute("href")).fragment)["setting"][0]
    first.focus()
    first.press("Enter")
    page.wait_for_function("document.activeElement.id === 'explorer-heading'")
    assert page.locator("#explorer").evaluate("node => node.open")
    assert (
        page.locator('circle.point[aria-pressed="true"]').evaluate(
            "node => node.__data__.id"
        )
        == setting
    )
    page.reload()
    page.wait_for_selector("#workspace", state="visible")
    assert (
        page.locator('circle.point[aria-pressed="true"]').evaluate(
            "node => node.__data__.id"
        )
        == setting
    )


def check_actual(assets, output):
    from test_pareto_browser import (
        QuietHandler,
        verify_crop_study,
        verify_study_refresh,
    )

    with tempfile.TemporaryDirectory(prefix="damacy-overview-") as temporary:
        root = Path(temporary)
        report = root / "pareto"
        shutil.copytree(assets, report)
        index = json.loads((report / "data/index.json").read_text())
        assert len(index["studies"]) == 2
        documents = {
            entry["id"]: json.loads((report / "data" / entry["file"]).read_text())
            for entry in index["studies"]
        }
        assert all(document.get("baked") for document in documents.values())
        assert sum(len(document["rows"]) for document in documents.values()) == 93
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
                verify_study_refresh(browser, url, report / "data/index.json")
                page = browser.new_page(
                    viewport={"width": 1440, "height": 1100},
                    locale="en-US",
                    color_scheme="light",
                )
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                page.wait_for_selector("#overview", state="visible")
                page.wait_for_function("document.getElementById('status').hidden")
                assert not page.locator("#explorer").evaluate("node => node.open")
                assert page.locator("#study option").count() == 2
                assert "history" not in page.locator("header").inner_text().lower()
                page.screenshot(path=str(output / "overview-initial.png"), full_page=True)
                verify_overview(page, documents, output)
                for document in documents.values():
                    page.select_option("#study", document["id"])
                    page.wait_for_selector("#workspace", state="visible")
                    verify_crop_study(page, document, output)
                page.select_option("#view", "device")
                page.select_option("#appearance", "system")
                page.emulate_media(color_scheme="dark")
                page.wait_for_function(
                    "document.documentElement.dataset.theme === 'dark'"
                )
                page.select_option("#appearance", "light")
                page.reload()
                page.wait_for_selector("#workspace", state="visible")
                assert (
                    page.evaluate("document.documentElement.dataset.theme") == "light"
                )
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
                        page.locator("#workspace").screenshot(
                            path=str(output / f"storage-{theme}-{width}.png")
                        )
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        (output / "validation.json").write_text(
            json.dumps(
                {
                    "actual_measurements": True,
                    "settings": 93,
                    "studies_checked": 2,
                    "console_errors": errors,
                    "viewports": [1440, 768, 390, 320],
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Browser checks passed for the two crop studies and overview: {output}")
