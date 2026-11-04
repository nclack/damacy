"""Checks for paired queries, measurement units, and retained study integrity."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

from pareto import (
    DTYPES,
    GIB,
    IO_METRICS,
    block_devices,
    chunk_shape,
    identity,
    make_plan,
    observation_metrics,
    read_json,
    relative_path,
    schedule,
    validate_prepared,
    validate_saved_metrics,
)


def fixture():
    definition = read_json(Path(__file__).parent / "studies/microscopy.json")
    definition["cache_policy"] = "preloaded files"
    assets = []
    for name in definition["inputs"]:
        dtype = "float32" if name == "dynacell-a549-phase" else "uint16"
        if name == "cosem-cos7-em":
            dtype = "uint8"
        assets.append(
            {
                "id": name,
                "name": name,
                "path": f"data/{name}.raw",
                "dtype": dtype,
                "shape": [5, 1024, 1024],
                "sha256": "1" * 64,
            }
        )
    corpus = {
        "format": {
            "name": "chucky-image-corpus",
            "version": 2,
            "encoding": "raw",
            "byte_order": "little",
            "axes": ["plane", "y", "x"],
            "order": "C",
        },
        "id": "test-corpus",
        "version": 3,
        "datasets": [
            {
                "id": "test",
                "version": 1,
                "modality": "test",
                "source": {},
                "assets": assets,
            }
        ],
    }
    return definition, corpus


class PlanTests(unittest.TestCase):
    def test_cold_passes_contain_one_batch(self):
        definition, corpus = fixture()
        definition["cache_policy"] = "cold"
        plan = make_plan(definition, corpus, pilot=True)
        self.assertEqual(plan["cache_policy"], "cold")
        for case in plan["cases"]:
            self.assertEqual(case["sampling"]["n_batches"], 1)
            if case["sampling"]["pattern"] == "planes":
                self.assertEqual(case["sampling"]["samples_per_batch"], 5)

    def test_matrix_and_pilot_coverage(self):
        definition, corpus = fixture()
        full = make_plan(definition, corpus)
        pilot = make_plan(definition, corpus, pilot=True)
        self.assertEqual(
            (len(full["variants"]), len(full["cases"]), len(schedule(full))),
            (105, 420, 1428),
        )
        self.assertEqual(
            (len(pilot["variants"]), len(pilot["cases"]), len(schedule(pilot))),
            (12, 48, 144),
        )
        self.assertEqual({v["dtype"] for v in pilot["variants"]}, set(DTYPES))

    def test_queries_match_across_layouts_and_backends(self):
        plan = make_plan(*fixture())
        conditions = {}
        for case in plan["cases"]:
            key = (case["input_id"], case["workload"])
            conditions.setdefault(key, set()).add(identity(case["sampling"]))
        self.assertTrue(all(len(values) == 1 for values in conditions.values()))

    def test_whole_plane_passes_cover_every_plane(self):
        plan = make_plan(*fixture(), pilot=True)
        for case in plan["cases"]:
            if case["sampling"]["pattern"] == "planes":
                count = (
                    case["sampling"]["samples_per_batch"]
                    * case["sampling"]["n_batches"]
                )
                self.assertEqual(count % 5, 0)

    def test_rounds_bracket_each_condition(self):
        plan = make_plan(*fixture(), pilot=True)
        tasks = schedule(plan)
        cases = {case["id"]: case for case in plan["cases"]}
        self.assertEqual(tasks, schedule(plan))
        self.assertEqual(len({t["id"] for t in tasks}), len(tasks))
        groups = {}
        for task in tasks:
            case = cases[task["case_id"]]
            key = (task["round"], case["input_id"], case["backend"], case["workload"])
            groups.setdefault(key, []).append(task["role"])
        for roles in groups.values():
            self.assertEqual(
                roles, ["reference-before", *(["sample"] * 4), "reference-after"]
            )

    def test_corpus_version_must_match(self):
        definition, corpus = fixture()
        corpus["version"] = 1
        with self.assertRaisesRegex(ValueError, "Corpus version"):
            make_plan(definition, corpus)

    def test_no_codec_or_block_silently_substituted(self):
        definition, corpus = fixture()
        definition["codecs"].append("lz4")
        with self.assertRaisesRegex(ValueError, "Unsupported reader codec"):
            make_plan(definition, corpus)
        definition["codecs"].pop()
        definition["blosc_block_bytes"] = [0]
        with self.assertRaisesRegex(ValueError, "positive integer"):
            make_plan(definition, corpus)

    def test_chunk_shapes_preserve_bytes(self):
        for dtype, (_, itemsize) in DTYPES.items():
            for byte_count in (16384, 65536, 262144, 1048576):
                with self.subTest(dtype=dtype, byte_count=byte_count):
                    shape = chunk_shape(byte_count, itemsize)
                    self.assertEqual(math.prod(shape) * itemsize, byte_count)
                    self.assertEqual(shape[0], 1)
        with self.assertRaises(ValueError):
            chunk_shape(65000, 2)

    def test_block_count_fits_the_reader_limit(self):
        definition, corpus = fixture()
        definition["pipeline"]["max_substreams_per_chunk"] = 32
        with self.assertRaisesRegex(ValueError, "Blosc blocks exceed"):
            make_plan(definition, corpus)

    def test_prepared_layouts_cannot_change_bytes_or_codec(self):
        plan = make_plan(*fixture(), pilot=True)
        prepared = {"plan_sha256": identity(plan), "variants": []}
        for variant in plan["variants"]:
            prepared["variants"].append(
                {
                    **variant,
                    "logical_bytes": math.prod(variant["shape"])
                    * DTYPES[variant["dtype"]][1],
                    "stored_bytes": 64,
                    "files": [
                        {"path": f"{variant['id']}/array/zarr.json", "bytes": 64}
                    ],
                }
            )
        validate_prepared(plan, prepared)
        wrong = copy.deepcopy(prepared)
        wrong["variants"][0]["stored_bytes"] = 32
        with self.assertRaisesRegex(ValueError, "Stored byte count"):
            validate_prepared(plan, wrong)
        wrong = copy.deepcopy(prepared)
        wrong["variants"][0]["codec"] = "other"
        with self.assertRaisesRegex(ValueError, "settings changed"):
            validate_prepared(plan, wrong)

    def test_file_paths_cannot_escape_the_prepared_tree(self):
        for path in ("../outside", "/outside", "array/../../outside", "array\\outside"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                relative_path(Path("data"), path)


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.sampling = {
            "pattern": "random",
            "sample_shape": [1, 16, 16],
            "samples_per_batch": 4,
            "n_batches": 3,
            "n_warmup_batches": 2,
            "min_seconds": 0.5,
            "seed": 1234,
        }
        self.case = {"sampling": self.sampling}
        self.variant = {"dtype": "uint16"}
        self.result = {
            "measurement_passes": 2,
            "scenario": {"sampling": copy.deepcopy(self.sampling)},
            "counters": {"batches_emitted": 6, "reads_issued": 12},
            "derived": {"bytes_per_sample": 1024},
            "timings_ms": {"wall": 1000, "time_to_first_batch": 5},
            "stages": [
                {"name": "io", "input_bytes": 6144, "count": 3, "ms_total": 2000},
                {"name": "decode", "output_bytes": 24576},
            ],
        }

    def test_source_and_returned_byte_rates_are_distinct(self):
        metrics = observation_metrics(self.result, self.case, self.variant)
        self.assertEqual(metrics["useful_bytes"], 12288)
        self.assertEqual(metrics["useful_gib_s"], 12288 / GIB)
        self.assertEqual(metrics["returned_gib_s"], 2 * metrics["useful_gib_s"])
        self.assertEqual(metrics["read_amplification"], 0.5)
        self.assertEqual(metrics["decode_amplification"], 2)

    def test_missing_decode_counter_is_not_zero_amplification(self):
        self.result["stages"][1]["output_bytes"] = 0
        self.assertIsNone(
            observation_metrics(self.result, self.case, self.variant)[
                "decode_amplification"
            ]
        )

    def test_iops_uses_file_requests_and_elapsed_wall_time(self):
        metrics = observation_metrics(self.result, self.case, self.variant)
        self.assertEqual(metrics["read_iops"], 12)
        self.assertEqual(metrics["read_gib_s"], 6144 / GIB)
        self.assertEqual(metrics["mean_read_bytes"], 512)
        self.assertEqual(metrics["reads_per_useful_gib"], 12 * GIB / 12288)

    def test_missing_read_counter_remains_unavailable(self):
        del self.result["counters"]["reads_issued"]
        metrics = observation_metrics(self.result, self.case, self.variant)
        for key in ("read_iops", "mean_read_bytes", "reads_per_useful_gib"):
            self.assertIsNone(metrics[key])
        self.assertEqual(metrics["read_gib_s"], 6144 / GIB)

    def test_invalid_read_counters_are_rejected(self):
        for reads in (0, -1, True, 1.5, math.nan):
            with (
                self.subTest(reads=reads),
                self.assertRaisesRegex(ValueError, "reads_issued"),
            ):
                self.result["counters"]["reads_issued"] = reads
                observation_metrics(self.result, self.case, self.variant)

    def test_old_observations_can_gain_iops_without_changing_measurements(self):
        metrics = observation_metrics(self.result, self.case, self.variant)
        old = {key: value for key, value in metrics.items() if key not in IO_METRICS}
        validate_saved_metrics({"metrics": old}, metrics)
        validate_saved_metrics({"metrics_version": 2, "metrics": metrics}, metrics)
        for row in (
            {"metrics": {**old, "wall_seconds": 2}},
            {
                "metrics": {
                    key: value for key, value in old.items() if key != "useful_bytes"
                }
            },
            {"metrics_version": 2, "metrics": old},
            {"metrics_version": 3, "metrics": metrics},
        ):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_saved_metrics(row, metrics)

    def test_incomplete_run_is_rejected(self):
        self.result["counters"]["batches_emitted"] = 5
        with self.assertRaisesRegex(ValueError, "batch count"):
            observation_metrics(self.result, self.case, self.variant)

    def test_wrong_queries_are_rejected(self):
        self.result["scenario"]["sampling"]["seed"] = 23
        with self.assertRaisesRegex(ValueError, "queries disagree"):
            observation_metrics(self.result, self.case, self.variant)

    def test_short_or_nonfinite_windows_are_rejected(self):
        for wall in (1, 0, -1, math.nan, math.inf):
            with self.subTest(wall=wall), self.assertRaises(ValueError):
                self.result["timings_ms"]["wall"] = wall
                observation_metrics(self.result, self.case, self.variant)

    def cold_result(self):
        self.sampling["n_batches"] = 1
        self.result["scenario"]["sampling"]["n_batches"] = 1
        self.result["scenario"]["cache"] = {"policy": "cold"}
        self.result["counters"]["batches_emitted"] = 2
        self.result["cold_reads"] = {
            "passes": 2,
            "files": 1,
            "pages_per_check": 16,
            "resident_pages_after": 0,
            "device_reads": 4,
            "device_read_bytes": 16384,
            "device_writes": 0,
            "idle_seconds": 0.25,
            "idle_device_reads": 1,
            "idle_device_read_bytes": 4096,
            "idle_device_writes": 0,
            "preparation_ms": 30,
        }

    def test_device_and_application_counters_remain_separate(self):
        self.cold_result()
        metrics = observation_metrics(self.result, self.case, self.variant)
        self.assertEqual(metrics["read_iops"], 12)
        self.assertEqual(metrics["device_iops"], 4)
        self.assertEqual(metrics["device_read_gib_s"], 16384 / GIB)
        self.assertEqual(metrics["device_mean_read_bytes"], 4096)
        self.assertEqual(metrics["idle_device_iops"], 4)
        self.assertEqual(metrics["cache_preparation_seconds"], 0.03)

    def test_unverified_cold_runs_are_rejected(self):
        self.cold_result()
        original = copy.deepcopy(self.result)
        for key, value in (
            ("passes", 1),
            ("resident_pages_after", 1),
            ("device_reads", 0),
            ("device_read_bytes", 0),
            ("idle_seconds", 0),
            ("pages_per_check", 0),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.result = copy.deepcopy(original)
                self.result["cold_reads"][key] = value
                observation_metrics(self.result, self.case, self.variant)
        self.result = original
        del self.result["cold_reads"]
        with self.assertRaisesRegex(ValueError, "checked cold start"):
            observation_metrics(self.result, self.case, self.variant)


class StorageTests(unittest.TestCase):
    def test_partitions_and_arrays_resolve_to_unique_nvme_devices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller = root / "nvme0"
            device = controller / "nvme0n1"
            partition = device / "nvme0n1p1"
            partition.mkdir(parents=True)
            (controller / "transport").write_text("pcie\n")
            (controller / "model").write_text("Test NVMe\n")
            (device / "size").write_text("1024\n")
            (device / "stat").write_text("0 0 0 0 0\n")
            (partition / "partition").write_text("1\n")
            slaves = root / "md0/slaves"
            slaves.mkdir(parents=True)
            (slaves / "part").symlink_to(partition)
            (slaves / "whole").symlink_to(device)
            result = block_devices(root / "md0")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["bytes"], 512 * 1024)
            self.assertEqual(result[0]["model"], "Test NVMe")
            (controller / "transport").write_text("tcp\n")
            with self.assertRaisesRegex(ValueError, "not verified PCIe NVMe"):
                block_devices(root / "md0")


if __name__ == "__main__":
    unittest.main()
