"""Check pairing, unrestricted sampling, and per-process baked file budgets."""

import copy
import math
import unittest
from collections import defaultdict
from pathlib import Path

from pareto import (
    DTYPES,
    GIB,
    case_variant,
    make_plan,
    read_json,
    scenario_for,
    schedule,
)
from pareto_crops import baked_origins, random_origins
from pareto_mixture import query_records
from test_pareto_mixture import mixture_fixture


class CropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.corpus = mixture_fixture()
        cls.definition = read_json(Path(__file__).parent / "studies/microscopy.json")
        cls.plan = make_plan(cls.definition, cls.corpus)

    def test_reduced_matrix_retains_small_chunk_check(self):
        self.assertEqual(len(self.plan["cohorts"]), 1)
        self.assertEqual(len(self.plan["cases"]), 62)
        self.assertEqual(len(schedule(self.plan)), 258)
        small = [c for c in self.plan["cases"] if c["setting"]["chunk_bytes"] == 16384]
        self.assertEqual(len(small), 2)
        self.assertEqual({c["workload"] for c in small}, {"random-128"})
        for query in self.plan["query_sets"]:
            self.assertEqual(len(query["input_ids"]), 7)

    def test_random_translations_can_touch_every_origin_and_repeat(self):
        source = {"id": "small", "shape": [2, 3, 3]}
        origins = random_origins(source, 7, "small", 1000, [1, 2, 2])
        self.assertEqual(
            set(origins),
            {(z, y, x) for z in range(2) for y in range(2) for x in range(2)},
        )
        self.assertLess(len(set(origins)), len(origins))
        with self.assertRaisesRegex(ValueError, "fit"):
            random_origins(source, 7, "small", 1, [1, 4, 2])

    def test_paired_queries_cross_shards_and_baked_reads_move_forward(self):
        for crop in self.definition["crop_sizes"]:
            random = next(
                q for q in self.plan["query_sets"] if q["workload"] == f"random-{crop}"
            )
            baked = next(
                q for q in self.plan["query_sets"] if q["workload"] == f"baked-{crop}"
            )
            self.assertGreater(random["summary"]["samples_crossing_shards"], 0)
            self.assertEqual(baked["summary"]["samples_crossing_shards"], 0)
            self.assertEqual(baked["source_query_id"], random["id"])
            self.assertEqual(baked["source_query_sha256"], random["summary"]["sha256"])
            self.assertEqual(
                baked["summary"]["useful_bytes"], random["summary"]["useful_bytes"]
            )
            self.assertEqual(baked["summary"]["distinct_shards"], 16)
            self.assertEqual(baked["summary"]["shards_per_window"]["batch"]["min"], 16)
            self.assertEqual(
                [r[0] for r in query_records(baked)],
                [r[0] for r in query_records(random)],
            )
            for source in baked["sources"]:
                lanes = defaultdict(list)
                depth = source["shard_shape"][0]
                for z, y, x in baked_origins(source, baked["samples_per_source"]):
                    self.assertEqual((y, x), (0, 0))
                    lanes[z // depth].append(z % depth)
                for positions in lanes.values():
                    self.assertEqual(positions, list(range(len(positions))))
                self.assertLessEqual(
                    max(map(len, lanes.values())) - min(map(len, lanes.values())), 1
                )

    def test_shards_are_near_one_gib_and_chunks_hold_one_sample(self):
        variants = {v["id"]: v for v in self.plan["variants"]}
        for case in self.plan["cases"]:
            if case["access_pattern"] != "baked":
                continue
            total = 0
            for key in case["variant_ids"]:
                variant = variants[key]
                bpe = DTYPES[variant["dtype"]][1]
                size = math.prod(variant["shard_shape"]) * bpe
                self.assertGreater(size, 0.8 * GIB)
                self.assertLess(size, 1.2 * GIB)
                total += variant["shape"][0] // variant["shard_shape"][0]
                self.assertEqual(
                    variant["chunk_shape"], case["sampling"]["sample_shape"]
                )
                self.assertEqual(
                    variant["chunk_bytes"], math.prod(variant["chunk_shape"]) * bpe
                )
            self.assertEqual(total, 16)
            aggregate = case_variant(case, variants)
            self.assertIsNone(aggregate["chunk_bytes"])
            self.assertEqual(aggregate["chunk_mode"], "sample")

    def test_byte_budgets_and_controls_match_each_condition(self):
        for case in self.plan["cases"]:
            sample_bytes = math.prod(case["sampling"]["sample_shape"]) * 4
            self.assertEqual(
                case["sampling"]["samples_per_batch"] * sample_bytes,
                self.definition["batch_output_bytes"],
            )
            self.assertEqual(
                case["pipeline"]["lookahead_samples"] * sample_bytes,
                self.definition["lookahead_output_bytes"],
            )
            scenario = scenario_for(
                {**self.plan, "cache_policy": "preloaded files"}, case, "/tmp/data"
            )
            self.assertEqual(
                scenario["pipeline"]["lookahead_samples"],
                case["pipeline"]["lookahead_samples"],
            )
        cases = {c["id"]: c for c in self.plan["cases"]}
        for task in schedule(self.plan):
            if task["role"] != "sample":
                self.assertEqual(cases[task["case_id"]]["setting"]["codec"], "none")

    def test_random_layouts_reuse_verified_v2_arrays(self):
        definition, corpus = mixture_fixture()
        old = make_plan(definition, corpus, pilot=True)
        old_variants = {v["id"]: v for v in old["variants"]}
        for variant in self.plan["variants"]:
            if "baked" not in variant:
                self.assertEqual(variant, old_variants[variant["id"]])

    def test_invalid_budgets_and_changed_recipe_fail(self):
        definition = copy.deepcopy(self.definition)
        definition["batch_output_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "whole samples"):
            make_plan(definition, self.corpus)
        variants = {v["id"]: copy.deepcopy(v) for v in self.plan["variants"]}
        case = next(c for c in self.plan["cases"] if c["access_pattern"] == "baked")
        variants[case["variant_ids"][0]]["chunk_shape"] = [2, 128, 128]
        with self.assertRaisesRegex(ValueError, "one sample"):
            case_variant(case, variants)

    def test_insufficient_caches_fail_before_preparation(self):
        for key in (
            "n_array_meta_cache",
            "n_chunk_layout_cache",
            "n_shard_index_cache",
        ):
            with self.subTest(cache=key):
                definition = copy.deepcopy(self.definition)
                smallest_crop = min(definition["crop_sizes"])
                required = (
                    definition["lookahead_output_bytes"]
                    + 2 * definition["batch_output_bytes"]
                ) // (smallest_crop**2 * 4)
                if key == "n_shard_index_cache":
                    required *= definition["pipeline"]["max_shards_per_sample"]
                definition["pipeline"][key] = required - 1
                with self.assertRaisesRegex(ValueError, key):
                    make_plan(definition, self.corpus)


if __name__ == "__main__":
    unittest.main()
