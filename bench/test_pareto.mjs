import assert from "node:assert/strict";
import test from "node:test";
import {cropSeries, csv, devicePlottable, frontier, matches, samplesPerSecond, sensitivity} from "../docs/pareto/analysis.mjs";

function point(id, fold, rate, condition = "cpu") {
  return {id, condition_id: condition, input_id: "image", backend: condition, workload: "crop",
    compression_fold: fold, setting: {codec: "zstd", chunk_bytes: 65536, block_bytes: null,
      chunk_shape: [1, 128, 256], shard_shape: [1, 1024, 1024]},
    metrics: {useful_gib_s: {median: rate, min: rate * .9, max: rate * 1.1, n: 3}}};
}

test("crop comparisons preserve access, encoding, and backend groups", () => {
  const a = {...point("128", 1, 2), crop_size: 128, access_pattern: "random"};
  a.setting.recipe_id = "zstd";
  const b = structuredClone(a);
  b.crop_size = 256;
  b.id = "256";
  b.condition_id = "cpu-256";
  const baked = structuredClone(a);
  baked.access_pattern = "baked";
  baked.setting.chunk_mode = "sample";
  baked.setting.chunk_bytes = null;
  baked.setting.block_mode = "chunk";
  baked.setting.block_bytes = null;
  baked.setting.codec = "blosc-zstd";
  assert(matches(a, {workload: "random", crop: "128"}));
  assert(!matches(b, {workload: "random", crop: "128"}));
  assert(!matches(baked, {workload: "random"}));
  assert(matches(baked, {chunk: "sample", block: "chunk"}));
  assert(!matches(a, {chunk: "sample"}));
  assert.deepEqual(cropSeries([b, baked, a]).map(series => series.map(row => row.crop_size)), [[128, 256], [128]]);
  assert.equal(frontier([a, b]).length, 2);
});

test("sample rates convert image-byte rates using the exact mixed query composition", () => {
  const row = point("mixed", 2, 3);
  row.query_summary = {samples: 7 * 4096, useful_bytes: 3.75 * 2 ** 30};
  const metric = samplesPerSecond(row);
  assert.equal(metric.median, 3 * 7 * 4096 / 3.75);
  assert.equal(metric.n, 3);
  assert(Math.abs(metric.min / metric.median - .9) < 1e-12);
  assert(Math.abs(metric.max / metric.median - 1.1) < 1e-12);
  assert.equal(samplesPerSecond(point("old", 1, 1)), null);
});

test("joint frontier keeps codec and size tradeoffs", () => {
  const rows = [point("fast", 1, 10), point("small", 5, 2), point("balanced", 3, 5), point("dominated", 2, 4)];
  assert.deepEqual(frontier(rows).map(row => row.id), ["fast", "balanced", "small"]);
});

test("different measurement conditions never dominate each other", () => {
  const rows = [point("cpu", 1, 1), point("gpu", 10, 10, "cuda")];
  assert.equal(frontier(rows).length, 2);
});

test("ties remain observable and invalid objectives are excluded", () => {
  const rows = [point("a", .5, 2), point("b", .5, 2), point("invalid", 0, 10), point("missing", 2, NaN)];
  assert.deepEqual(frontier(rows).map(row => row.id), ["a", "b"]);
});

test("Blosc block filters preserve raw controls", () => {
  const raw = point("raw", 2, 2), blosc = point("blosc", 3, 3);
  blosc.setting.codec = "blosc-zstd";
  blosc.setting.block_bytes = 65536;
  assert(matches(raw, {block: "16384"}));
  assert(!matches(blosc, {block: "16384"}));
  assert(matches(blosc, {block: "65536"}));
  blosc.setting.codec = "blosc-lz4";
  assert(!matches(blosc, {block: "16384"}));
});

test("CSV keeps exact values and measurement identities", () => {
  const row = point("exact", 2.123456789012345, 4.123456789012345);
  row.input_id = 'quoted "image"';
  const output = csv([row]);
  assert(output.includes('"2.123456789012345"'));
  assert(output.includes('"4.123456789012345"'));
  assert(output.includes('"quoted ""image"""'));
  assert(output.includes('"cpu","exact"'));
});

test("application and device counters cannot substitute for each other", () => {
  const row = point("reads", 2, 4);
  assert(!devicePlottable(row));
  row.metrics.read_iops = {median: 12345.678901234};
  row.metrics.read_gib_s = {median: 1.2345678901234};
  assert(!devicePlottable(row));
  const output = csv([row]);
  assert(!output.includes('"12345.678901234"'));
  assert(!output.includes('"1.2345678901234"'));
  assert(output.includes('"storage_bandwidth_gib_s_median"'));
  assert(output.includes('"iops_median"'));
  assert(!output.includes('"read_iops_median"'));
  row.metrics.device_iops = {median: 17};
  row.metrics.device_read_gib_s = {median: 1};
  assert(devicePlottable(row));
  const measured = csv([row]);
  assert(measured.includes('"17"'));
  assert(measured.includes('"1"'));
  row.metrics.device_iops.median = NaN;
  assert(!devicePlottable(row));
});

test("sensitivity only compares matching settings and backends", () => {
  const baseline = point("all", 2, 10);
  baseline.input_id = "all-sources";
  baseline.metrics.returned_gib_s = {median: 20};
  baseline.reference_useful_gib_s = {median: 10};
  const omission = structuredClone(baseline);
  omission.id = "omit";
  omission.input_id = "without-source";
  omission.condition_id = "omitted";
  omission.compression_fold = 1.6;
  omission.metrics.useful_gib_s.median = 8;
  omission.metrics.returned_gib_s.median = 22;
  omission.reference_useful_gib_s.median = 16;
  const unrelated = structuredClone(omission);
  unrelated.backend = "cuda";
  const values = sensitivity([baseline, omission, unrelated], omission);
  assert.equal(values.length, 1);
  assert(Math.abs(values[0].compression_percent + 20) < 1e-12);
  assert.equal(values[0].raw_relative_percent, -50);
  assert(Math.abs(values[0].useful_percent + 20) < 1e-12);
  assert(Math.abs(values[0].returned_percent - 10) < 1e-12);
  assert.equal(frontier([baseline, omission]).length, 2);
});
