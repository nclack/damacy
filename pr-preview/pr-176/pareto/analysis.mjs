import {storagePerSample} from "./comparisons.mjs?v=74e63ec5e1a3";

export const plottable = row => Number.isFinite(row.compression_fold) && row.compression_fold > 0
  && Number.isFinite(row.metrics.useful_gib_s.median) && row.metrics.useful_gib_s.median > 0;

export const devicePlottable = row => ["device_iops", "device_read_gib_s"].every(key =>
  Number.isFinite(row.metrics[key]?.median) && row.metrics[key].median > 0);

export function dominates(a, b) {
  return a.condition_id === b.condition_id && plottable(a) && plottable(b)
    && a.compression_fold >= b.compression_fold
    && a.metrics.useful_gib_s.median >= b.metrics.useful_gib_s.median
    && (a.compression_fold > b.compression_fold || a.metrics.useful_gib_s.median > b.metrics.useful_gib_s.median);
}

export function frontier(rows) {
  return rows.filter(row => plottable(row) && !rows.some(other => dominates(other, row)))
    .sort((a, b) => a.compression_fold - b.compression_fold);
}

export function matches(row, filters) {
  return (!filters.input || row.input_id === filters.input)
    && (!filters.workload || (row.access_pattern || row.workload) === filters.workload)
    && (!filters.crop || row.crop_size === Number(filters.crop))
    && (!filters.codec || row.setting.codec === filters.codec)
    && (!filters.chunk || (filters.chunk === "sample" ? row.setting.chunk_mode === "sample" : row.setting.chunk_bytes === Number(filters.chunk)))
    && (!filters.block || !row.setting.codec.startsWith("blosc-") || (filters.block === "chunk" ? row.setting.block_mode === "chunk" : row.setting.block_bytes === Number(filters.block)));
}

export function samplesPerSecond(row) {
  if (!row.query_summary?.samples || !row.query_summary?.useful_bytes) return null;
  const factor = row.query_summary.samples * 2 ** 30 / row.query_summary.useful_bytes;
  const metric = row.metrics.useful_gib_s;
  return {median: metric.median * factor, min: metric.min * factor, max: metric.max * factor, n: metric.n};
}

export function cropSeries(rows) {
  const result = new Map();
  for (const row of rows) {
    const key = JSON.stringify([row.input_id, row.backend, row.access_pattern, row.setting.recipe_id]);
    if (!row.crop_size || !row.setting.recipe_id) continue;
    if (!result.has(key)) result.set(key, []);
    result.get(key).push(row);
  }
  return [...result.values()].map(values => values.sort((a, b) => a.crop_size - b.crop_size));
}

export function groups(rows) {
  const result = new Map();
  for (const row of rows) {
    if (!result.has(row.condition_id)) result.set(row.condition_id, []);
    result.get(row.condition_id).push(row);
  }
  return [...result.values()];
}

export function sensitivity(rows, selected) {
  const same = row => row.backend === selected.backend && row.workload === selected.workload
    && ["codec", "chunk_bytes", "block_bytes", "level", "shuffle", "recipe_id"].every(key => row.setting[key] === selected.setting[key]);
  const baseline = rows.find(row => row.input_id === "all-sources" && same(row));
  if (!baseline) return [];
  return rows.filter(row => row.input_id !== "all-sources" && same(row)).map(row => ({
    row,
    compression_percent: 100 * (row.compression_fold / baseline.compression_fold - 1),
    useful_percent: 100 * (row.metrics.useful_gib_s.median / baseline.metrics.useful_gib_s.median - 1),
    raw_relative_percent: row.reference_useful_gib_s && baseline.reference_useful_gib_s
      ? 100 * ((row.metrics.useful_gib_s.median / row.reference_useful_gib_s.median)
        / (baseline.metrics.useful_gib_s.median / baseline.reference_useful_gib_s.median) - 1) : null,
    returned_percent: row.metrics.returned_gib_s && baseline.metrics.returned_gib_s
      ? 100 * (row.metrics.returned_gib_s.median / baseline.metrics.returned_gib_s.median - 1) : null,
  }));
}

export function layoutShapes(setting, key) {
  return setting.sources ? setting.sources.map(source => `${source.input_id}:${source[key].join("x")}`).join("; ")
    : setting[key].join("x");
}

export function csv(rows, context = {}) {
  const columns = ["cache_policy", "host", "input", "backend", "workload", "codec", "chunk_bytes", "block_request_bytes",
    "chunk_shape", "shard_shape", "compression_ratio", "image_throughput_gib_s_median", "image_throughput_gib_s_min",
    "image_throughput_gib_s_max", "repeats", "storage_bandwidth_gib_s_median", "storage_bandwidth_gib_s_min", "storage_bandwidth_gib_s_max",
    "iops_median", "iops_min", "iops_max", "speed_vs_uncompressed", "source_ids", "source_samples_per_pass",
    "distinct_shards_per_pass", "peak_simultaneous_file_reads", "access_pattern", "crop_size", "chunk_mode", "block_mode",
    "samples_per_second_median", "samples_per_second_min", "samples_per_second_max", "source_query_id",
    "storage_bytes_per_sample_median", "storage_bytes_per_sample_min", "storage_bytes_per_sample_max",
    "storage_reads_per_sample_median", "storage_reads_per_sample_min", "storage_reads_per_sample_max", "condition_id", "setting_id"];
  const work = row => context.document ? storagePerSample(context.document, row) : null;
  const values = rows.map(row => [context.cache_policy, context.host, row.input_id, row.backend, row.workload, row.setting.codec,
    row.setting.chunk_bytes, row.setting.block_bytes, layoutShapes(row.setting, "chunk_shape"), layoutShapes(row.setting, "shard_shape"),
    row.compression_fold, row.metrics.useful_gib_s.median, row.metrics.useful_gib_s.min, row.metrics.useful_gib_s.max,
    row.metrics.useful_gib_s.n, row.metrics.device_read_gib_s?.median, row.metrics.device_read_gib_s?.min, row.metrics.device_read_gib_s?.max,
    row.metrics.device_iops?.median, row.metrics.device_iops?.min, row.metrics.device_iops?.max,
    row.reference_useful_gib_s ? row.metrics.useful_gib_s.median / row.reference_useful_gib_s.median : null,
    row.setting.sources?.map(source => source.input_id).join("; ") ?? row.input_id,
    row.query_summary ? JSON.stringify(row.query_summary.source_samples) : null,
    row.query_summary?.distinct_shards, row.metrics.peak_active_reads?.max, row.access_pattern, row.crop_size,
    row.setting.chunk_mode, row.setting.block_mode, samplesPerSecond(row)?.median, samplesPerSecond(row)?.min,
    samplesPerSecond(row)?.max, row.query_summary?.source_query_id,
    work(row)?.bytes.median, work(row)?.bytes.min, work(row)?.bytes.max,
    work(row)?.reads.median, work(row)?.reads.min, work(row)?.reads.max, row.condition_id, row.id]);
  const quote = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
  return [columns, ...values].map(row => row.map(quote).join(",")).join("\r\n") + "\r\n";
}
