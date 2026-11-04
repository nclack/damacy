export const plottable = row => Number.isFinite(row.compression_fold) && row.compression_fold > 0
  && Number.isFinite(row.metrics.useful_gib_s.median) && row.metrics.useful_gib_s.median > 0;

export const ioPlottable = row => ["read_iops", "read_gib_s"].every(key =>
  Number.isFinite(row.metrics[key]?.median) && row.metrics[key].median > 0);

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
    && (!filters.workload || row.workload === filters.workload)
    && (!filters.codec || row.setting.codec === filters.codec)
    && (!filters.chunk || row.setting.chunk_bytes === Number(filters.chunk))
    && (!filters.block || row.setting.codec !== "blosc-zstd" || row.setting.block_bytes === Number(filters.block));
}

export function groups(rows) {
  const result = new Map();
  for (const row of rows) {
    if (!result.has(row.condition_id)) result.set(row.condition_id, []);
    result.get(row.condition_id).push(row);
  }
  return [...result.values()];
}

export function csv(rows, context = {}) {
  const columns = ["cache_policy", "host", "input", "backend", "workload", "codec", "chunk_bytes", "block_request_bytes",
    "chunk_shape", "shard_shape", "compression_fold", "useful_gib_s_median", "useful_gib_s_min",
    "useful_gib_s_max", "repeats", "read_amplification", "decode_amplification", "reference_spread_percent",
    "read_iops_median", "read_iops_min", "read_iops_max", "read_gib_s_median", "read_gib_s_min", "read_gib_s_max",
    "mean_read_bytes_median", "reads_per_useful_gib_median", "device_iops_median", "device_iops_min", "device_iops_max",
    "device_read_gib_s_median", "device_read_gib_s_min", "device_read_gib_s_max", "device_mean_read_bytes_median",
    "idle_device_iops_median", "idle_device_gib_s_median", "condition_id", "setting_id"];
  const values = rows.map(row => [context.cache_policy, context.host, row.input_id, row.backend, row.workload, row.setting.codec,
    row.setting.chunk_bytes, row.setting.block_bytes, row.setting.chunk_shape.join("x"), row.setting.shard_shape.join("x"),
    row.compression_fold, row.metrics.useful_gib_s.median, row.metrics.useful_gib_s.min, row.metrics.useful_gib_s.max,
    row.metrics.useful_gib_s.n, row.metrics.read_amplification?.median, row.metrics.decode_amplification?.median,
    row.reference_spread_percent, row.metrics.read_iops?.median, row.metrics.read_iops?.min, row.metrics.read_iops?.max,
    row.metrics.read_gib_s?.median, row.metrics.read_gib_s?.min, row.metrics.read_gib_s?.max,
    row.metrics.mean_read_bytes?.median, row.metrics.reads_per_useful_gib?.median,
    row.metrics.device_iops?.median, row.metrics.device_iops?.min, row.metrics.device_iops?.max,
    row.metrics.device_read_gib_s?.median, row.metrics.device_read_gib_s?.min, row.metrics.device_read_gib_s?.max,
    row.metrics.device_mean_read_bytes?.median, row.metrics.idle_device_iops?.median, row.metrics.idle_device_gib_s?.median,
    row.condition_id, row.id]);
  const quote = value => `"${String(value ?? "").replaceAll('"', '""')}"`;
  return [columns, ...values].map(row => row.map(quote).join(",")).join("\r\n") + "\r\n";
}
