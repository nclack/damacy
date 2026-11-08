export const cropSizes = [128, 192, 256];
export const codecNames = {none: "Uncompressed", zstd: "Zstd", "blosc-zstd": "Blosc Zstd", "blosc-lz4": "Blosc LZ4"};
export const bytesLabel = value => value >= 2 ** 20 ? `${value / 2 ** 20} MiB` : `${value / 1024} KiB`;

export function metricStrings(metric, digits = 3) {
  const format = value => Number.isFinite(value) ? value.toLocaleString("en", {maximumSignificantDigits: digits}) : "—";
  while (Number.isFinite(metric.min) && Number.isFinite(metric.max) && metric.min !== metric.max
    && format(metric.min) === format(metric.max) && digits < 9) digits++;
  return Object.fromEntries(["median", "min", "max"].map(key => [key, format(metric[key])]));
}

export function recipeLabel(row) {
  const setting = row.setting;
  const chunk = setting.chunk_mode === "sample" ? "one sample/chunk" : `${bytesLabel(setting.chunk_bytes)} chunks`;
  const block = setting.codec.startsWith("blosc-")
    ? ` · bitshuffle · ${setting.block_mode === "chunk" ? "whole-chunk" : bytesLabel(setting.block_bytes)} blocks` : "";
  return `${codecNames[setting.codec]} · ${chunk}${block}`;
}

export function readers(documents) {
  return documents.filter(document => document.baked).flatMap(document => {
    const backends = [...new Set(document.rows.map(row => row.backend))].sort();
    const epyc = /EPYC/.test(document.context.machine.cpu?.["Model name"] || "");
    const h100 = /H100/.test(document.context.machine.gpu || "");
    return backends.map(backend => ({
      id: `${document.id}:${backend}`, document, backend,
      label: epyc ? "EPYC CPU" : h100 ? backend === "cpu" ? "H100-node CPU" : "H100 CUDA" : `${document.context.machine.host} · ${backend.toUpperCase()}`,
      output: backend === "cpu" ? "Output in system RAM" : "Output in GPU memory",
      order: epyc ? 0 : backend === "cpu" ? 1 : 2,
      rows: document.rows.filter(row => row.backend === backend && row.input_id === "all-sources"),
    }));
  }).sort((a, b) => a.order - b.order || a.id.localeCompare(b.id));
}

export function bestRows(reader, crop, access) {
  const rows = reader.rows.filter(row => row.crop_size === crop && row.access_pattern === access);
  return rows.slice().sort((a, b) => b.metrics.useful_gib_s.median - a.metrics.useful_gib_s.median || a.id.localeCompare(b.id));
}

function family(row) {
  const setting = row.setting;
  return JSON.stringify([setting.codec, setting.level, setting.shuffle, setting.block_mode, setting.block_bytes]);
}

export function fixedRecipes(readerList) {
  if (!readerList.length) return [];
  const complete = reader => {
    const byRecipe = new Map();
    for (const row of reader.rows) {
      const key = `${row.access_pattern}:${row.setting.recipe_id}`;
      if (!byRecipe.has(key)) byRecipe.set(key, []);
      byRecipe.get(key).push(row);
    }
    return [...byRecipe.values()].filter(rows => rows.length === cropSizes.length
      && cropSizes.every(crop => rows.some(row => row.crop_size === crop)));
  };
  const available = readerList.map(complete);
  const choices = [];
  for (const random of available[0].filter(rows => rows[0].access_pattern === "random")) {
    const baked = available[0].find(rows => rows[0].access_pattern === "baked" && family(rows[0]) === family(random[0]));
    if (!baked) continue;
    const ids = [random[0].setting.recipe_id, baked[0].setting.recipe_id];
    if (!available.every(series => ids.every(id => series.some(rows => rows[0].setting.recipe_id === id)))) continue;
    const setting = random[0].setting;
    choices.push({id: ids.join(":"), random: ids[0], baked: ids[1],
      label: `${codecNames[setting.codec]}${setting.codec.startsWith("blosc-") ? ` · ${setting.block_mode === "chunk" ? "whole-chunk" : bytesLabel(setting.block_bytes)} blocks` : ""}`,
      randomRow: random[0], bakedRow: baked[0]});
  }
  return choices.sort((a, b) => (a.randomRow.setting.codec !== "none") - (b.randomRow.setting.codec !== "none") || a.label.localeCompare(b.label));
}

export function comparableRows(readerList, recipe, access) {
  if (!recipe) return [];
  const series = readerList.map(reader => ({reader,
    rows: reader.rows.filter(row => row.access_pattern === access && row.setting.recipe_id === recipe[access])
      .sort((a, b) => a.crop_size - b.crop_size)}));
  return series.map(({reader, rows}) => ({reader, rows: rows.filter(row => series.every(other => {
    const paired = other.rows.find(candidate => candidate.crop_size === row.crop_size);
    return paired && paired.sampling.query_sha256 === row.sampling.query_sha256
      && paired.setting.input_sha256 === row.setting.input_sha256;
  }))}));
}

export function observed(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = Math.floor(sorted.length / 2);
  return {median: sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2,
    min: sorted[0], max: sorted.at(-1), n: sorted.length};
}

export function storagePerSample(document, row) {
  const observations = document.observations.filter(value => row.observation_ids.includes(value.task.id));
  const values = observations.map(value => {
    const samples = value.metrics.passes * row.query_summary?.samples;
    const reads = value.result.cold_reads;
    return samples > 0 && reads ? {bytes: reads.device_read_bytes / samples, reads: reads.device_reads / samples} : null;
  });
  if (!values.length || values.some(value => !value || !Number.isFinite(value.bytes) || !Number.isFinite(value.reads))) return null;
  return {bytes: observed(values.map(value => value.bytes)), reads: observed(values.map(value => value.reads))};
}
