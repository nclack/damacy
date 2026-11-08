import {cropSeries, csv, devicePlottable, frontier, groups, matches, plottable, samplesPerSecond, sensitivity} from "./analysis.mjs?v=dc1b0eb38b2f";
import {mountOverview} from "./overview.js?v=dc1b0eb38b2f";
import {metricStrings, storagePerSample} from "./comparisons.mjs?v=dc1b0eb38b2f";
import {storagePanel} from "./storage.js?v=dc1b0eb38b2f";

const $ = id => document.getElementById(id);
const number = value => Number.isFinite(value) ? value.toLocaleString(undefined, {maximumSignificantDigits: 4}) : "—";
const kib = bytes => bytes == null ? "—" : `${number(bytes / 1024)} KiB`;
const chunkLabel = setting => setting.chunk_mode === "sample" ? "One sample" : kib(setting.chunk_bytes);
const blockLabel = setting => setting.block_mode === "chunk" ? "Whole chunk" : kib(setting.block_bytes);
const range = metric => {
  if (!metric) return "Not recorded";
  const value = metricStrings(metric, 4);
  return `${value.min}–${value.max}`;
};
const colors = {none: "var(--gray)", zstd: "var(--blue)", "blosc-zstd": "var(--orange)", "blosc-lz4": "var(--green)"};
const codecs = {none: "Uncompressed", zstd: "Zstd", "blosc-zstd": "Blosc Zstd", "blosc-lz4": "Blosc LZ4"};
const workloads = {random: "Random crops", baked: "Baked samples · sequential", "balanced-256": "Shard-balanced 256 × 256 crops", "random-256": "Random 256 × 256 crops", "whole-planes": "Whole-plane reads"};
const views = {pareto: "Image throughput and compression", device: "Storage reads", crop: "Crop size and samples/s"};
const saved = new URLSearchParams(location.hash.slice(1));
const openExplorer = location.hash === "#explorer" || (saved.has("explorer") ? saved.get("explorer") === "open" : saved.has("study"));
if (saved.get("view") === "io") saved.set("view", "device");
let study, entries, selected, filtered = [], generation = 0;
const documents = new Map();

function element(tag, text) {
  const item = document.createElement(tag);
  if (text !== undefined) item.textContent = text;
  return item;
}

function options(id, values, label = value => value, all = false) {
  const select = $(id);
  const previous = saved.get(id) ?? select.value;
  select.replaceChildren();
  if (all) select.append(new Option("All", ""));
  for (const value of values) select.append(new Option(label(value), value));
  if ([...select.options].some(option => option.value === previous)) select.value = previous;
  saved.delete(id);
}

async function json(url) {
  const response = await fetch(url, {cache: "no-cache"});
  if (!response.ok) throw new Error(`Could not load measurements (${response.status}).`);
  return response.json();
}

function fields(target, rows) {
  const list = element("dl");
  for (const [name, value, description] of rows) {
    const cell = element("dd");
    if (value instanceof Node) cell.append(value);
    else cell.textContent = value;
    list.append(element("dt", name), cell);
    if (description) {
      const definition = element("dd", description);
      definition.className = "measurement-definition";
      list.append(definition);
    }
  }
  target.append(list);
}

function configurationName(row) {
  return `${codecs[row.setting.codec]} · ${row.backend.toUpperCase()} · ${chunkLabel(row.setting)} chunk`
    + (row.crop_size ? ` · ${row.crop_size} × ${row.crop_size} crop` : "")
    + (row.setting.block_bytes == null && row.setting.block_mode !== "chunk" ? "" : ` · ${blockLabel(row.setting)} block`);
}

function measurementValue(metric, unit = "") {
  const value = element("span", metric ? `${metricStrings(metric, 4).median}${unit}` : "Not recorded");
  if (metric) {
    const limits = element("span", range(metric));
    limits.className = "observed-range";
    value.append(limits);
  }
  return value;
}

function select(row, announce = true) {
  selected = row.id;
  const target = $("detail-content");
  const diagnosticsOpen = target.querySelector(".diagnostics")?.open ?? false;
  target.replaceChildren(element("p", `${codecs[row.setting.codec]} · ${row.backend.toUpperCase()}`));
  const setting = row.setting, metrics = row.metrics;
  const summary = element("section");
  summary.className = "primary-metrics";
  summary.setAttribute("aria-label", "Measurements");
  summary.append(element("p", `Medians and observed ranges across ${metrics.useful_gib_s.n} process repetitions. Compression ratio is fixed by the stored files.`));
  fields(summary, [
    ["Image throughput", measurementValue(metrics.useful_gib_s, " GiB/s"),
      "Requested pixel bytes delivered per second, in the source data type."],
    ["Compression ratio", `${number(row.compression_fold)}×`,
      "Uncompressed image bytes divided by stored file bytes."],
    ["Storage bandwidth", measurementValue(metrics.device_read_gib_s, " GiB/s"),
      "Bytes read across the storage system’s drives per second, including extra reads."],
    ["IOPS", measurementValue(metrics.device_iops, " reads/s"),
      "Read operations completed by the storage system per second."],
  ]);
  if (row.crop_size) fields(summary, [["Samples/s", measurementValue(samplesPerSecond(row)),
    `Completed ${row.crop_size} × ${row.crop_size} single-plane crops per second. This is the same image throughput expressed as a sample count.`]]);
  const work = storagePerSample(study, row);
  if (work) fields(summary, [
    ["Storage bytes / sample", measurementValue(Object.fromEntries(Object.entries(work.bytes).map(([key, value]) => [key, key === "n" ? value : value / 1024])), " KiB"),
      "Storage bytes divided by completed crops in each execution; lower means less work for the same crop size."],
    ["Storage reads / sample", measurementValue(work.reads),
      "Completed storage operations divided by completed crops. One operation can serve multiple samples."],
  ]);
  target.append(summary, element("h3", "Layout"));
  fields(target, [
    ["Chunk", chunkLabel(setting)],
    ["Chunk shape", setting.sources ? "Per source, below" : setting.chunk_shape.join(" × ")],
    ["Shard shape", setting.sources ? "Per source, below" : setting.shard_shape.join(" × ")],
    ["Blosc block", blockLabel(setting)],
    ["Compression level", setting.level || "—"],
    ["Source type", setting.dtype],
    ["Sample shape", row.sampling.sample_shape.join(" × ")],
  ]);
  if (row.query_summary) {
    const query = row.query_summary;
    target.append(element("h3", "Query coverage"));
    fields(target, [
      ["Samples per pass", number(query.samples)],
      ["Image bytes per pass", `${number(query.useful_bytes / 2 ** 30)} GiB`],
      ["Shard files per pass", number(query.distinct_shards)],
      ["Planned files per batch", range(query.shards_per_window.batch)],
      [`Planned files per ${row.pipeline?.lookahead_samples ?? study.pipeline.lookahead_samples}-query block`, range(query.shards_per_window.lookahead)],
      ["Peak simultaneous file reads", metrics.peak_active_reads ? number(metrics.peak_active_reads.max) : "Not recorded"],
    ]);
    target.append(element("p", "Planned shard coverage counts distinct files. Simultaneous file reads can include multiple reads from the same file."));
    const link = element("a", "Download retained query stream");
    link.href = `data/queries/${row.sampling.query_id}.bin`;
    target.append(link);
    if (query.samples_crossing_shards !== undefined) fields(target, [["Crops crossing shards", `${number(query.samples_crossing_shards)} / ${number(query.samples)}`]]);
    if (query.source_query_id) {
      const coordinates = element("a", "Download paired source coordinates");
      coordinates.href = `data/queries/${query.source_query_id}.bin`;
      target.append(element("p", "Baked samples preserve these random crops and native pixel types. Each pass reads forward through a prefix of every file; it does not scan the full baked corpus."), coordinates);
    }
    const comparisons = sensitivity(study.rows, row);
    if (comparisons.length) {
      const baseline = study.rows.find(value => value.input_id === "all-sources"
        && value.backend === row.backend && value.workload === row.workload
        && ["codec", "chunk_bytes", "block_bytes", "level", "shuffle", "recipe_id"].every(key => value.setting[key] === row.setting[key]));
      target.append(element("h3", "Leave-one-out sensitivity"));
      target.append(element("p", `Speed vs uncompressed divides image throughput by the same mixture’s uncompressed control${study.reference_setting?.chunk_bytes ? ` (${kib(study.reference_setting.chunk_bytes)} chunks)` : ""}. A value of 1.2× means 20% faster. This accounts for source-type differences; removing a collection also changes file count.`));
      for (const comparison of [baseline, ...comparisons.map(value => value.row)]) {
        const group = element("section");
        group.className = "sensitivity-comparison";
        group.dataset.input = comparison.input_id;
        group.append(element("h4", study.cohorts.find(cohort => cohort.id === comparison.input_id).name));
        fields(group, [
          ["Compression ratio", `${number(comparison.compression_fold)}×`],
          ["Speed vs uncompressed", comparison.reference_useful_gib_s
            ? `${number(comparison.metrics.useful_gib_s.median / comparison.reference_useful_gib_s.median)}×` : "Not recorded"],
        ]);
        target.append(group);
      }
      target.append(element("p", "These are comparisons of medians, not significance tests."));
    }
  }
  const diagnostics = element("details");
  diagnostics.className = "diagnostics";
  diagnostics.open = diagnosticsOpen;
  diagnostics.append(element("summary", "Diagnostics"));
  diagnostics.append(element("h3", "Reader requests"));
  diagnostics.append(element("p", "File read requests are the byte ranges Damacy reads before decompression, after adjacent ranges are combined. They exclude metadata. The OS can cache, split, combine, or read ahead of these requests, so their counts and bytes differ from storage measurements."));
  fields(diagnostics, [
    ["File read requests/s", number(metrics.read_iops?.median)],
    ["File read bandwidth", measurementValue(metrics.read_gib_s, " GiB/s")],
    ["Mean file read size", kib(metrics.mean_read_bytes?.median)],
    ["File bytes / image bytes", `${number(metrics.read_amplification?.median)}×`],
    ["Decoded bytes / image bytes", metrics.decode_amplification ? `${number(metrics.decode_amplification.median)}×` : "Not recorded"],
    ["Output type", study.pipeline.dtype],
    ["Blocks in headers", setting.actual_block_bytes.map(kib).join(", ") || "—"],
    ["Initial first batch", `${number(metrics.first_batch_ms.median)} ms`],
    ["Uncompressed control variation", `${number(row.reference_spread_percent)}%`],
    ["Stored file bytes", number(setting.stored_bytes)],
    ["Image bytes in stored arrays", number(setting.logical_bytes)],
  ]);
  if (metrics.device_iops) {
    diagnostics.append(element("h3", "Storage checks"));
    diagnostics.append(element("p", "Idle measurements are taken beside each execution and are not subtracted. Storage counters cover all activity on the selected drives."));
    fields(diagnostics, [
      ["Mean storage read size", kib(metrics.device_mean_read_bytes.median)],
      ["Storage bytes / image bytes", `${number(metrics.device_read_amplification.median)}×`],
      ["Idle IOPS", number(metrics.idle_device_iops.median)],
      ["Idle storage bandwidth", `${number(metrics.idle_device_gib_s.median)} GiB/s`],
      ["Cache clearing (excluded)", `${number(metrics.cache_preparation_seconds.median)} s`],
    ]);
  }
  diagnostics.append(element("h3", "Stage timings"));
  diagnostics.append(element("p", "Execution nearest the image throughput median. Stage timings overlap and cannot be added together."));
  const plottedMetric = "useful_gib_s";
  const observations = study.observations.filter(value => row.observation_ids.includes(value.task.id));
  observations.sort((a, b) => Math.abs(a.metrics[plottedMetric] - metrics[plottedMetric]?.median)
    - Math.abs(b.metrics[plottedMetric] - metrics[plottedMetric]?.median));
  fields(diagnostics, observations[0].result.stages.map(stage => [stage.name.replaceAll("_", " "), `${number(stage.ms_total)} ms`]));
  const metadata = observations[0].result.metadata;
  if (metadata) {
    diagnostics.append(element("h3", "Metadata requests"));
    const counts = [
      ["Startup read operations", number(metadata.startup.read)],
      ["Measured read operations", number(metadata.measured.read)],
    ];
    if (metadata.version === 2) counts.push(
      ["Startup Zarr metadata misses", number(metadata.startup.array_metadata_misses)],
      ["Measured Zarr metadata misses", number(metadata.measured.array_metadata_misses)],
      ["Startup shard-index misses", number(metadata.startup.shard_index_misses)],
      ["Measured shard-index misses", number(metadata.measured.shard_index_misses)],
    );
    fields(diagnostics, counts);
  }
  target.append(element("h3", "Input and source"));
  const sources = setting.sources || [setting];
  for (const source of sources) {
    const input = study.inputs.find(value => value.id === (source.input_id || row.input_id));
    const group = setting.sources ? element("details") : target;
    if (setting.sources) {
      group.className = "source-details";
      group.open = source === sources[0];
      group.append(element("summary", input.name));
      target.append(group);
    }
    const link = element("a", setting.sources ? "Source attribution" : input.name);
    if (/^https?:\/\//.test(input.source.url)) link.href = input.source.url;
    group.append(link, element("p", input.source.attribution), element("p", `License: ${input.source.license}`));
    if (setting.sources) fields(group, [
      ["Original shape", source.source_shape.join(" × ")],
      [source.baked ? "Baked shape" : "Expanded shape", source.shape.join(" × ")],
      ["Source type", source.dtype],
      ["Chunk shape", source.chunk_shape.join(" × ")],
      ["Chunk bytes", kib(source.chunk_bytes)],
      ["Shard shape", source.shard_shape.join(" × ")],
      ["Shard files", source.shard_files],
      ["Stored shard range", `${number(Math.min(...source.shard_file_bytes) / 2 ** 30)}–${number(Math.max(...source.shard_file_bytes) / 2 ** 30)} GiB`],
      ["Input SHA256", source.input_sha256],
    ]);
  }
  diagnostics.append(element("h3", "Provenance"));
  fields(diagnostics, [["Revision", `${study.context.source.revision?.slice(0, 12) || "Unknown"}${study.context.source.local_changes ? " + local changes" : ""}`],
    ["Binary SHA256", study.context.binary_sha256], ["Input SHA256", setting.input_sha256]]);
  if (row.sampling.query_sha256) fields(diagnostics, [["Query SHA256", row.sampling.query_sha256]]);
  target.append(diagnostics);
  d3.selectAll("circle.point").attr("stroke-width", point => point.id === selected ? 3 : 1.5)
    .attr("aria-pressed", point => point.id === selected);
  for (const button of $("rows").querySelectorAll("button")) button.setAttribute("aria-pressed", button.dataset.setting === selected);
  for (const button of $("plots").querySelectorAll("button")) button.setAttribute("aria-pressed", button.dataset.setting === selected);
  $("selection-details").hidden = false;
  const message = `${configurationName(row)} selected. ${number(metrics.useful_gib_s.median)} GiB/s image throughput, ${number(row.compression_fold)}× compression ratio, ${metrics.useful_gib_s.n} process repetitions.`;
  if ($("selection-status").textContent !== message) $("selection-status").textContent = message;
  if (announce) {
    const hash = new URLSearchParams(location.hash.slice(1));
    hash.set("setting", row.id);
    history.replaceState(null, "", `#${hash}`);
  }
}

function panel(rows, scales, isIO, isCrop = false) {
  const first = rows[0], wrapper = element("section");
  wrapper.className = "panel";
  const backend = first.backend === "cpu" ? "CPU" : "CUDA";
  wrapper.append(element("h2", backend));
  wrapper.append(element("p", first.backend === "cpu" ? "Output: system RAM." : "Output: GPU memory."));
  $("plots").append(wrapper);
  const padding = parseFloat(getComputedStyle(wrapper).paddingLeft) + parseFloat(getComputedStyle(wrapper).paddingRight);
  const width = Math.max(240, wrapper.clientWidth - padding);
  const height = Math.max(260, Math.min(325, width * .65 + 70)), left = 57, right = 18, top = 18, bottom = 54;
  const xTicks = isIO ? Math.max(2, Math.min(5, Math.floor((width - left - right) / 55))) : 5;
  const x = (isCrop ? d3.scalePoint().padding(.4) : isIO ? d3.scaleLinear() : d3.scaleLog()).domain(scales.x).range([left, width - right]);
  if (isIO) x.nice();
  const y = d3.scaleLinear().domain(scales.y).nice().range([height - bottom, top]);
  const iopsMetric = "device_iops";
  const bandwidthMetric = "device_read_gib_s";
  const xValue = row => isCrop ? row.crop_size : isIO ? row.metrics[iopsMetric].median : row.compression_fold;
  const yMetric = row => isCrop ? samplesPerSecond(row) : row.metrics[isIO ? bandwidthMetric : "useful_gib_s"];
  const svg = d3.select(wrapper).append("svg").attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "group").attr("aria-label", `${backend}: ${isCrop ? "Samples per second versus crop size" : isIO ? "Storage bandwidth versus IOPS" : "Image throughput versus compression ratio"}`);
  svg.append("g").attr("class", "grid").attr("transform", `translate(${left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickSize(-(width - left - right)).tickFormat(""));
  svg.append("g").attr("transform", `translate(0,${height - bottom})`)
    .call(d3.axisBottom(x).ticks(xTicks, isIO ? "~s" : ".2~g"));
  svg.append("g").attr("class", "y-axis").attr("transform", `translate(${left},0)`)
    .call(d3.axisLeft(y).ticks(5, isCrop ? "~s" : undefined));
  svg.append("text").attr("class", "axis-label").attr("x", (left + width - right) / 2)
    .attr("y", height - 7).attr("text-anchor", "middle").text(isCrop ? "Square crop width (pixels)" : isIO ? "IOPS (reads/s)" : "Compression ratio (×, log scale)");
  svg.append("text").attr("class", "axis-label").attr("transform", "rotate(-90)")
    .attr("x", -(top + height - bottom) / 2).attr("y", 14).attr("text-anchor", "middle").text(isCrop ? "Samples/s" : isIO ? "Storage bandwidth (GiB/s)" : "Image throughput (GiB/s)");
  const boundary = isIO || isCrop ? [] : frontier(rows), ids = new Set(boundary.map(row => row.id));
  const shown = rows.filter(row => isIO || isCrop || !$("frontier-only").checked || ids.has(row.id));
  if (!isIO && !isCrop) svg.append("path").datum(boundary).attr("class", "frontier").attr("fill", "none").attr("stroke", "var(--muted)")
    .attr("stroke-width", 1.5).attr("d", d3.line().x(row => x(xValue(row))).y(row => y(yMetric(row).median)));
  if (isCrop) for (const series of cropSeries(rows)) svg.append("path").datum(series)
    .attr("class", "crop-series").attr("fill", "none").attr("stroke", colors[series[0].setting.codec])
    .attr("stroke-width", 1.5).attr("stroke-dasharray", series[0].setting.block_mode === "bytes" ? "5 4" : null)
    .attr("d", d3.line().x(row => x(xValue(row))).y(row => y(yMetric(row).median)));
  const points = svg.append("g").selectAll("g").data(shown).join("g");
  const radius = row => isCrop ? 5 : 4 + Math.sqrt((row.setting.chunk_bytes ?? Math.max(...row.setting.sources.map(source => source.chunk_bytes))) / 16384) * .6;
  points.append("circle").attr("class", "point").attr("tabindex", 0).attr("role", "button")
    .attr("aria-pressed", row => row.id === selected).attr("aria-controls", "detail")
    .attr("aria-label", row => `${configurationName(row)}, ${isCrop ? `${number(yMetric(row).median)} samples/s` : isIO ? `${number(xValue(row))} IOPS, ${number(yMetric(row).median)} GiB/s storage bandwidth` : `${number(yMetric(row).median)} GiB/s image throughput, ${number(row.compression_fold)}× compression ratio`}`)
    .attr("cx", row => x(xValue(row))).attr("cy", row => y(yMetric(row).median))
    .attr("r", radius)
    .attr("fill", row => isIO || isCrop || ids.has(row.id) ? colors[row.setting.codec] : "var(--surface)")
    .attr("stroke", row => colors[row.setting.codec])
    .attr("stroke-width", row => row.id === selected ? 3 : 1.5)
    .on("click", (_, row) => select(row)).on("keydown", (event, row) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(row); }
    }).append("title").text(row => `${configurationName(row)}\n${isCrop ? `Samples/s: ${number(yMetric(row).median)}` : `${isIO ? "Storage bandwidth" : "Image throughput"}: ${number(yMetric(row).median)} GiB/s · ${isIO ? `IOPS: ${number(xValue(row))} reads/s` : `Compression ratio: ${number(row.compression_fold)}×`}`}`);
  points.append("path").attr("class", "range").attr("aria-hidden", "true")
    .attr("d", row => {
      const cx = x(xValue(row)), low = y(yMetric(row).min), high = y(yMetric(row).max), cap = radius(row) + 3;
      return `M${cx},${low}V${high}M${cx - cap},${low}H${cx + cap}M${cx - cap},${high}H${cx + cap}`;
    });
  if (isIO) points.append("path").attr("class", "range").attr("aria-hidden", "true")
    .attr("d", row => {
      const cy = y(yMetric(row).median), low = x(row.metrics[iopsMetric].min), high = x(row.metrics[iopsMetric].max), cap = radius(row) + 3;
      return `M${low},${cy}H${high}M${low},${cy - cap}V${cy + cap}M${high},${cy - cap}V${cy + cap}`;
    });
}

function table(rows, isIO, isCrop = false) {
  const columns = [
    ["Backend", row => row.backend.toUpperCase()],
    ["Chunk", row => chunkLabel(row.setting)],
    ["Block request", row => blockLabel(row.setting)],
  ];
  if (rows.some(row => row.crop_size)) columns.unshift(["Crop", row => `${row.crop_size} × ${row.crop_size}`]);
  if (isCrop) columns.push(["Samples/s", samplesPerSecond]);
  if (isIO) columns.push(
    ["IOPS (reads/s)", row => row.metrics.device_iops],
    ["Storage bandwidth (GiB/s)", row => row.metrics.device_read_gib_s],
  );
  if (isIO && rows.some(row => storagePerSample(study, row))) columns.push(
    ["Storage bytes / sample (KiB)", row => {
      const metric = storagePerSample(study, row)?.bytes;
      return metric ? {...metric, median: metric.median / 1024, min: metric.min / 1024, max: metric.max / 1024} : null;
    }],
    ["Storage reads / sample", row => storagePerSample(study, row)?.reads],
  );
  columns.push(
    ["Image throughput (GiB/s)", row => row.metrics.useful_gib_s],
    ["Compression ratio (×)", row => number(row.compression_fold)],
    ["Repeats", row => row.metrics.useful_gib_s.n],
  );
  $("columns").replaceChildren(...["Setting", ...columns.map(([name]) => name)].map(name => {
    const heading = element("th", name);
    heading.scope = "col";
    return heading;
  }));
  $("table-description").textContent = `${views[$("view").value]}. Rates show medians with observed min–max below, across independent process repetitions.`;
  for (const row of rows) {
    const tr = element("tr"), td = element("td"), button = element("button", codecs[row.setting.codec]);
    button.type = "button";
    button.dataset.setting = row.id;
    button.setAttribute("aria-label", `Select ${configurationName(row)}`);
    button.setAttribute("aria-controls", "detail");
    button.setAttribute("aria-pressed", row.id === selected);
    button.addEventListener("click", () => select(row));
    td.append(button);
    tr.append(td);
    for (const [, value] of columns) {
      const cell = element("td"), metric = value(row);
      if (metric && typeof metric === "object") {
        cell.append(element("span", metricStrings(metric, 4).median));
        const observed = element("span", range(metric));
        observed.className = "observed-range";
        cell.append(observed);
      } else cell.textContent = metric ?? "Not recorded";
      tr.append(cell);
    }
    $("rows").append(tr);
  }
}

function updateContext() {
  const cohort = study.cohorts?.find(value => value.id === $("input").value);
  const bakedFiles = study.baked && cohort?.input_ids.reduce((total, name) => total + study.baked.source_shards[name], 0);
  let geometry = cohort
    ? `${cohort.input_ids.length} source inputs, mixed within each pass. ${cohort.omitted_dataset ? `Omitted collection: ${cohort.omitted_dataset}. ` : ""}${study.volume.frame_shape.join(" × ")} frames; ${study.volume.shard_spatial_shape.join(" × ")} spatial shards; ${number(study.volume.shard_bytes / 2 ** 30)} GiB uncompressed per shard. ${study.weighting}. Compression combines equal logical source volumes. Expanded pixels are tiled source distributions.`
    : "Each input is a small replayed image pack.";
  if (study.baked) geometry = $("workload").value === "baked"
    ? `${cohort.input_ids.length} source inputs. ${bakedFiles} shard files total per process, approximately ${number(study.baked.shard_bytes / 2 ** 30)} GiB uncompressed each. One crop per compression chunk, read forward through each file. The paired random coordinates and native pixel types are preserved. Compression is weighted by the stored sample mixture.`
    : `${cohort.input_ids.length} source inputs, mixed within each pass. Uniform random translations across ${study.volume.frame_shape.join(" × ")} frames; crops may cross chunk and shard boundaries. ${number(study.volume.shard_bytes / 2 ** 30)} GiB uncompressed per spatial shard. Repeated or overlapping reads within a cold-start pass are allowed. Compression combines equal logical source volumes.`;
  if (study.batch_output_bytes) geometry += ` Batches hold ${number(study.batch_output_bytes / 2 ** 20)} MiB of float32 output; lookahead holds ${number(study.lookahead_output_bytes / 2 ** 20)} MiB at every crop size. These are single-plane storage workloads, not complete training batches.`;
  $("context").textContent = `${study.pilot ? "Pilot · limited settings. " : ""}${study.profile?.repeats ?? study.rows[0].metrics.useful_gib_s.n} process repetitions per setting · ${study.pipeline.decode_workers} CPU decode workers · ${study.pipeline.n_io_threads} file workers. ${geometry}`;
}

function render() {
  updateContext();
  const isIO = $("view").value === "device";
  const isCrop = $("view").value === "crop";
  const candidates = study.rows.filter(row => matches(row, {input: $("input").value,
    workload: $("workload").value, crop: isCrop ? "" : $("crop").value}));
  options("codec", [...new Set(candidates.map(row => row.setting.codec))], id => codecs[id], true);
  const byCodec = candidates.filter(row => matches(row, {codec: $("codec").value}));
  options("chunk", [...new Set(byCodec.map(row => row.setting.chunk_mode === "sample" ? "sample" : row.setting.chunk_bytes))].sort((a, b) => a - b), value => value === "sample" ? "One sample" : kib(value), true);
  const byChunk = byCodec.filter(row => matches(row, {chunk: $("chunk").value}));
  options("block", [...new Set(byChunk.map(row => row.setting.block_mode === "chunk" ? "chunk" : row.setting.block_bytes).filter(Boolean))].sort((a, b) => a - b), value => value === "chunk" ? "Whole chunk" : kib(value), true);
  const filters = Object.fromEntries(["input", "workload", "crop", "codec", "chunk", "block"].map(id => [id, $(id).value]));
  $("crop-label").hidden = isCrop || !study.rows.some(row => row.crop_size);
  filtered = study.rows.filter(row => matches(row, {...filters, crop: isCrop ? "" : filters.crop}));
  const query = new URLSearchParams({study: $("study").value, view: $("view").value, ...filters});
  query.set("explorer", $("explorer").open ? "open" : "closed");
  if ($("overview-recipe").value) query.set("recipe", $("overview-recipe").value);
  if (filtered.some(row => row.id === selected)) query.set("setting", selected);
  history.replaceState(null, "", `#${query}`);
  $("plots").replaceChildren();
  $("rows").replaceChildren();
  const valid = filtered.filter(isIO ? devicePlottable : plottable);
  const work = isIO && valid.length && valid.every(row => storagePerSample(study, row));
  const fold = valid.map(row => row.compression_fold);
  const max = d3.max(valid, row => isCrop ? samplesPerSecond(row).max : row.metrics[isIO ? "device_read_gib_s" : "useful_gib_s"].max);
  const scales = {x: isCrop ? [...new Set(study.rows.map(row => row.crop_size))].sort((a, b) => a - b) : isIO ? [0, d3.max(valid, row => row.metrics.device_iops.max) * 1.12]
    : [d3.min(fold) / 1.15, d3.max(fold) * 1.15], y: [0, max * 1.12]};
  const panels = (isCrop ? [...d3.group(valid, row => row.backend).values()] : groups(valid)).sort((a, b) => a[0].backend.localeCompare(b[0].backend));
  $("plots").classList.toggle("single-panel", panels.length === 1);
  const storageScales = work ? Object.fromEntries(["bytes", "reads"].map(key => [key,
    d3.max(valid, row => storagePerSample(study, row)[key].max)])) : null;
  for (const rows of panels) {
    if (work) storagePanel($("plots"), study, rows, storageScales, select, selected);
    else panel(rows, scales, isIO, isCrop);
  }
  if (!valid.length) $("plots").append(element("p", isIO && filtered.length ? "Storage counters were not recorded for this study. File read requests cannot substitute for them." : "No measurements match these filters."));
  $("frontier-filter").hidden = isIO || isCrop;
  $("codec-legend").hidden = Boolean(work);
  $("plot-note").textContent = isCrop
    ? `Samples/s counts completed single-plane crops. Each line follows the same encoding recipe across measured crop sizes; dashed lines use a fixed Blosc block request. Lines are guides, not Pareto frontiers or predictions. Batch and lookahead byte budgets stay fixed. Points show medians; bars show observed ranges.${panels.length > 1 ? " Both panels use the same axes." : ""}`
    : work
    ? `Storage work per delivered crop: bytes and completed read operations, each divided by the samples completed in that execution, then summarized. Lower is better for the same crop size. Ranges are observed min–max.${panels.length > 1 ? " Both readers share each metric’s scale." : ""} Counts include metadata, read-ahead, and other activity on the drives. Storage bandwidth and IOPS remain in the table and configuration details.`
    : isIO
    ? "Storage bandwidth counts bytes read from the storage system per second; IOPS counts completed read operations per second. Both are summed across its drives. More IOPS alone does not imply higher image throughput. Points show medians; bars show observed ranges."
    : "Image throughput counts requested pixel bytes per second in the source data type. Compression ratio is image bytes divided by stored file bytes. Upper right is better. Filled, connected points form each panel’s observed frontier. Points show medians; bars show observed ranges. All panels use the same axes.";
  if (isCrop && filters.workload === "baked" && study.pipeline.chunks_per_input_buffer) {
    const batches = [...new Map(valid.filter(row => row.backend === "cpu")
      .map(row => [row.crop_size, row.sampling.samples_per_batch])).entries()].sort((a, b) => a[0] - b[0]);
    if (batches.length && batches.every(([, samples]) => Number.isSafeInteger(samples) && samples > 0)) {
      const limit = study.pipeline.chunks_per_input_buffer;
      const counts = batches.map(([crop, samples]) => `${Math.ceil(samples / limit)} at ${crop}²`).join(", ");
      $("plot-note").textContent += ` CPU input buffers hold ${limit} chunks; minimum read/decode groups per baked batch are ${counts}. This batching affects the curve.`;
    }
  }
  const count = groups(filtered).reduce((total, rows) => total + frontier(rows).length, 0);
  $("count").textContent = isCrop ? `${valid.length} settings across crop sizes` : work ? `${valid.length} settings · storage work per ${filters.crop} × ${filters.crop} crop`
    : `${isIO ? `${valid.length} settings with storage measurements` : `${filtered.length} settings · ${count} on the observed frontiers`} · larger markers indicate larger chunks${study.baked ? " (largest source chunk for mixed sizes)" : ""}`;
  table(filtered, isIO, isCrop);
  if (!filtered.some(row => row.id === selected)) {
    selected = null;
    $("selection-status").textContent = "Select a setting to inspect its measurements.";
    $("selection-details").hidden = true;
    $("detail-content").replaceChildren(element("p", "Select a setting to inspect its measurements."));
  } else {
    select(filtered.find(row => row.id === selected), false);
  }
}

async function loadStudy() {
  const current = ++generation, entry = entries.find(value => value.id === $("study").value);
  $("workspace").hidden = true;
  $("status").hidden = false;
  $("status").textContent = "Loading measurements…";
  const value = documents.get(entry.id) || await json(new URL(`data/${entry.file}`, import.meta.url));
  if (current !== generation) return;
  study = value;
  selected = saved.get("setting");
  saved.delete("setting");
  if (study.version !== 1 || !study.rows.length) throw new Error("This study has no supported measurements.");
  const hasCrops = study.rows.some(row => row.crop_size);
  options("view", Object.keys(views).filter(id => id !== "crop" || hasCrops), id => id === "device" && hasCrops ? "Storage work per sample" : views[id]);
  const choices = study.cohorts || study.inputs;
  $("input-label").textContent = study.cohorts ? "Source mixture" : "Image input";
  options("input", choices.map(input => input.id), id => choices.find(input => input.id === id).name);
  $("input-filter").hidden = choices.length === 1;
  options("workload", [...new Set(study.rows.map(row => row.access_pattern || row.workload))], id => workloads[id] || id);
  options("crop", [...new Set(study.rows.map(row => row.crop_size).filter(Boolean))].sort((a, b) => a - b), size => `${size} × ${size}`);
  $("raw").href = `data/${entry.file}`;
  const context = study.context;
  const storage = context.machine.local_storage;
  const system = $("storage-content");
  system.replaceChildren(element("p", "All panels in this study use this storage system. Storage bandwidth and IOPS are Linux counters summed over its drives, including metadata, read-ahead, and other processes. They are not internal SSD flash-operation counts."));
  const models = new Map();
  for (const device of storage?.devices || []) {
    const model = `${device.model}${device.bytes ? ` (${number(device.bytes / 10 ** 12)} TB each)` : ""}`;
    models.set(model, (models.get(model) || 0) + 1);
  }
  const gpu = context.machine.gpu?.split("\n")[0]?.split(",");
  const filesystem = storage?.filesystem || context.machine.filesystem?.filesystems?.[0];
  fields(system, [
    ["Machine", context.machine.host],
    ["CPU", context.machine.cpu?.["Model name"] || "Not recorded"],
    ["Allowed logical CPUs", context.machine.cpu_affinity?.length ?? "Not recorded"],
    ["GPU", gpu?.[0] || "None"],
    ["GPU driver", gpu?.[2]?.trim() || "Not recorded"],
    ["Filesystem", filesystem ? `${filesystem.fstype}${filesystem.source ? ` on ${filesystem.source.split("[")[0]}` : ""}` : "Not recorded"],
    ["Drives", storage ? `${storage.devices.length} PCIe NVMe drives` : "Not recorded"],
    ["Drive models", [...models].map(([model, count]) => `${count} × ${model}`).join("; ") || "Not recorded"],
  ]);
  system.append(element("p", storage ? "The drives are not reserved exclusively for this study. Idle activity is retained in Diagnostics. SSD/controller caches are not controlled." : "A drive inventory and storage counters were not recorded for this cached study."));
  $("cache-condition").textContent = context.cache_policy === "cold-pass"
    ? "Cold file-cache passes. Every timed pass starts with zero resident shard-data pages and runs many batches through the pipeline. Storage reads are verified. Codecs and metadata are warmed."
    : context.cache_policy === "cold"
    ? "Cold file-cache reads. Every timed batch starts with zero resident data-file pages; storage reads are verified. Codecs and metadata are warmed."
    : "Cached pilot. Files were preloaded into RAM and reads were replayed. Storage bandwidth and IOPS were not recorded.";
  $("workspace").hidden = false;
  $("status").hidden = true;
  render();
}

function error(value) {
  $("workspace").hidden = true;
  $("status").hidden = false;
  $("status").textContent = `${value.message} Reload this page to try again.`;
}

async function openSetting(document, row) {
  $("explorer").open = true;
  $("study").value = document.id;
  await loadStudy();
  $("input").value = row.input_id;
  $("workload").value = row.access_pattern;
  $("crop").value = String(row.crop_size);
  $("view").value = "pareto";
  for (const id of ["codec", "chunk", "block"]) $(id).value = "";
  $("frontier-only").checked = false;
  selected = row.id;
  render();
  select(row);
  $("explorer-heading").focus();
}

$("explore-link").addEventListener("click", event => {
  if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  $("explorer").open = true;
  $("explorer-heading").focus();
});
$("explorer").addEventListener("toggle", () => {
  if (!study) return;
  if ($("explorer").open) render();
  else {
    const hash = new URLSearchParams(location.hash.slice(1));
    hash.set("explorer", "closed");
    history.replaceState(null, "", `#${hash}`);
  }
});

$("selection-details").addEventListener("click", event => {
  event.preventDefault();
  $("detail-heading").focus();
});
$("study").addEventListener("change", () => loadStudy().catch(error));
for (const id of ["input", "workload", "crop", "view", "codec", "chunk", "block", "frontier-only"]) $(id).addEventListener("change", render);
window.addEventListener("resize", () => { if (study && !$("workspace").hidden) render(); });
window.addEventListener("hashchange", () => {
  if (location.hash === "#explorer") {
    $("explorer").open = true;
    $("explorer-heading").focus();
  } else {
    const query = new URLSearchParams(location.hash.slice(1));
    if (["study", "view", "input", "workload", "crop", "codec", "chunk", "block", "setting", "recipe", "explorer"].some(key => query.has(key))) location.reload();
  }
});
$("csv").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([csv(filtered, {cache_policy: study.context.cache_policy, host: study.context.machine.host, document: study})], {type: "text/csv;charset=utf-8"}));
  const link = element("a");
  link.href = url;
  link.download = `damacy-${study.id}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

try {
  if (!globalThis.d3) throw new Error("The plotting library could not be loaded.");
  const index = await json(new URL("data/index.json", import.meta.url));
  entries = index.studies;
  if (!entries.length) {
    $("status").textContent = "No retained reader study has been published yet. See the measurement protocol for the prepared study.";
  } else {
    if (entries.some(entry => !/^[a-f0-9]{20}\.json$/.test(entry.file))) throw new Error("Invalid study index.");
    options("study", entries.map(entry => entry.id), id => {
      const entry = entries.find(value => value.id === id);
      return `${entry.label}${entry.pilot ? " (pilot)" : ""}`;
    });
    const loaded = await Promise.all(entries.map(entry => json(new URL(`data/${entry.file}`, import.meta.url))));
    loaded.forEach((document, index) => documents.set(entries[index].id, document));
    mountOverview(loaded, (document, row) => openSetting(document, row).catch(error));
    await loadStudy();
    if (!$("overview").hidden) $("explorer").open = openExplorer;
  }
} catch (value) { error(value); }
