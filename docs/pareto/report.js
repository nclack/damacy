import {csv, devicePlottable, frontier, groups, ioPlottable, matches, plottable} from "./analysis.mjs";

const $ = id => document.getElementById(id);
const number = value => Number.isFinite(value) ? value.toLocaleString(undefined, {maximumSignificantDigits: 4}) : "—";
const kib = bytes => bytes == null ? "—" : `${number(bytes / 1024)} KiB`;
const range = metric => metric ? `${number(metric.min)}–${number(metric.max)}` : "Not recorded";
const colors = {none: "var(--gray)", zstd: "var(--blue)", "blosc-zstd": "var(--orange)"};
const codecs = {none: "Uncompressed", zstd: "Zstd", "blosc-zstd": "Blosc Zstd"};
const workloads = {"random-256": "Random 256 × 256 crops", "whole-planes": "Whole-plane reads"};
const views = {pareto: "Throughput and compression", io: "Application reads and bandwidth", device: "NVMe IOPS and bandwidth"};
const saved = new URLSearchParams(location.hash.slice(1));
let study, entries, selected, filtered = [], generation = 0;

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
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load measurements (${response.status}).`);
  return response.json();
}

function fields(target, rows) {
  const list = element("dl");
  for (const [name, value] of rows) list.append(element("dt", name), element("dd", value));
  target.append(list);
}

function configurationName(row) {
  return `${codecs[row.setting.codec]} · ${row.backend.toUpperCase()} · ${kib(row.setting.chunk_bytes)} chunk`
    + (row.setting.block_bytes == null ? "" : ` · ${kib(row.setting.block_bytes)} block`);
}

function select(row, announce = true) {
  selected = row.id;
  const target = $("detail-content");
  target.replaceChildren(element("p", `${codecs[row.setting.codec]} · ${row.backend.toUpperCase()}`));
  const setting = row.setting, metrics = row.metrics;
  fields(target, [
    ["Useful throughput", `${number(metrics.useful_gib_s.median)} GiB/s`],
    ["Observed useful range", `${range(metrics.useful_gib_s)} GiB/s`],
    ["Compression", `${number(row.compression_fold)}×`],
    ["Process repetitions", metrics.useful_gib_s.n],
    ["Chunk", kib(setting.chunk_bytes)],
    ["Chunk shape", setting.chunk_shape.join(" × ")],
    ["Shard shape", setting.shard_shape.join(" × ")],
    ["Blosc block request", kib(setting.block_bytes)],
    ["Blocks in headers", setting.actual_block_bytes.map(kib).join(", ") || "—"],
    ["Compression level", setting.level || "—"],
    ["Source type", setting.dtype],
    ["Returned type", study.pipeline.dtype],
    ["Sample shape", row.sampling.sample_shape.join(" × ")],
    ["Encoded reads / useful bytes", `${number(metrics.read_amplification?.median)}×`],
    ["Decoded / useful bytes", metrics.decode_amplification ? `${number(metrics.decode_amplification.median)}×` : "Not recorded"],
    ["Returned throughput", `${number(metrics.returned_gib_s.median)} GiB/s`],
    ["Application read requests/s", number(metrics.read_iops?.median)],
    ["Request rate range", range(metrics.read_iops)],
    ["Encoded read bandwidth", `${number(metrics.read_gib_s?.median)} GiB/s`],
    ["Encoded bandwidth range", `${range(metrics.read_gib_s)} GiB/s`],
    ["Mean encoded read size", kib(metrics.mean_read_bytes?.median)],
    ["Read requests / useful GiB", number(metrics.reads_per_useful_gib?.median)],
    ["Initial first batch", `${number(metrics.first_batch_ms.median)} ms`],
    ["Reference range / median", `${number(row.reference_spread_percent)}%`],
    ["Stored file bytes", number(setting.stored_bytes)],
    ["Logical image bytes", number(setting.logical_bytes)],
  ]);
  if (metrics.device_iops) {
    target.append(element("h3", "NVMe device counters"));
    fields(target, [
      ["Read IOPS", number(metrics.device_iops.median)],
      ["Read IOPS range", range(metrics.device_iops)],
      ["Read bandwidth", `${number(metrics.device_read_gib_s.median)} GiB/s`],
      ["Read bandwidth range", `${range(metrics.device_read_gib_s)} GiB/s`],
      ["Mean device read size", kib(metrics.device_mean_read_bytes.median)],
      ["Device reads / useful bytes", `${number(metrics.device_read_amplification.median)}×`],
      ["Idle device read IOPS", number(metrics.idle_device_iops.median)],
      ["Idle device bandwidth", `${number(metrics.idle_device_gib_s.median)} GiB/s`],
      ["Cache clearing (excluded)", `${number(metrics.cache_preparation_seconds.median)} s`],
    ]);
  }
  target.append(element("h3", "Stage measurements"));
  target.append(element("p", "One execution nearest the plotted bandwidth median; stage intervals overlap."));
  const plottedMetric = $("view").value === "device" ? "device_read_gib_s" : $("view").value === "io" ? "read_gib_s" : "useful_gib_s";
  const observations = study.observations.filter(value => row.observation_ids.includes(value.task.id));
  observations.sort((a, b) => Math.abs(a.metrics[plottedMetric] - metrics[plottedMetric]?.median)
    - Math.abs(b.metrics[plottedMetric] - metrics[plottedMetric]?.median));
  fields(target, observations[0].result.stages.map(stage => [stage.name.replaceAll("_", " "), `${number(stage.ms_total)} ms`]));
  const metadata = observations[0].result.metadata;
  if (metadata) {
    target.append(element("h3", "Metadata in this execution"));
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
    fields(target, counts);
  }
  target.append(element("h3", "Input and source"));
  const input = study.inputs.find(value => value.id === row.input_id);
  const link = element("a", input.name);
  if (/^https?:\/\//.test(input.source.url)) link.href = input.source.url;
  target.append(link, element("p", input.source.attribution), element("p", `License: ${input.source.license}`));
  fields(target, [["Revision", `${study.context.source.revision?.slice(0, 12) || "Unknown"}${study.context.source.local_changes ? " + local changes" : ""}`],
    ["Binary SHA256", study.context.binary_sha256], ["Input SHA256", setting.input_sha256]]);
  d3.selectAll("circle.point").attr("stroke-width", point => point.id === selected ? 3 : 1.5)
    .attr("aria-pressed", point => point.id === selected);
  for (const button of $("rows").querySelectorAll("button")) button.setAttribute("aria-pressed", button.dataset.setting === selected);
  $("selection-details").hidden = false;
  if (announce) $("selection-status").textContent = `${configurationName(row)} selected. ${number(metrics.useful_gib_s.median)} useful GiB/s, ${number(row.compression_fold)}× compression, ${metrics.useful_gib_s.n} process repetitions.`;
}

function panel(rows, scales, isIO, isDevice) {
  const first = rows[0], wrapper = element("section");
  wrapper.className = "panel";
  const backend = first.backend === "cpu" ? "CPU" : "CUDA";
  wrapper.append(element("h2", backend), element("p", study.context.storage));
  $("plots").append(wrapper);
  const padding = parseFloat(getComputedStyle(wrapper).paddingLeft) + parseFloat(getComputedStyle(wrapper).paddingRight);
  const width = Math.max(240, wrapper.clientWidth - padding);
  const height = Math.max(260, Math.min(325, width * .65 + 70)), left = 57, right = 18, top = 18, bottom = 54;
  const x = (isIO ? d3.scaleLinear() : d3.scaleLog()).domain(scales.x).range([left, width - right]);
  if (isIO) x.nice();
  const y = d3.scaleLinear().domain(scales.y).nice().range([height - bottom, top]);
  const iopsMetric = isDevice ? "device_iops" : "read_iops";
  const bandwidthMetric = isDevice ? "device_read_gib_s" : "read_gib_s";
  const xValue = row => isIO ? row.metrics[iopsMetric].median : row.compression_fold;
  const yMetric = row => row.metrics[isIO ? bandwidthMetric : "useful_gib_s"];
  const svg = d3.select(wrapper).append("svg").attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "group").attr("aria-label", `${backend}: ${isIO ? (isDevice ? "NVMe bandwidth versus IOPS" : "application bandwidth versus requests per second") : "useful throughput versus compression"}`);
  svg.append("g").attr("class", "grid").attr("transform", `translate(${left},0)`)
    .call(d3.axisLeft(y).ticks(5).tickSize(-(width - left - right)).tickFormat(""));
  svg.append("g").attr("transform", `translate(0,${height - bottom})`)
    .call(d3.axisBottom(x).ticks(5, isIO ? "~s" : ".2~g"));
  svg.append("g").attr("transform", `translate(${left},0)`).call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("class", "axis-label").attr("x", (left + width - right) / 2)
    .attr("y", height - 7).attr("text-anchor", "middle").text(isDevice ? "NVMe read IOPS" : isIO ? "Application read requests/s" : "Compression fold × (log scale)");
  svg.append("text").attr("class", "axis-label").attr("transform", "rotate(-90)")
    .attr("x", -(top + height - bottom) / 2).attr("y", 14).attr("text-anchor", "middle").text(isDevice ? "Device read GiB/s" : isIO ? "Encoded read GiB/s" : "Useful image GiB/s");
  const boundary = isIO ? [] : frontier(rows), ids = new Set(boundary.map(row => row.id));
  const shown = rows.filter(row => isIO || !$("frontier-only").checked || ids.has(row.id));
  if (!isIO) svg.append("path").datum(boundary).attr("class", "frontier").attr("fill", "none").attr("stroke", "var(--muted)")
    .attr("stroke-width", 1.5).attr("d", d3.line().x(row => x(xValue(row))).y(row => y(yMetric(row).median)));
  const points = svg.append("g").selectAll("g").data(shown).join("g");
  const radius = row => 4 + Math.sqrt(row.setting.chunk_bytes / 16384) * .6;
  points.append("circle").attr("class", "point").attr("tabindex", 0).attr("role", "button")
    .attr("aria-pressed", row => row.id === selected).attr("aria-controls", "detail")
    .attr("aria-label", row => `${codecs[row.setting.codec]}, chunk ${kib(row.setting.chunk_bytes)}, block ${kib(row.setting.block_bytes)}, ${isIO ? `${number(xValue(row))} ${isDevice ? "device IOPS" : "application requests/s"}, ${number(yMetric(row).median)} GiB/s` : `${number(yMetric(row).median)} useful GiB/s, ${number(row.compression_fold)} fold`}`)
    .attr("cx", row => x(xValue(row))).attr("cy", row => y(yMetric(row).median))
    .attr("r", radius)
    .attr("fill", row => isIO || ids.has(row.id) ? colors[row.setting.codec] : "var(--surface)")
    .attr("stroke", row => colors[row.setting.codec])
    .attr("stroke-width", row => row.id === selected ? 3 : 1.5)
    .on("click", (_, row) => select(row)).on("keydown", (event, row) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(row); }
    }).append("title").text(row => `${codecs[row.setting.codec]} · ${kib(row.setting.chunk_bytes)} chunk · ${kib(row.setting.block_bytes)} block\n${number(yMetric(row).median)} ${isDevice ? "device" : isIO ? "encoded" : "useful"} GiB/s · ${isIO ? `${number(xValue(row))} requests/s · ${kib(row.metrics[isDevice ? "device_mean_read_bytes" : "mean_read_bytes"].median)} per read` : `${number(row.compression_fold)}×`}`);
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

function table(rows, isIO, isDevice) {
  const columns = [
    ["Backend", row => row.backend.toUpperCase()],
    ["Chunk", row => kib(row.setting.chunk_bytes)],
    ["Block request", row => kib(row.setting.block_bytes)],
  ];
  if (isIO) columns.push(
    [isDevice ? "NVMe read IOPS" : "Application reads/s", row => row.metrics[isDevice ? "device_iops" : "read_iops"]],
    [isDevice ? "Device read GiB/s" : "Encoded read GiB/s", row => row.metrics[isDevice ? "device_read_gib_s" : "read_gib_s"]],
    [isDevice ? "Mean device read" : "Mean encoded read", row => kib(row.metrics[isDevice ? "device_mean_read_bytes" : "mean_read_bytes"]?.median)],
  );
  columns.push(
    ["Useful GiB/s", row => row.metrics.useful_gib_s],
    ["Compression ×", row => number(row.compression_fold)],
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
        cell.append(element("span", number(metric.median)));
        const observed = element("span", range(metric));
        observed.className = "observed-range";
        cell.append(observed);
      } else cell.textContent = metric ?? "—";
      tr.append(cell);
    }
    $("rows").append(tr);
  }
}

function render() {
  const isDevice = $("view").value === "device", isIO = $("view").value !== "pareto";
  const filters = Object.fromEntries(["input", "workload", "codec", "chunk", "block"].map(id => [id, $(id).value]));
  filtered = study.rows.filter(row => matches(row, filters));
  const query = new URLSearchParams({study: $("study").value, view: $("view").value, ...filters});
  history.replaceState(null, "", `#${query}`);
  $("plots").replaceChildren();
  $("rows").replaceChildren();
  const valid = filtered.filter(isDevice ? devicePlottable : isIO ? ioPlottable : plottable);
  const fold = valid.map(row => row.compression_fold);
  const max = d3.max(valid, row => row.metrics[isDevice ? "device_read_gib_s" : isIO ? "read_gib_s" : "useful_gib_s"].max);
  const scales = {x: isIO ? [0, d3.max(valid, row => row.metrics[isDevice ? "device_iops" : "read_iops"].max) * 1.12]
    : [d3.min(fold) / 1.15, d3.max(fold) * 1.15], y: [0, max * 1.12]};
  const panels = groups(valid).sort((a, b) => a[0].backend.localeCompare(b[0].backend));
  $("plots").classList.toggle("single-panel", panels.length === 1);
  for (const rows of panels) panel(rows, scales, isIO, isDevice);
  if (!valid.length) $("plots").append(element("p", isDevice && filtered.length ? "No NVMe counters were recorded for this study." : isIO && filtered.length ? "No application read counters were recorded for these settings." : "No measurements match these filters."));
  $("frontier-filter").hidden = isIO;
  $("plot-note").textContent = isDevice
    ? "NVMe counters include readahead and any other activity on those devices. Adjacent idle intervals are recorded in configuration details. The axes show read operations and bytes per timed second; higher IOPS alone does not imply better application performance. Points are medians; capped bars show observed min–max across process repetitions."
    : isIO
    ? "Each point shows application read requests/s and encoded bandwidth. For each execution, bandwidth equals requests/s × mean request size. More requests can reflect smaller reads; inspect request size and useful throughput in configuration details. Points are medians; capped bars show observed min–max across process repetitions."
    : "Upper right is better. Filled, connected points form the observed Pareto frontier; hollow points are other tested settings. Points are medians; capped bars show observed min–max across process repetitions. Each panel has its own frontier; all panels use the same axes.";
  const count = groups(filtered).reduce((total, rows) => total + frontier(rows).length, 0);
  $("count").textContent = `${isIO ? `${valid.length} settings with read counts` : `${filtered.length} settings · ${count} on the observed frontiers`} · larger markers indicate larger chunks`;
  table(filtered, isIO, isDevice);
  if (!filtered.some(row => row.id === selected)) {
    selected = null;
    $("selection-status").textContent = "Select a point or table setting to inspect its measurements.";
    $("selection-details").hidden = true;
    $("detail-content").replaceChildren(element("p", "Select a point or table setting to inspect its measurements."));
  } else {
    select(filtered.find(row => row.id === selected), false);
  }
}

async function loadStudy() {
  const current = ++generation, entry = entries.find(value => value.id === $("study").value);
  $("workspace").hidden = true;
  $("status").hidden = false;
  $("status").textContent = "Loading measurements…";
  const value = await json(new URL(`data/${entry.file}`, import.meta.url));
  if (current !== generation) return;
  study = value;
  selected = null;
  if (study.version !== 1 || !study.rows.length) throw new Error("This study has no supported measurements.");
  options("view", Object.keys(views), id => views[id]);
  options("input", study.inputs.map(input => input.id), id => study.inputs.find(input => input.id === id).name);
  options("workload", [...new Set(study.rows.map(row => row.workload))], id => workloads[id] || id);
  options("codec", [...new Set(study.rows.map(row => row.setting.codec))], id => codecs[id], true);
  options("chunk", [...new Set(study.rows.map(row => row.setting.chunk_bytes))].sort((a, b) => a - b), kib, true);
  options("block", [...new Set(study.rows.map(row => row.setting.block_bytes).filter(Boolean))].sort((a, b) => a - b), kib, true);
  $("raw").href = `data/${entry.file}`;
  const context = study.context;
  const storage = context.machine.local_storage;
  $("cache-condition").textContent = context.cache_policy === "cold"
    ? `Cold file-cache reads · ${storage.devices.length} PCIe NVMe devices (${[...new Set(storage.devices.map(device => device.model))].join(", ")}). Every timed batch starts with zero resident data-file pages; device reads are verified. Codec and metadata caches are warmed. SSD/controller caches are not controlled.`
    : "Cached pilot · files were preloaded into RAM and reads were replayed. These measurements characterize the cached reader and do not measure cold NVMe performance.";
  $("context").textContent = `${study.pilot ? "Pilot · limited settings. " : ""}${context.machine.cpu?.["Model name"] || context.machine.host} · ${context.machine.cpu_affinity?.length ?? "Unknown"} allowed logical CPUs · ${study.pipeline.decode_workers} CPU decode workers · ${study.pipeline.n_io_threads} file workers · ${context.machine.gpu?.split("\n")[0]?.split(",")[0] || "CPU only"}. Each input is a small replayed image pack.`;
  $("workspace").hidden = false;
  $("status").hidden = true;
  render();
}

function error(value) {
  $("workspace").hidden = true;
  $("status").hidden = false;
  $("status").textContent = `${value.message} Reload this page to try again.`;
}

$("selection-details").addEventListener("click", event => {
  event.preventDefault();
  $("detail-heading").focus();
});
$("study").addEventListener("change", () => loadStudy().catch(error));
for (const id of ["input", "workload", "view", "codec", "chunk", "block", "frontier-only"]) $(id).addEventListener("change", render);
window.addEventListener("resize", () => { if (study && !$("workspace").hidden) render(); });
$("csv").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([csv(filtered, {cache_policy: study.context.cache_policy, host: study.context.machine.host})], {type: "text/csv;charset=utf-8"}));
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
    $("status").textContent = "No retained reader study has been published yet. The measurement protocol describes the prepared microscopy study; existing throughput and timing histories remain available above.";
  } else {
    if (entries.some(entry => !/^[a-f0-9]{20}\.json$/.test(entry.file))) throw new Error("Invalid study index.");
    options("study", entries.map(entry => entry.id), id => {
      const entry = entries.find(value => value.id === id);
      return `${entry.label}${entry.pilot ? " (pilot)" : ""}`;
    });
    await loadStudy();
  }
} catch (value) { error(value); }
