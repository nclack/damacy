import {samplesPerSecond} from "./analysis.mjs?v=5d9b4385b308";
import {bestRows, comparableRows, cropSizes, fixedRecipes, metricStrings, readers, recipeLabel} from "./comparisons.mjs?v=5d9b4385b308";

const $ = id => document.getElementById(id);
const number = value => value.toLocaleString("en", {maximumSignificantDigits: 3});
const metricText = metric => {
  const value = metricStrings(metric);
  return `${value.median} [${value.min}–${value.max}]`;
};
const colors = ["var(--gray)", "var(--blue)", "var(--orange)"];
const symbols = [d3.symbolCircle, d3.symbolSquare, d3.symbolDiamond];

function node(tag, text, className) {
  const value = document.createElement(tag);
  if (text !== undefined) value.textContent = text;
  if (className) value.className = className;
  return value;
}

function settingLink(reader, row, text, openRow) {
  const link = node("a", text);
  link.href = `#study=${reader.document.id}&workload=${row.access_pattern}&crop=${row.crop_size}&setting=${row.id}`;
  link.addEventListener("click", event => {
    if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    openRow(reader.document, row);
  });
  return link;
}

function chart(target, height, label) {
  const width = Math.max(220, target.clientWidth);
  const svg = d3.select(target).append("svg").attr("viewBox", `0 0 ${width} ${height}`)
    .attr("role", "group").attr("aria-label", label);
  return {svg, width};
}

function range(svg, metric, x, y, color) {
  svg.append("path").attr("class", "summary-range").attr("stroke", color).attr("aria-hidden", "true")
    .attr("d", `M${x(metric.min)},${y}H${x(metric.max)}M${x(metric.min)},${y - 4}V${y + 4}M${x(metric.max)},${y - 4}V${y + 4}`);
}

function mark(svg, x, y, symbol, color, label, action, filled = true) {
  const point = svg.append("path").attr("class", "summary-mark").attr("d", d3.symbol().type(symbol).size(65))
    .attr("transform", `translate(${x},${y})`).attr("fill", filled ? color : "var(--surface)").attr("stroke", color)
    .attr("stroke-width", 2).attr("role", "button").attr("tabindex", 0).attr("aria-label", label)
    .on("click", action).on("keydown", event => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); action(); }
    });
  point.append("title").text(label);
}

function legend(target, entries) {
  const list = node("ul", undefined, "summary-legend");
  for (const [index, entry] of entries.entries()) {
    const item = node("li");
    const swatch = node("span", entry.symbol || ["●", "■", "◆"][index], "summary-swatch");
    swatch.style.color = entry.color || colors[index];
    swatch.setAttribute("aria-hidden", "true");
    item.append(swatch, document.createTextNode(entry.label));
    list.append(item);
  }
  target.append(list);
}

function axis(target, domain) {
  const {svg, width} = chart(target, 45, "Shared image throughput scale in GiB/s");
  const x = d3.scaleLinear().domain(domain).range([12, width - 12]);
  svg.append("g").attr("transform", "translate(0,1)").call(d3.axisBottom(x).ticks(4).tickSizeOuter(0));
  svg.append("text").attr("x", width / 2).attr("y", 42).attr("text-anchor", "middle")
    .attr("class", "summary-axis-label").text("Image throughput (GiB/s) →");
}

function layoutOverview(readerList, openRow) {
  const target = $("layout-comparison");
  target.replaceChildren();
  const maximum = d3.max(readerList.flatMap(reader => reader.rows), row => row.metrics.useful_gib_s.max);
  const domain = d3.scaleLinear().domain([0, maximum * 1.04]).nice(4).domain();
  const ratios = [];
  for (const reader of readerList) {
    const card = node("article", undefined, "comparison-panel");
    card.append(node("h3", reader.label), node("p", reader.output, "reader-output"));
    target.append(card);
    for (const crop of cropSizes) {
      const random = bestRows(reader, crop, "random"), baked = bestRows(reader, crop, "baked");
      if (!random.length || !baked.length) continue;
      const ratio = baked[0].metrics.useful_gib_s.median / random[0].metrics.useful_gib_s.median;
      ratios.push(ratio);
      const group = node("section", undefined, "layout-pair");
      const heading = node("h4", `${crop} × ${crop}`);
      heading.append(node("span", `${ratio.toFixed(1)}× baked / random`, "gain"));
      group.append(heading);
      card.append(group);
      const plot = node("div");
      group.append(plot);
      const {svg, width} = chart(plot, 48, `${reader.label}, ${crop} square: all setting medians, best observed with min–max`);
      const x = d3.scaleLinear().domain(domain).range([12, width - 12]);
      for (const tick of d3.ticks(...domain, 4)) svg.append("line").attr("class", "summary-grid")
        .attr("x1", x(tick)).attr("x2", x(tick)).attr("y1", 0).attr("y2", 48);
      for (const [index, rows] of [random, baked].entries()) {
        const y = index ? 35 : 13, color = index ? "var(--blue)" : "var(--gray)", row = rows[0];
        const symbol = index ? d3.symbolDiamond : d3.symbolCircle;
        for (const other of rows.slice(1)) svg.append("path").attr("d", d3.symbol().type(symbol).size(23))
          .attr("transform", `translate(${x(other.metrics.useful_gib_s.median)},${y})`)
          .attr("fill", "var(--surface)").attr("stroke", color).attr("opacity", .65)
          .append("title").text(`${recipeLabel(other)}: ${metricText(other.metrics.useful_gib_s)} GiB/s`);
        range(svg, row.metrics.useful_gib_s, x, y, color);
        mark(svg, x(row.metrics.useful_gib_s.median), y, symbol, color,
          `${reader.label}, ${crop} square, ${index ? "baked" : "random"}: ${recipeLabel(row)}, ${metricText(row.metrics.useful_gib_s)} GiB/s`,
          () => openRow(reader.document, row), Boolean(index));
        const detail = node("p", undefined, "winner");
        const value = node("span", `${index ? "◆ Baked" : "○ Random"}  ${metricText(row.metrics.useful_gib_s)}`, "winner-value");
        value.style.color = color;
        detail.append(value, settingLink(reader, row, recipeLabel(row), openRow));
        group.append(detail);
      }
    }
    axis(card, domain);
  }
  if (ratios.length) $("layout-title").textContent = `Baked layouts deliver ${Math.min(...ratios).toFixed(1)}–${Math.max(...ratios).toFixed(1)}× more image data per second`;
}

function cropTable(target, series, openRow) {
  const scroll = node("div", undefined, "table-scroll crop-values");
  scroll.tabIndex = 0;
  scroll.setAttribute("role", "region");
  scroll.setAttribute("aria-label", "Sample rates by reader and crop, scroll for all columns");
  const table = node("table");
  const caption = node("caption", "Samples/s · median [min–max], n = 3");
  const head = node("thead"), tr = node("tr");
  for (const text of ["Reader", ...cropSizes.map(crop => `${crop}²`)]) {
    const th = node("th", text); th.scope = "col"; tr.append(th);
  }
  head.append(tr);
  const body = node("tbody");
  for (const {reader, rows} of series) {
    const line = node("tr"), th = node("th", reader.label); th.scope = "row"; line.append(th);
    for (const crop of cropSizes) {
      const td = node("td"), row = rows.find(value => value.crop_size === crop);
      if (row) {
        const metric = samplesPerSecond(row);
        const value = metricStrings(metric), label = node("span", `${crop}²`, "mobile-crop-label");
        label.setAttribute("aria-hidden", "true");
        td.append(label, settingLink(reader, row, value.median, openRow),
          node("span", `${value.min}–${value.max}`, "observed-range"));
      } else td.textContent = "Not matched";
      line.append(td);
    }
    body.append(line);
  }
  table.append(caption, head, body); scroll.append(table); target.append(scroll);
}

function cropOverview(readerList, recipes, openRow) {
  const target = $("crop-comparison");
  target.replaceChildren();
  const recipe = recipes.find(value => value.id === $("overview-recipe").value);
  if (!recipe) return;
  const random = comparableRows(readerList, recipe, "random"), baked = comparableRows(readerList, recipe, "baked");
  $("recipe-description").textContent = `Random: ${recipeLabel(recipe.randomRow)}. Baked: ${recipeLabel(recipe.bakedRow)}. Each line keeps this recipe and follows the same paired queries across machines.`;
  for (const [access, series] of [["random", random], ["baked", baked]]) {
    const maximum = d3.max(series.flatMap(value => value.rows), row => samplesPerSecond(row).max);
    const yDomain = d3.scaleLinear().domain([0, maximum * 1.04]).nice(4).domain();
    const card = node("article", undefined, "comparison-panel");
    card.append(node("h3", access === "random" ? "Random crops" : "Baked samples"));
    target.append(card);
    legend(card, readerList.map(reader => ({label: reader.label})));
    const plot = node("div"); card.append(plot);
    const height = 270, left = 52, top = 14, bottom = 48, right = 18;
    const {svg, width} = chart(plot, height, `${access}: samples per second at a fixed encoding; all readers share this scale, the other read pattern has a different scale`);
    svg.attr("data-access", access).attr("data-maximum", yDomain[1]);
    const x = d3.scalePoint().domain(cropSizes).padding(.3).range([left, width - right]);
    const y = d3.scaleLinear().domain(yDomain).range([height - bottom, top]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(${left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickSize(-(width - left - right)).tickFormat(""));
    svg.append("g").attr("class", "summary-y-axis").attr("transform", `translate(${left},0)`)
      .call(d3.axisLeft(y).ticks(4, "~s"));
    svg.append("g").attr("transform", `translate(0,${height - bottom})`).call(d3.axisBottom(x).tickSizeOuter(0));
    svg.append("text").attr("x", width / 2).attr("y", height - 7).attr("text-anchor", "middle")
      .attr("class", "summary-axis-label").text("Square crop width (pixels)");
    svg.append("text").attr("x", left).attr("y", 11).attr("class", "summary-axis-label").text("Samples/s ↑");
    series.forEach(({reader, rows}, index) => {
      svg.append("path").datum(rows).attr("class", "summary-curve").attr("data-recipe", recipe[access])
        .attr("stroke", colors[index]).attr("stroke-dasharray", [null, "7 3", "2 3"][index])
        .attr("d", d3.line().x(row => x(row.crop_size)).y(row => y(samplesPerSecond(row).median)));
      for (const row of rows) {
        const metric = samplesPerSecond(row), cx = x(row.crop_size), low = y(metric.min), high = y(metric.max);
        svg.append("path").attr("class", "summary-range").attr("stroke", colors[index]).attr("aria-hidden", "true")
          .attr("d", `M${cx},${low}V${high}M${cx - 4},${low}H${cx + 4}M${cx - 4},${high}H${cx + 4}`);
        mark(svg, cx, y(metric.median), symbols[index], colors[index],
          `${reader.label}, ${row.crop_size} square, ${access}: ${metricText(metric)} samples/s`, () => openRow(reader.document, row));
      }
    });
    cropTable(card, series, openRow);
  }
  const cuda = baked.find(series => series.reader.backend === "cuda");
  const a = cuda?.rows.find(row => row.crop_size === 192), b = cuda?.rows.find(row => row.crop_size === 256);
  const fewerSamples = a && b && samplesPerSecond(b).median < samplesPerSecond(a).median;
  const moreBytes = a && b && b.metrics.useful_gib_s.median > a.metrics.useful_gib_s.median;
  $("crop-title").textContent = moreBytes && fewerSamples ? "More image bytes per second can mean fewer samples" : "Crop size changes sample delivery";
  $("crop-finding").textContent = a && b
    ? `${cuda.reader.label}, this baked recipe: 192² delivers ${number(samplesPerSecond(a).median)} samples/s at ${number(a.metrics.useful_gib_s.median)} GiB/s; 256² delivers ${number(samplesPerSecond(b).median)} samples/s at ${number(b.metrics.useful_gib_s.median)} GiB/s.${moreBytes && fewerSamples ? " More image bytes per second, but fewer completed samples." : !moreBytes && fewerSamples ? " Both image throughput and sample delivery fall at 256²." : " The two rates answer different questions about the workload."}`
    : "Sample rates count completed single-plane crops. Compare the same recipe and crop size across readers; CPU output still needs a transfer for GPU training.";
}

function parameterOverview(readerList, openRow) {
  const target = $("parameter-comparison");
  target.replaceChildren();
  const comparisons = [
    {title: "Smaller chunks help random crops", subtitle: "128 × 128 · plain Zstd · level 3",
      labels: ["1 MiB chunks", "16 KiB chunks"],
      find: (reader, small) => reader.rows.find(row => row.access_pattern === "random" && row.crop_size === 128
        && row.setting.codec === "zstd" && row.setting.chunk_bytes === (small ? 16384 : 1048576)),
      note: "Same codec; chunk geometry changes. For uint16: [2, 512, 512] → [1, 64, 128]. This does not isolate chunk bytes from shape, or establish the best size for larger crops."},
    {title: "Smaller Blosc blocks help CUDA", subtitle: "256 × 256 baked · Blosc Zstd · bitshuffle · level 3",
      labels: ["Whole-chunk blocks", "16 KiB blocks"],
      find: (reader, small) => reader.rows.find(row => row.access_pattern === "baked" && row.crop_size === 256
        && row.setting.codec === "blosc-zstd" && (small ? row.setting.block_bytes === 16384 : row.setting.block_mode === "chunk")),
      note: "Same codec and one-sample chunk shape; only the block request changes. CPU results do not show the same gain. EPYC repeat ranges overlap."},
  ];
  for (const comparison of comparisons) {
    const pairs = readerList.map(reader => ({reader, rows: [comparison.find(reader, false), comparison.find(reader, true)]}))
      .filter(pair => pair.rows.every(Boolean));
    if (!pairs.length) continue;
    const card = node("article", undefined, "comparison-panel parameter-panel");
    card.append(node("h3", comparison.title), node("p", comparison.subtitle, "reader-output"));
    legend(card, comparison.labels.map((label, index) => ({label, symbol: index ? "◆" : "○", color: colors[index]})));
    target.append(card);
    const domain = d3.scaleLinear().domain([0, d3.max(pairs.flatMap(pair => pair.rows), row => row.metrics.useful_gib_s.max) * 1.04]).nice(4).domain();
    for (const {reader, rows} of pairs) {
      const group = node("section", undefined, "parameter-pair");
      const ratio = rows[1].metrics.useful_gib_s.median / rows[0].metrics.useful_gib_s.median;
      const heading = node("h4", reader.label);
      heading.append(node("span", `${ratio.toFixed(2)}× with smaller ${comparison === comparisons[0] ? "chunks" : "blocks"}`, "gain"));
      group.append(heading); card.append(group);
      const plot = node("div"); group.append(plot);
      const {svg, width} = chart(plot, 48, `${reader.label}: ${comparison.labels.join(" versus ")}`);
      const x = d3.scaleLinear().domain(domain).range([12, width - 12]);
      for (const tick of d3.ticks(...domain, 4)) svg.append("line").attr("class", "summary-grid")
        .attr("x1", x(tick)).attr("x2", x(tick)).attr("y1", 0).attr("y2", 48);
      rows.forEach((row, index) => {
        const y = index ? 35 : 13;
        range(svg, row.metrics.useful_gib_s, x, y, colors[index]);
        mark(svg, x(row.metrics.useful_gib_s.median), y, index ? d3.symbolDiamond : d3.symbolCircle, colors[index],
          `${reader.label}, ${comparison.labels[index]}: ${metricText(row.metrics.useful_gib_s)} GiB/s`,
          () => openRow(reader.document, row), Boolean(index));
        const value = node("p", undefined, "parameter-value");
        value.append(settingLink(reader, row, `${index ? "◆" : "○"} ${comparison.labels[index]}`, openRow),
          node("span", `${metricText(row.metrics.useful_gib_s)} GiB/s`));
        group.append(value);
      });
    }
    axis(card, domain);
    card.append(node("p", `Compression ratio: ${pairs[0].rows.map(row => `${number(row.compression_fold)}×`).join(" → ")} with smaller ${comparison === comparisons[0] ? "chunks" : "blocks"}. Higher means less storage.`, "comparison-note"),
      node("p", comparison.note, "comparison-note"));
  }
}

export function mountOverview(documents, openRow) {
  const readerList = readers(documents), recipes = fixedRecipes(readerList);
  if (!readerList.length || !recipes.length) return;
  $("overview").hidden = false;
  const select = $("overview-recipe");
  select.replaceChildren(...recipes.map(recipe => new Option(recipe.label, recipe.id)));
  const saved = new URLSearchParams(location.hash.slice(1)).get("recipe");
  if (recipes.some(recipe => recipe.id === saved)) select.value = saved;
  const render = () => {
    layoutOverview(readerList, openRow);
    cropOverview(readerList, recipes, openRow);
    parameterOverview(readerList, openRow);
  };
  select.addEventListener("change", () => {
    const hash = new URLSearchParams(location.hash.slice(1));
    hash.set("recipe", select.value);
    history.replaceState(null, "", `#${hash}`);
    cropOverview(readerList, recipes, openRow);
  });
  window.addEventListener("resize", render);
  render();
}
