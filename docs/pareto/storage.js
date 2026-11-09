import {fileReadMetrics, metricStrings, recipeLabel} from "./comparisons.mjs";

export function storagePanel(target, document, rows, scales, select, selected) {
  const wrapper = window.document.createElement("section");
  wrapper.className = "panel storage-panel";
  const heading = window.document.createElement("h2");
  heading.textContent = rows[0].backend.toUpperCase();
  const description = window.document.createElement("p");
  description.textContent = rows[0].backend === "cuda" ? "Output in GPU memory" : "Output in system RAM";
  wrapper.append(heading, description);
  target.append(wrapper);
  const ordered = rows.slice().sort((a, b) => recipeLabel(a).localeCompare(recipeLabel(b)));
  for (const [key, label, unit, definition] of [
    ["requests", "File read requests / sample", "", "Issued data-file reads divided by delivered crops. One request can serve multiple crops."],
    ["amplification", "Read amplification", "×", "File bytes requested divided by uncompressed crop bytes in the source data type. Compression can bring this below 1×."],
  ]) {
    const section = window.document.createElement("section");
    section.className = "storage-work-grid";
    const title = window.document.createElement("h3");
    title.textContent = `${label} · lower is better`;
    const description = window.document.createElement("p");
    description.textContent = definition;
    section.append(title, description); wrapper.append(section);
    const width = Math.max(220, section.clientWidth), left = 10, right = 12;
    const x = d3.scaleLinear().domain([0, scales[key] * 1.05]).nice(4).range([left, width - right]);
    for (const row of ordered) {
      const metric = fileReadMetrics(document, row)[key];
      const formatted = metricStrings(metric);
      const group = window.document.createElement("div");
      group.className = "metric-row";
      const values = window.document.createElement("div");
      values.className = "work-values";
      const button = window.document.createElement("button");
      button.type = "button";
      button.textContent = recipeLabel(row);
      button.dataset.setting = row.id;
      button.setAttribute("aria-controls", "detail");
      button.setAttribute("aria-pressed", row.id === selected);
      button.addEventListener("click", () => select(row));
      const value = window.document.createElement("span");
      value.textContent = `${formatted.median}${unit}`;
      values.append(button, value); group.append(values); section.append(group);
      const svg = d3.select(group).append("svg").attr("viewBox", `0 0 ${width} 34`)
        .attr("role", "img").attr("class", "storage-work").attr("data-metric", key)
        .attr("data-setting", row.id).attr("data-median", metric.median)
        .attr("data-min", metric.min).attr("data-max", metric.max)
        .attr("aria-label", `${label}: ${formatted.median}${unit}, observed ${formatted.min}–${formatted.max}${unit}, ${metric.n} repetitions`);
      for (const tick of x.ticks(4)) svg.append("line").attr("class", "summary-grid")
        .attr("x1", x(tick)).attr("x2", x(tick)).attr("y1", 0).attr("y2", 34);
      svg.append("path").attr("class", "summary-range").attr("stroke", "var(--blue)")
        .attr("d", `M${x(metric.min)},17H${x(metric.max)}M${x(metric.min)},12V22M${x(metric.max)},12V22`);
      svg.append("circle").attr("cx", x(metric.median)).attr("cy", 17).attr("r", 4)
        .attr("fill", "var(--blue)");
    }
    const svg = d3.select(section).append("svg").attr("viewBox", `0 0 ${width} 35`).attr("aria-hidden", "true");
    svg.append("g").attr("class", "storage-axis").attr("data-metric", key).attr("transform", "translate(0,1)")
      .call(d3.axisBottom(x).ticks(4).tickFormat(value => `${d3.format("~g")(value)}${unit}`).tickSizeOuter(0));
  }
}
