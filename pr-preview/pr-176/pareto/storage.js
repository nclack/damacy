import {metricStrings, recipeLabel, storagePerSample} from "./comparisons.mjs?v=74e63ec5e1a3";

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
  for (const [key, label, factor] of [["bytes", "Storage bytes / sample (KiB)", 1024], ["reads", "Storage reads / sample", 1]]) {
    const section = window.document.createElement("section");
    section.className = "storage-work-grid";
    const title = window.document.createElement("h3");
    title.textContent = `${label} · lower is better`;
    section.append(title); wrapper.append(section);
    const width = Math.max(220, section.clientWidth), left = 10, right = 12;
    const x = d3.scaleLinear().domain([0, scales[key] * 1.05 / factor]).nice(4).range([left, width - right]);
    for (const row of ordered) {
      const metric = storagePerSample(document, row)[key];
      const formatted = metricStrings({median: metric.median / factor, min: metric.min / factor, max: metric.max / factor});
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
      value.textContent = formatted.median;
      values.append(button, value); group.append(values); section.append(group);
      const svg = d3.select(group).append("svg").attr("viewBox", `0 0 ${width} 34`)
        .attr("role", "img").attr("class", "storage-work").attr("data-metric", key)
        .attr("data-setting", row.id).attr("data-median", metric.median / factor)
        .attr("data-min", metric.min / factor).attr("data-max", metric.max / factor)
        .attr("aria-label", `${label}: ${formatted.median}, observed ${formatted.min}–${formatted.max}, ${metric.n} repetitions`);
      for (const tick of x.ticks(4)) svg.append("line").attr("class", "summary-grid")
        .attr("x1", x(tick)).attr("x2", x(tick)).attr("y1", 0).attr("y2", 34);
      svg.append("path").attr("class", "summary-range").attr("stroke", "var(--blue)")
        .attr("d", `M${x(metric.min / factor)},17H${x(metric.max / factor)}M${x(metric.min / factor)},12V22M${x(metric.max / factor)},12V22`);
      svg.append("circle").attr("cx", x(metric.median / factor)).attr("cy", 17).attr("r", 4)
        .attr("fill", "var(--blue)");
    }
    const svg = d3.select(section).append("svg").attr("viewBox", `0 0 ${width} 35`).attr("aria-hidden", "true");
    svg.append("g").attr("class", "storage-axis").attr("data-metric", key).attr("transform", "translate(0,1)")
      .call(d3.axisBottom(x).ticks(4).tickSizeOuter(0));
  }
}
