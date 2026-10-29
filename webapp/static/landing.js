// Landing page: fills metrics, the per-finding chart and the findings grid.

const METRICS = {
  auroc: { title: "AUROC per finding", sub: "Axis starts at 0.5, the score of random guessing.", min: 0.5, max: 1, ticks: [0.5, 0.6, 0.7, 0.8, 0.9, 1] },
  auprc: { title: "AUPRC per finding", sub: "Baseline is each finding's prevalence, so rare findings score low.", min: 0, max: 0.6, ticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] },
};

let rows = [];
let metric = "auroc";
const tooltip = document.getElementById("tooltip");

function renderChart() {
  const m = METRICS[metric];
  const chart = document.getElementById("chart");
  document.getElementById("chart-title").textContent = m.title;
  document.getElementById("chart-sub").textContent = m.sub;
  const pct = (v) => `${Math.max(0, Math.min(1, (v - m.min) / (m.max - m.min))) * 100}%`;

  const sorted = [...rows].sort((a, b) => b[metric] - a[metric]);
  chart.replaceChildren(
    el("div", { class: "hbar-gridlines" }, m.ticks.map((t) => el("i", { style: `left:${pct(t)}` }))),
    ...sorted.map((r) => {
      const fill = el("div", { class: "hbar-fill", style: "width:0" });
      const row = el("div", { class: "hbar-row" },
        el("span", { class: "lbl", title: r.name }, r.name),
        el("div", { class: "hbar-track" }, fill),
        el("span", { class: "val" }, fmt(r[metric], 2)));
      row.addEventListener("mousemove", (e) => showTip(e, r));
      row.addEventListener("mouseleave", hideTip);
      requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = pct(r[metric]); }));
      return row;
    }),
    el("div", { class: "hbar-axis" }, el("span"),
      el("div", { class: "hbar-ticks" }, m.ticks.map((t) => el("span", { style: `left:${pct(t)}` }, String(t)))),
      el("span")),
  );

  const table = document.getElementById("chart-table");
  table.replaceChildren(
    el("thead", {}, el("tr", {}, el("th", {}, "Finding"), el("th", { class: "num" }, "AUROC"),
      el("th", { class: "num" }, "AUPRC"), el("th", { class: "num" }, "Positives"))),
    el("tbody", {}, sorted.map((r) => el("tr", {}, el("td", {}, r.name),
      el("td", { class: "num" }, fmt(r.auroc)), el("td", { class: "num" }, fmt(r.auprc)),
      el("td", { class: "num" }, r.n_pos.toLocaleString())))),
  );
}

function showTip(e, r) {
  tooltip.replaceChildren(
    el("div", { class: "t-title" }, r.name),
    el("div", { class: "t-row" }, "AUROC", el("b", {}, fmt(r.auroc))),
    el("div", { class: "t-row" }, "AUPRC", el("b", {}, fmt(r.auprc))),
    el("div", { class: "t-row" }, "Positive images", el("b", {}, r.n_pos.toLocaleString())),
    el("div", { class: "t-row" }, "Threshold", el("b", {}, fmt(r.threshold, 2))),
  );
  const x = Math.min(e.clientX + 14, innerWidth - tooltip.offsetWidth - 8);
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${e.clientY + 14}px`;
  tooltip.classList.add("show");
}
function hideTip() { tooltip.classList.remove("show"); }

document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
  metric = b.dataset.metric;
  document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  renderChart();
}));

document.getElementById("table-toggle").addEventListener("click", (e) => {
  const table = document.getElementById("chart-table");
  const chart = document.getElementById("chart");
  const showTable = table.hidden;
  table.hidden = !showTable;
  chart.hidden = showTable;
  e.target.textContent = showTable ? "Show as chart" : "Show as table";
});

getModelInfo().then((info) => {
  rows = info.labels
    .filter((l) => info.per_class[l] && info.per_class[l].auroc != null)
    .map((l) => ({ label: l, name: prettyLabel(l), threshold: info.thresholds[l], ...info.per_class[l] }));

  const split = splitName(info.metrics_split);
  document.getElementById("stat-auroc").textContent = fmt(info.auroc_mean);
  document.getElementById("stat-auroc-sub").textContent = info.metrics_split === "test" ? "official test split" : "validation split";
  document.getElementById("mini-auroc").textContent = fmt(info.auroc_mean);
  document.getElementById("mini-auprc").textContent = fmt(info.auprc_mean);
  document.getElementById("chart-split").textContent = `Source: ${split}`;
  document.getElementById("mini-split").textContent = info.metrics_split === "test"
    ? "The official NIH test list: patients never seen in training or validation."
    : "A patient-disjoint validation split. These are the numbers the checkpoint was selected on, so expect the official test split to score somewhat lower.";
  document.getElementById("spec-arch").textContent = info.arch === "densenet121" ? "DenseNet-121" : info.arch;
  if (info.epoch != null) document.getElementById("spec-epoch").textContent = `Epoch ${info.epoch + 1}, best validation mean AUROC`;
  document.querySelectorAll(".js-size").forEach((n) => { n.textContent = info.image_size; });

  renderChart();

  document.getElementById("findings-grid").replaceChildren(...info.labels.map((l) => {
    const pc = info.per_class[l] || {};
    return el("article", { class: "card finding" },
      el("h3", {}, prettyLabel(l), el("span", { class: "auc" }, pc.auroc != null ? `AUROC ${fmt(pc.auroc, 2)}` : "")),
      el("p", {}, info.descriptions[l] || ""));
  }));
}).catch((err) => {
  document.getElementById("chart-split").textContent = `Could not load metrics (${err.message}).`;
});
