// Analyzer page: upload -> /api/predict -> ranked findings + Grad-CAM overlay.

const MAX_BYTES = 25 * 1024 * 1024;
const $ = (id) => document.getElementById(id);

const fileInput = $("file-input");
const dropzone = $("dropzone");
const viewer = $("viewer");
const baseImg = $("base-img");
const camCanvas = $("cam-canvas");

let result = null;      // last /api/predict response
let selected = null;    // label whose heatmap is shown
let busy = false;

// ---------- heatmap colour ramp ----------
// Sequential YlOrRd: low attention fades to transparent so the film stays readable.
const STOPS = [[0, [255, 255, 204]], [0.25, [254, 217, 118]], [0.5, [253, 141, 60]], [0.75, [227, 26, 28]], [1, [128, 0, 38]]];
const LUT = (() => {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < STOPS.length - 2 && t > STOPS[k + 1][0]) k++;
    const [t0, c0] = STOPS[k], [t1, c1] = STOPS[k + 1];
    const f = (t - t0) / (t1 - t0);
    for (let j = 0; j < 3; j++) lut[i * 4 + j] = c0[j] + (c1[j] - c0[j]) * f;
    lut[i * 4 + 3] = 255 * 0.9 * Math.pow(Math.max(0, (t - 0.12) / 0.88), 0.85);
  }
  return lut;
})();

const camCache = {};
function camImageData(label) {
  if (camCache[label]) return camCache[label];
  const n = result.cam_size;
  const raw = atob(result.cams[label]);
  const img = new ImageData(n, n);
  for (let i = 0; i < n * n; i++) {
    const v = raw.charCodeAt(i);
    img.data.set(LUT.subarray(v * 4, v * 4 + 4), i * 4);
  }
  const off = document.createElement("canvas");
  off.width = off.height = n;
  off.getContext("2d").putImageData(img, 0, 0);
  return (camCache[label] = off);
}

function drawCam(label) {
  const ctx = camCanvas.getContext("2d");
  ctx.clearRect(0, 0, camCanvas.width, camCanvas.height);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(camImageData(label), 0, 0, camCanvas.width, camCanvas.height);
}

// ---------- selection ----------
function select(label) {
  selected = label;
  drawCam(label);
  $("badge-text").textContent = `Grad-CAM · ${prettyLabel(label)}`;
  $("viewer-badge").hidden = camCanvas.hidden;
  document.querySelectorAll(".pred").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.label === label)));
}

// ---------- rendering ----------
const ICON_ALERT = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8v5M12 17h.01"/><circle cx="12" cy="12" r="10"/></svg>';
const ICON_OK = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>';

function renderResults(data) {
  // Rank by score relative to its own threshold: raw scores are not
  // comparable across findings because class weighting shifts each one.
  const preds = [...data.predictions].sort((a, b) =>
    (b.flagged - a.flagged) || (b.score / b.threshold - a.score / a.threshold));
  const flagged = preds.filter((p) => p.flagged);

  const summary = $("summary");
  summary.className = `summary ${flagged.length ? "flagged" : "clear"}`;
  $("summary-icon").innerHTML = flagged.length ? ICON_ALERT : ICON_OK;
  $("summary-title").textContent = flagged.length
    ? `${flagged.length} finding${flagged.length > 1 ? "s" : ""} above threshold`
    : "No findings above threshold";
  $("summary-text").textContent = flagged.length
    ? `${flagged.map((p) => prettyLabel(p.label)).join(", ")}. Check each heatmap to see what drove the score.`
    : `All ${preds.length} scores are below their validation-tuned thresholds.`;

  $("pred-list").replaceChildren(...preds.map((p) => {
    const btn = el("button", {
      type: "button", class: `pred${p.flagged ? " is-flagged" : ""}`, "data-label": p.label,
      "aria-pressed": "false",
      "aria-label": `${prettyLabel(p.label)}: score ${fmt(p.score, 2)}, threshold ${fmt(p.threshold, 2)}${p.flagged ? ", above threshold" : ""}`,
      onclick: () => select(p.label),
    },
    el("span", { class: "p-name" }, prettyLabel(p.label), p.flagged ? el("span", { class: "badge" }, "▲ Above threshold") : null),
    el("span", { class: "p-score" }, fmt(p.score, 2)),
    el("span", { class: "p-bar" },
      el("span", { class: "p-fill", style: "width:0" }),
      el("span", { class: "p-thr", style: `left:calc(${p.threshold * 100}% - 1px)`, title: `Threshold ${fmt(p.threshold, 2)}` })));
    return el("li", {}, btn);
  }));
  requestAnimationFrame(() => requestAnimationFrame(() => {
    document.querySelectorAll(".pred").forEach((b, i) => {
      b.querySelector(".p-fill").style.width = `${preds[i].score * 100}%`;
    });
  }));

  $("res-meta").replaceChildren(
    el("span", { class: "pill" }, `${(data.elapsed_ms / 1000).toFixed(2)} s inference`),
    el("span", { class: "pill" }, `Input ${data.input.width}×${data.input.height}`),
    el("span", { class: "pill" }, `Model view ${data.input.model_size}×${data.input.model_size}`),
  );

  select(preds[0].label);
}

function setState(state) {
  const empty = state === "empty";
  dropzone.hidden = !empty;
  viewer.hidden = empty;
  $("controls").hidden = state !== "done";
  $("viewer-loading").hidden = state !== "loading";
  $("results-empty").hidden = !empty;
  $("results-loading").hidden = state !== "loading";
  $("results").hidden = state !== "done";
  if (state !== "done") {
    $("viewer-badge").hidden = true;
    camCanvas.getContext("2d").clearRect(0, 0, camCanvas.width, camCanvas.height);
  }
}

$("skeleton-rows").replaceChildren(...Array.from({ length: 7 }, () =>
  el("div", { class: "skeleton", style: "height:38px;margin:10px 0" })));

// ---------- upload ----------
async function analyze(file) {
  if (busy || !file) return;
  if (file.size > MAX_BYTES) return toast("That image is larger than 25 MB.");
  if (file.type && !file.type.startsWith("image/")) return toast("Please choose an image file.");

  busy = true;
  result = null;
  for (const k of Object.keys(camCache)) delete camCache[k];
  const preview = URL.createObjectURL(file);
  baseImg.src = preview;
  setState("loading");

  try {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch("/api/predict", { method: "POST", body });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Server error (HTTP ${res.status})`);
    result = data;
    baseImg.src = data.image;
    setState("done");
    renderResults(data);
  } catch (err) {
    setState("empty");
    toast(err.message || "Something went wrong.");
  } finally {
    URL.revokeObjectURL(preview);
    busy = false;
    fileInput.value = "";
  }
}

fileInput.addEventListener("change", () => analyze(fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
  e.preventDefault();
  dropzone.classList.add("drag");
}));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
}));
dropzone.addEventListener("drop", (e) => analyze(e.dataTransfer.files[0]));
// Dropping anywhere on the page should not navigate away from it.
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  if (!dropzone.contains(e.target) && e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]);
});
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) analyze(item.getAsFile());
});

// ---------- controls ----------
$("opacity").addEventListener("input", (e) => { camCanvas.style.opacity = e.target.value / 100; });
camCanvas.style.opacity = 0.75;

document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
  const showHeat = b.dataset.view === "heatmap";
  camCanvas.hidden = !showHeat;
  $("viewer-badge").hidden = !showHeat;
  document.querySelectorAll("[data-view]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
}));

$("btn-new").addEventListener("click", () => {
  setState("empty");
  fileInput.click();
});

$("btn-download").addEventListener("click", () => {
  if (!result || !selected) return;
  const out = document.createElement("canvas");
  out.width = camCanvas.width;
  out.height = camCanvas.height;
  const ctx = out.getContext("2d");
  ctx.drawImage(baseImg, 0, 0, out.width, out.height);
  if (!camCanvas.hidden) {
    ctx.globalAlpha = Number(camCanvas.style.opacity || 0.75);
    ctx.drawImage(camCanvas, 0, 0);
  }
  const a = document.createElement("a");
  a.download = `lunglens-gradcam-${selected.toLowerCase()}.png`;
  a.href = out.toDataURL("image/png");
  a.click();
});

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), 4200);
}

setState("empty");
