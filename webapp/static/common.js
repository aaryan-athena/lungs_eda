// Shared helpers for both pages: theme toggle, label formatting, model info.

(function initTheme() {
  const btn = document.getElementById("theme-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const root = document.documentElement;
    const isDark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = isDark ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (_) { /* storage blocked */ }
  });
})();

const prettyLabel = (l) => l.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const fmt = (v, d = 3) => (v == null ? "–" : Number(v).toFixed(d));

let _modelInfo = null;
async function getModelInfo() {
  if (!_modelInfo) {
    _modelInfo = fetch("/api/model").then((r) => {
      if (!r.ok) throw new Error(`model info: HTTP ${r.status}`);
      return r.json();
    });
  }
  return _modelInfo;
}

function splitName(split) {
  return split === "test" ? "official NIH test split"
    : split === "val" ? "patient-held-out validation split" : "no metrics file";
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c != null) node.append(c instanceof Node ? c : document.createTextNode(c));
  }
  return node;
}
