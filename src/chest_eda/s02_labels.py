"""STAGE 2 - Label structure: prevalence, imbalance, co-occurrence.

Why it matters for a CNN: ChestX-ray14 is *multi-label* (an image can be
Effusion + Atelectasis at once), and severely imbalanced (Hernia ~0.2% vs
Infiltration ~18%). Those two facts decide your loss function, your
final activation, and your metric - before you write a single layer.

What it does
  1. Per-label positive counts, prevalence, and imbalance ratio (neg/pos).
  2. BCE `pos_weight` per label = N_neg / N_pos, ready to paste into
     torch.nn.BCEWithLogitsLoss(pos_weight=...).
  3. Label cardinality: how many findings co-occur per image.
  4. 14x14 co-occurrence matrix + normalised conditional P(B|A), which
     exposes label pairs a model will confuse.
  5. Figures: prevalence bar, cardinality histogram, co-occurrence heatmap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config
from .io_utils import get_logger, save_fig, save_table

log = get_logger(__name__)
P = config.PATHOLOGIES


def analyse_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n = len(df)
    pos = df[P].sum().astype(int)

    prev = pd.DataFrame({
        "label": P,
        "n_positive": pos.values,
        "prevalence": (pos / n).values,
        "n_negative": (n - pos).values,
        # pos_weight for BCEWithLogitsLoss: how much to up-weight a positive
        "pos_weight": ((n - pos) / pos.replace(0, np.nan)).values,
    }).sort_values("n_positive", ascending=False).reset_index(drop=True)
    prev["imbalance_ratio_1_to"] = prev["pos_weight"].round(1)
    save_table(prev, "02_label_prevalence")

    card = df["n_findings"].value_counts().sort_index()
    card_tbl = pd.DataFrame({"n_findings": card.index, "n_images": card.values})
    card_tbl["share"] = card_tbl["n_images"] / n
    save_table(card_tbl, "02_label_cardinality")

    # --- co-occurrence ---------------------------------------------------
    M = df[P].to_numpy(dtype=np.int32)
    co = pd.DataFrame(M.T @ M, index=P, columns=P)          # raw counts
    save_table(co, "02_cooccurrence_counts", index=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = co.div(np.diag(co), axis=0)                   # P(col | row)
    save_table(cond.round(4), "02_cooccurrence_conditional", index=True)

    report = {
        "n_images": int(n),
        "n_normal_images": int(df["is_normal"].sum()),
        "normal_share": float(df["is_normal"].mean()),
        "mean_findings_per_image": float(df["n_findings"].mean()),
        "max_findings_per_image": int(df["n_findings"].max()),
        "rarest_label": prev.iloc[-1]["label"],
        "rarest_label_positives": int(prev.iloc[-1]["n_positive"]),
        "most_common_label": prev.iloc[0]["label"],
        "worst_imbalance_1_to": float(prev["pos_weight"].max()),
        "pos_weight": {r.label: round(float(r.pos_weight), 2) for r in prev.itertuples()},
        "top_cooccurring_pairs": _top_pairs(co),
    }

    _plot_prevalence(prev)
    _plot_cardinality(card_tbl)
    _plot_heatmap(cond)
    return prev, report


def _top_pairs(co: pd.DataFrame, k: int = 10) -> list[dict]:
    m = co.where(np.triu(np.ones(co.shape), 1).astype(bool)).stack()
    return [
        {"pair": f"{a} + {b}", "n_images": int(v)}
        for (a, b), v in m.sort_values(ascending=False).head(k).items()
    ]


def _plot_prevalence(prev: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(prev["label"][::-1], prev["n_positive"][::-1], color="#3b7dd8")
    ax.set_xscale("log")
    ax.set_xlabel("positive images (log scale)")
    ax.set_title("Label prevalence - ChestX-ray14 (log x: the imbalance is ~100x)")
    for y, (v, p) in enumerate(zip(prev["n_positive"][::-1], prev["prevalence"][::-1])):
        ax.text(v * 1.05, y, f"{v:,} ({p:.1%})", va="center", fontsize=8)
    ax.set_xlim(right=prev["n_positive"].max() * 4)
    save_fig(fig, "02_label_prevalence")


def _plot_cardinality(card: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(card["n_findings"].astype(str), card["n_images"], color="#d8853b")
    ax.set_xlabel("number of simultaneous findings")
    ax.set_ylabel("images")
    ax.set_title("Label cardinality (0 = 'No Finding')")
    save_fig(fig, "02_label_cardinality")


def _plot_heatmap(cond: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cond.to_numpy(), cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(cond)), cond.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cond)), cond.index, fontsize=8)
    ax.set_title("P(column | row) - label co-occurrence")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save_fig(fig, "02_cooccurrence_heatmap")
