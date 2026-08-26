"""STAGE 5 - Official splits, distribution drift, and the bounding boxes.

The dataset ships two text files, train_val_list.txt (~86,524 images) and
test_list.txt (~25,596). Using them is what makes your numbers comparable to
published ChestX-ray14 results; rolling your own split makes you
incomparable to every paper.

What it does
  1. Verifies the two lists cover the metadata exactly once each.
  2. Confirms zero patient overlap between train_val and test (the official
     split is patient-disjoint - your own val split must be too).
  3. Measures per-label prevalence drift train_val vs test. Large drift on a
     rare label means your validation AUC for it will be noisy.
  4. Carves a patient-grouped validation set out of train_val and reports
     the resulting prevalence drift, so you can see the split is sane.
  5. Analyses BBox_List_2017.csv: 984 hand-annotated boxes over 8 of the 14
     classes. Useful later for localisation / Grad-CAM sanity checks, and it
     tells you how large lesions are relative to the frame - which is the
     real argument against downscaling to 224x224.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit

from . import config
from .io_utils import get_logger, save_fig, save_table

log = get_logger(__name__)
P = config.PATHOLOGIES


def _read_list(path) -> set[str]:
    if not path.exists():
        log.warning("missing %s", path.name)
        return set()
    return {ln.strip() for ln in path.read_text().splitlines() if ln.strip()}


def analyse_splits(df: pd.DataFrame) -> dict:
    report: dict = {}
    train_val = _read_list(config.TRAIN_VAL_LIST)
    test = _read_list(config.TEST_LIST)

    if train_val and test:
        df = df.copy()
        df["official_split"] = np.where(
            df["image_id"].isin(test), "test",
            np.where(df["image_id"].isin(train_val), "train_val", "UNASSIGNED"),
        )
        counts = df["official_split"].value_counts().to_dict()
        pat = df.groupby("official_split")["patient_id"].apply(set)
        overlap = pat.get("train_val", set()) & pat.get("test", set())

        report["official"] = {
            "n_train_val_images": int(counts.get("train_val", 0)),
            "n_test_images": int(counts.get("test", 0)),
            "unassigned_images": int(counts.get("UNASSIGNED", 0)),
            "list_ids_not_in_metadata": len((train_val | test) - set(df["image_id"])),
            "patients_train_val": len(pat.get("train_val", set())),
            "patients_test": len(pat.get("test", set())),
            "patient_overlap": len(overlap),
            "test_image_share": round(counts.get("test", 0) / len(df), 4),
        }

        drift = df.groupby("official_split")[P].mean().T
        if {"train_val", "test"}.issubset(drift.columns):
            drift["abs_drift"] = (drift["train_val"] - drift["test"]).abs()
            drift["rel_drift_pct"] = (
                (drift["test"] - drift["train_val"]) / drift["train_val"] * 100
            ).round(1)
            drift = drift.sort_values("abs_drift", ascending=False)
            save_table(drift.round(4), "05_official_split_drift", index=True)
            report["official"]["largest_drift_labels"] = (
                drift["rel_drift_pct"].head(5).to_dict()
            )
            _plot_drift(drift)

        # --- 3) patient-grouped val split carved out of train_val ---------
        tv = df[df["official_split"] == "train_val"]
        if len(tv):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.125,
                                    random_state=config.RANDOM_SEED)
            tr_idx, va_idx = next(gss.split(tv, groups=tv["patient_id"]))
            tr, va = tv.iloc[tr_idx], tv.iloc[va_idx]
            sub = pd.DataFrame({
                "train": tr[P].mean(), "val": va[P].mean(),
            })
            sub["abs_drift"] = (sub["train"] - sub["val"]).abs()
            save_table(sub.round(4), "05_proposed_train_val_drift", index=True)
            report["proposed_val_split"] = {
                "train_images": len(tr), "val_images": len(va),
                "train_patients": tr["patient_id"].nunique(),
                "val_patients": va["patient_id"].nunique(),
                "patient_overlap": len(set(tr["patient_id"]) & set(va["patient_id"])),
                "max_label_abs_drift": round(float(sub["abs_drift"].max()), 4),
                "worst_drift_label": sub["abs_drift"].idxmax(),
            }
            save_table(
                pd.concat([
                    tr[["image_id", "patient_id"]].assign(split="train"),
                    va[["image_id", "patient_id"]].assign(split="val"),
                    df.loc[df["official_split"] == "test", ["image_id", "patient_id"]]
                      .assign(split="test"),
                ]),
                "05_split_assignment",
            )
    else:
        report["official"] = {"skipped": "train_val_list.txt / test_list.txt not found"}

    report["bbox"] = _analyse_bbox()
    return report


def _analyse_bbox() -> dict:
    if not config.BBOX_CSV.exists():
        return {"skipped": "BBox_List_2017.csv not found"}
    bb = pd.read_csv(config.BBOX_CSV)
    bb = bb.drop(columns=[c for c in bb.columns if str(c).startswith("Unnamed")])
    bb.columns = ["image_id", "label", "x", "y", "w", "h"][: len(bb.columns)]

    # box area as a share of the 1024x1024 frame -> the resolution argument
    bb["area_frac"] = (bb["w"] * bb["h"]) / (config.EXPECTED_SIZE[0] * config.EXPECTED_SIZE[1])
    bb["min_side_px"] = bb[["w", "h"]].min(axis=1)
    # how many pixels that lesion's short side survives as at 224x224
    bb["min_side_at_224"] = bb["min_side_px"] * config.CNN_INPUT_SIZE / config.EXPECTED_SIZE[0]

    per_label = bb.groupby("label").agg(
        n_boxes=("image_id", "size"),
        median_area_frac=("area_frac", "median"),
        median_min_side_px=("min_side_px", "median"),
        median_min_side_at_224=("min_side_at_224", "median"),
    ).round(3)
    save_table(per_label, "05_bbox_by_label", index=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot([g["area_frac"].values for _, g in bb.groupby("label")],
               tick_labels=[k for k, _ in bb.groupby("label")], vert=True)
    ax.set_ylabel("box area / frame area")
    ax.set_title("Annotated lesion size relative to the 1024x1024 frame")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save_fig(fig, "05_bbox_area")

    return {
        "n_boxes": int(len(bb)),
        "n_images_with_boxes": int(bb["image_id"].nunique()),
        "labels_covered": sorted(bb["label"].unique().tolist()),
        "median_area_frac": round(float(bb["area_frac"].median()), 4),
        "p10_area_frac": round(float(bb["area_frac"].quantile(0.10)), 4),
        "median_min_side_at_224px": round(float(bb["min_side_at_224"].median()), 1),
        "p10_min_side_at_224px": round(float(bb["min_side_at_224"].quantile(0.10)), 1),
        "per_label": per_label.reset_index().to_dict("records"),
    }


def _plot_drift(drift: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    idx = np.arange(len(drift))
    ax.barh(idx - 0.2, drift["train_val"], height=0.4, label="train_val")
    ax.barh(idx + 0.2, drift["test"], height=0.4, label="test")
    ax.set_yticks(idx, drift.index, fontsize=8)
    ax.set_xlabel("prevalence")
    ax.set_title("Official split: label prevalence drift")
    ax.legend()
    save_fig(fig, "05_official_split_drift")
