"""STAGE 3 - Patient-level structure and demographics.

This is the stage that saves the project. 112,120 images come from only
~30,805 patients, i.e. ~3.6 images per patient. If you split randomly by
*image*, the same patient's follow-up scans land in both train and test and
your AUC is inflated by identity leakage. Any split must be grouped by
patient_id.

What it does
  1. Images-per-patient distribution (the leakage magnitude).
  2. Demographics: age distribution (after cleaning the >100 bug), sex,
     view position (PA vs AP - AP films are portable/sicker patients and
     correlate with disease, a classic shortcut feature).
  3. Prevalence broken down by sex and by view -> spots confounders your CNN
     could latch onto instead of anatomy.
  4. A demonstration of how bad image-level splitting is: measures patient
     overlap for a naive random 80/20 split vs a GroupShuffleSplit.
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


def analyse_patients(df: pd.DataFrame) -> dict:
    per_patient = df.groupby("patient_id").size()
    save_table(
        per_patient.value_counts().sort_index().rename_axis("images_per_patient")
        .reset_index(name="n_patients"),
        "03_images_per_patient",
    )

    report = {
        "n_patients": int(per_patient.size),
        "images_per_patient_mean": float(per_patient.mean()),
        "images_per_patient_median": float(per_patient.median()),
        "images_per_patient_max": int(per_patient.max()),
        "share_patients_with_multiple_images": float((per_patient > 1).mean()),
        "share_images_from_multi_image_patients": float(
            per_patient[per_patient > 1].sum() / per_patient.sum()
        ),
        "age": _describe(df["age_clean"]),
        "sex_counts": df["sex"].value_counts().to_dict(),
        "view_counts": df["view"].value_counts().to_dict(),
        "followup_max": int(df["followup_num"].max()),
    }

    # --- confounder checks ----------------------------------------------
    by_sex = df.groupby("sex")[P].mean().T
    by_sex["abs_diff"] = (by_sex.iloc[:, 0] - by_sex.iloc[:, 1]).abs()
    save_table(by_sex.round(4), "03_prevalence_by_sex", index=True)

    by_view = df.groupby("view")[P].mean().T
    save_table(by_view.round(4), "03_prevalence_by_view", index=True)
    report["view_confounding"] = {
        "normal_rate_by_view": df.groupby("view")["is_normal"].mean().round(4).to_dict(),
        "mean_findings_by_view": df.groupby("view")["n_findings"].mean().round(3).to_dict(),
        "largest_sex_gap": by_sex["abs_diff"].idxmax(),
    }

    # --- leakage demonstration ------------------------------------------
    report["split_leakage"] = _leakage_check(df)

    _plot_demographics(df, per_patient)
    _plot_prevalence_by_group(by_view, "03_prevalence_by_view", "Prevalence by view position")
    return report


def _describe(s: pd.Series) -> dict:
    d = s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    return {k: round(float(v), 2) for k, v in d.items()}


def _leakage_check(df: pd.DataFrame, test_size: float = 0.2) -> dict:
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_size))
    naive_train = set(df.iloc[idx[:cut]]["patient_id"])
    naive_test = set(df.iloc[idx[cut:]]["patient_id"])
    naive_overlap = naive_train & naive_test

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.RANDOM_SEED)
    tr, te = next(gss.split(df, groups=df["patient_id"]))
    grouped_overlap = set(df.iloc[tr]["patient_id"]) & set(df.iloc[te]["patient_id"])

    return {
        "naive_image_split_leaked_patients": len(naive_overlap),
        "naive_image_split_leaked_test_images": int(
            df.iloc[idx[cut:]]["patient_id"].isin(naive_overlap).sum()
        ),
        "naive_image_split_leaked_test_share": float(
            df.iloc[idx[cut:]]["patient_id"].isin(naive_overlap).mean()
        ),
        "grouped_split_leaked_patients": len(grouped_overlap),
        "verdict": "split by patient_id (GroupShuffleSplit / official lists), never by image",
    }


def _plot_demographics(df: pd.DataFrame, per_patient: pd.Series) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    axes[0, 0].hist(df["age_clean"].dropna(), bins=40, color="#3b7dd8")
    axes[0, 0].set_title("Patient age (implausible >100 removed)")
    axes[0, 0].set_xlabel("years")

    df["sex"].value_counts().plot.bar(ax=axes[0, 1], color="#5aa469", rot=0)
    axes[0, 1].set_title("Sex")

    df["view"].value_counts().plot.bar(ax=axes[1, 0], color="#d8853b", rot=0)
    axes[1, 0].set_title("View position (AP = portable, sicker patients)")

    axes[1, 1].hist(per_patient, bins=range(1, 40), color="#8e5ad8")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title(f"Images per patient (max {per_patient.max()})")
    axes[1, 1].set_xlabel("images")

    save_fig(fig, "03_demographics")


def _plot_prevalence_by_group(tbl: pd.DataFrame, name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    tbl.plot.barh(ax=ax)
    ax.set_xlabel("prevalence")
    ax.set_title(title)
    save_fig(fig, name)
