"""STAGE 1 - Load the metadata CSV, validate it, and normalise it.

Why this comes first: every later stage (labels, patients, splits, image
checks) is a groupby on this one dataframe. Getting the schema and the
integrity checks right here means the rest of the pipeline can assume clean
input.

What it does
  1. Locates Data_Entry_2017.csv (Kaggle also ships a *_v2020 variant).
  2. Renames the awkward original columns to snake_case.
  3. Drops the trailing unnamed column pandas creates from the CSV's
     trailing comma.
  4. Integrity checks: row count, duplicate Image Index, nulls, impossible
     ages (the CSV contains ages like 411 - a known data bug), label tokens
     outside the known 14 + "No Finding".
  5. One-hot encodes the multi-label "Finding Labels" field into 14 binary
     columns  -> this is the actual CNN target matrix.
  6. Cross-references the metadata against the PNGs actually on disk.
"""
from __future__ import annotations

import pandas as pd

from . import config
from .io_utils import get_logger, index_images, save_table

log = get_logger(__name__)

RENAME = {
    "Image Index": "image_id",
    "Finding Labels": "finding_labels",
    "Follow-up #": "followup_num",
    "Patient ID": "patient_id",
    "Patient Age": "age",
    "Patient Gender": "sex",
    "View Position": "view",
    "OriginalImage[Width": "width",
    "Height]": "height",
    "OriginalImagePixelSpacing[x": "spacing_x",
    "y]": "spacing_y",
}


def _resolve_metadata_csv():
    candidates = [
        config.METADATA_CSV,
        config.DATA_ROOT / "Data_Entry_2017_v2020.csv",
        *sorted(config.DATA_ROOT.glob("Data_Entry_*.csv")),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No Data_Entry_2017*.csv found under {config.DATA_ROOT}. "
        "See README.md for where to put the Kaggle download."
    )


def load_metadata(check_disk: bool = True) -> tuple[pd.DataFrame, dict]:
    csv_path = _resolve_metadata_csv()
    df = pd.read_csv(csv_path)
    log.info("read %s -> %d rows x %d cols", csv_path.name, *df.shape)

    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df = df.rename(columns=RENAME)

    report: dict = {"csv": str(csv_path), "n_rows": int(len(df))}

    # --- integrity -------------------------------------------------------
    report["n_rows_expected"] = config.EXPECTED_N_IMAGES
    report["row_count_matches_publication"] = len(df) == config.EXPECTED_N_IMAGES
    report["duplicate_image_ids"] = int(df["image_id"].duplicated().sum())
    report["null_counts"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    report["n_patients"] = int(df["patient_id"].nunique())

    # Known data bug: a handful of rows carry ages in the hundreds.
    bad_age = df["age"].gt(100) | df["age"].lt(0)
    report["implausible_ages"] = int(bad_age.sum())
    report["implausible_age_values"] = sorted(df.loc[bad_age, "age"].unique().tolist())
    df["age_clean"] = df["age"].where(~bad_age)  # NaN, not deleted: the image is still usable

    # Unexpected sex codes / views are worth knowing before you stratify on them.
    report["sex_values"] = df["sex"].value_counts().to_dict()
    report["view_values"] = df["view"].value_counts().to_dict()

    # --- multi-label one-hot --------------------------------------------
    tokens = df["finding_labels"].str.split("|")
    known = set(config.PATHOLOGIES) | {config.NO_FINDING}
    unknown = {t for row in tokens for t in row} - known
    report["unknown_label_tokens"] = sorted(unknown)

    for label in config.PATHOLOGIES:
        df[label] = tokens.apply(lambda ts, l=label: int(l in ts))
    df["n_findings"] = df[config.PATHOLOGIES].sum(axis=1)
    df["is_normal"] = (df["n_findings"] == 0).astype(int)

    # Consistency: "No Finding" in the text must mean zero positive labels.
    txt_normal = df["finding_labels"].eq(config.NO_FINDING)
    report["no_finding_inconsistencies"] = int((txt_normal != df["is_normal"].astype(bool)).sum())

    # --- disk cross-reference -------------------------------------------
    if check_disk:
        paths = index_images()
        report["n_png_on_disk"] = int(len(paths))
        if len(paths):
            df = df.merge(paths.rename(columns={"Image Index": "image_id"}),
                          on="image_id", how="left")
            report["metadata_rows_missing_file"] = int(df["path"].isna().sum())
            report["files_not_in_metadata"] = int(
                len(set(paths["Image Index"]) - set(df["image_id"]))
            )
        else:
            log.warning("no PNGs found - running in metadata-only mode")
            df["path"] = pd.NA
            report["metadata_rows_missing_file"] = int(len(df))

    save_table(df, "01_metadata_clean")
    return df, report
