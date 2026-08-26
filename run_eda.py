#!/usr/bin/env python
"""Entry point for the ChestX-ray14 EDA pipeline.

    python run_eda.py                    # all stages
    python run_eda.py --sample 500       # faster pixel stage
    python run_eda.py --no-images        # metadata only (no PNGs needed)
    python run_eda.py --stages 1,2,3     # subset of stages

Stages
    1  metadata  - load + validate Data_Entry_2017.csv, one-hot the labels
    2  labels    - prevalence, imbalance, pos_weight, co-occurrence
    3  patients  - demographics, confounders, patient-leakage check
    4  images    - pixel stats on a sample: shapes, channels, mean/std, junk
    5  splits    - official train/test lists, drift, bounding boxes
    6  report    - render EDA_REPORT.md + eda_summary.json
"""
from __future__ import annotations

import argparse
import sys
import traceback

sys.path.insert(0, "src")

from chest_eda import config  # noqa: E402
from chest_eda.io_utils import get_logger  # noqa: E402
from chest_eda.report import build_report  # noqa: E402
from chest_eda.s01_metadata import load_metadata  # noqa: E402
from chest_eda.s02_labels import analyse_labels  # noqa: E402
from chest_eda.s03_patients import analyse_patients  # noqa: E402
from chest_eda.s04_images import analyse_images  # noqa: E402
from chest_eda.s05_splits import analyse_splits  # noqa: E402

log = get_logger("run_eda")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", help="override the dataset directory")
    ap.add_argument("--sample", type=int, default=None,
                    help="images to inspect in stage 4 (default 2000)")
    ap.add_argument("--no-images", action="store_true",
                    help="skip stage 4 and the disk cross-reference")
    ap.add_argument("--stages", default="1,2,3,4,5,6",
                    help="comma-separated stage numbers to run")
    args = ap.parse_args()

    if args.data_root:
        config.DATA_ROOT = __import__("pathlib").Path(args.data_root).resolve()
        config.METADATA_CSV = config.DATA_ROOT / "Data_Entry_2017.csv"
        config.BBOX_CSV = config.DATA_ROOT / "BBox_List_2017.csv"
        config.TRAIN_VAL_LIST = config.DATA_ROOT / "train_val_list.txt"
        config.TEST_LIST = config.DATA_ROOT / "test_list.txt"

    stages = {s.strip() for s in args.stages.split(",")}
    log.info("data root: %s", config.DATA_ROOT)
    if not config.DATA_ROOT.exists():
        log.error("data root does not exist. See README.md ('Where to put the data').")
        return 2

    results: dict = {}

    # Stage 1 is mandatory - everything else is a view over its output.
    df, results["metadata"] = load_metadata(check_disk=not args.no_images)

    for num, name, fn in (
        ("2", "labels", lambda: analyse_labels(df)[1]),
        ("3", "patients", lambda: analyse_patients(df)),
        ("4", "images", lambda: analyse_images(df, args.sample)[1]),
        ("5", "splits", lambda: analyse_splits(df)),
    ):
        if num not in stages or (num == "4" and args.no_images):
            log.info("stage %s (%s) skipped", num, name)
            continue
        log.info("--- stage %s: %s ---", num, name)
        try:
            results[name] = fn()
        except Exception:
            log.error("stage %s (%s) failed:\n%s", num, name, traceback.format_exc())
            results[name] = {"error": traceback.format_exc(limit=2)}

    if "6" in stages:
        log.info("--- stage 6: report ---")
        build_report(results)
        print("\n" + "=" * 70)
        print(f"Report:  {config.REPORT_DIR / 'EDA_REPORT.md'}")
        print(f"Tables:  {config.TBL_DIR}")
        print(f"Figures: {config.FIG_DIR}")
        print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
