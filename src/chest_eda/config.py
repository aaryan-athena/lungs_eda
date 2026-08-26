"""Central configuration: paths, constants, dataset facts.

Every other module imports from here so that moving the dataset means
editing exactly one file (or setting CHEST_XRAY_DATA env var).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Project layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where the Kaggle download lives. Override with:
#   PowerShell:  $env:CHEST_XRAY_DATA = "D:\datasets\nih-chest-xrays"
DATA_ROOT = Path(
    os.environ.get("CHEST_XRAY_DATA", PROJECT_ROOT / "data" / "raw" / "nih-chest-xrays")
).resolve()

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TBL_DIR = OUT_DIR / "tables"
REPORT_DIR = OUT_DIR / "report"

for _d in (INTERIM_DIR, FIG_DIR, TBL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Expected files inside DATA_ROOT --------------------------------------
METADATA_CSV = DATA_ROOT / "Data_Entry_2017.csv"
BBOX_CSV = DATA_ROOT / "BBox_List_2017.csv"
TRAIN_VAL_LIST = DATA_ROOT / "train_val_list.txt"
TEST_LIST = DATA_ROOT / "test_list.txt"

# Kaggle ships the 112k PNGs as images_001 ... images_012, each with an
# inner "images/" folder. We glob rather than hard-code.
IMAGE_GLOBS = ("images_*/images/*.png", "images/*.png", "**/*.png")

# --- Dataset constants ----------------------------------------------------
# The 14 pathology labels of ChestX-ray14. "No Finding" is the 15th token in
# the CSV but is the *absence* of all 14, not a class of its own.
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]
NO_FINDING = "No Finding"

# Native PNG size for this dataset (a few files deviate -> we check).
EXPECTED_SIZE = (1024, 1024)

# Sanity anchors from the published dataset description.
EXPECTED_N_IMAGES = 112_120
EXPECTED_N_PATIENTS = 30_805

# --- Analysis knobs -------------------------------------------------------
IMAGE_SAMPLE_SIZE = int(os.environ.get("CHEST_XRAY_IMG_SAMPLE", 2000))
RANDOM_SEED = 42
CNN_INPUT_SIZE = 224  # what the preprocessing recommendation is benchmarked against
