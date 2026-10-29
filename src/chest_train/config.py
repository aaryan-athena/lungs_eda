"""Training configuration, derived from the EDA wherever a number was measured.

Every non-obvious default here traces back to a specific finding in
outputs/report/EDA_REPORT.md. Those links are written out as comments so the
choices stay auditable instead of looking like folklore.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Same env override the EDA uses, so both halves of the project agree.
DATA_ROOT = Path(
    os.environ.get("CHEST_XRAY_DATA", PROJECT_ROOT / "data" / "raw" / "nih-chest-xrays")
).resolve()

TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "report"
RUNS_DIR = PROJECT_ROOT / "runs"

METADATA_TABLE = TABLES_DIR / "01_metadata_clean.csv"
PREVALENCE_TABLE = TABLES_DIR / "02_label_prevalence.csv"
SPLIT_TABLE = TABLES_DIR / "05_split_assignment.csv"
EDA_SUMMARY = REPORT_DIR / "eda_summary.json"

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]

# Fallback intensity stats. ChestX-ray14 is 8-bit greyscale; the EDA measures
# the real values in stage 4 and we prefer those when the report has them.
FALLBACK_MEAN = 0.5
FALLBACK_STD = 0.25


@dataclass
class TrainConfig:
    # --- data ------------------------------------------------------------
    image_size: int = 320
    # WHY 320 and not the usual 224: the EDA's bounding-box stage measured
    # median lesion area at 4.5% of the frame, and the 10th percentile lesion
    # is only 14.9 px on its short side once resized to 224. At 320 that same
    # lesion is ~21 px, which a 32-px-stride backbone can still represent.
    # 224 trains ~2x faster and is the honest choice if VRAM is tight.
    batch_size: int = 32
    num_workers: int = 8
    val_batch_size: int = 64

    # --- model -----------------------------------------------------------
    arch: str = "densenet121"        # CheXNet lineage; strong CXR baseline
    pretrained: bool = True          # ImageNet init is worth ~3-4 AUROC points
    drop_rate: float = 0.2

    # --- optimisation ----------------------------------------------------
    epochs: int = 20
    lr: float = 2e-4                 # AdamW fine-tuning of a pretrained trunk
    # WHY 1e-2: at 1e-5 two runs peaked at epoch 3 and then overfit (val loss
    # 0.66 -> 1.1 while train loss kept falling). 1e-2 is AdamW's own default.
    weight_decay: float = 1e-2
    warmup_epochs: int = 1
    min_lr: float = 1e-6
    grad_clip: float = 5.0
    accum_steps: int = 1

    # --- loss ------------------------------------------------------------
    # BCE with per-class pos_weight, read from 02_label_prevalence.csv.
    # WHY capped: measured pos_weight ranges 4.6 (Infiltration) to 492.9
    # (Hernia, 227 positives in 112k). Feeding 492 in raw makes the Hernia
    # term dominate the gradient and destabilises every other class. Capping
    # keeps the rebalancing without letting one rare class hijack training.
    pos_weight_cap: float = 20.0
    label_smoothing: float = 0.0
    hflip: bool = False
    use_focal: bool = False
    focal_gamma: float = 2.0

    # --- runtime ---------------------------------------------------------
    amp: bool = True                 # fp16 autocast; ~2x throughput on Ampere
    channels_last: bool = True
    seed: int = 42
    early_stop_patience: int = 5
    monitor: str = "val_auroc_mean"

    # --- bookkeeping -----------------------------------------------------
    run_name: str = "densenet121_320"
    resume: str | None = None
    limit_batches: int | None = None  # smoke-test escape hatch

    # filled in at runtime
    labels: list[str] = field(default_factory=lambda: list(PATHOLOGIES))
    norm_mean: float = FALLBACK_MEAN
    norm_std: float = FALLBACK_STD

    @property
    def run_dir(self) -> Path:
        return RUNS_DIR / self.run_name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_root"] = str(DATA_ROOT)
        return d


def load_pos_weight(cap: float) -> list[float]:
    """Per-class BCE pos_weight from the EDA prevalence table, capped."""
    if not PREVALENCE_TABLE.exists():
        raise FileNotFoundError(
            f"{PREVALENCE_TABLE} not found - run `python run_eda.py` first."
        )
    tbl = pd.read_csv(PREVALENCE_TABLE).set_index("label")
    return [float(min(tbl.loc[c, "pos_weight"], cap)) for c in PATHOLOGIES]


def load_norm_stats() -> tuple[float, float]:
    """Dataset intensity mean/std from EDA stage 4, if that stage succeeded."""
    if EDA_SUMMARY.exists():
        try:
            img = json.loads(EDA_SUMMARY.read_text(encoding="utf-8")).get("images", {})
            mean, std = img.get("normalize_mean_0_1"), img.get("normalize_std_0_1")
            if mean and std:
                return float(mean), float(std)
        except Exception:
            pass
    return FALLBACK_MEAN, FALLBACK_STD
