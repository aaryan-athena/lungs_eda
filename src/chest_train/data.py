"""Datasets, transforms and loaders.

The EDA already produced everything needed to build the splits, so this module
reads its tables rather than re-deriving anything:
  01_metadata_clean.csv  -> one-hot labels + machine-independent `rel` paths
  05_split_assignment.csv -> patient-disjoint train/val + the official test set
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config as C

if TYPE_CHECKING:
    import pandas as pd


def load_frame() -> pd.DataFrame:
    """Metadata joined to the split assignment, with usable image paths."""
    # Imported here, not at module level: the web app uses build_transforms
    # only, and pandas would cost it ~40 MB of a 512 MB hosting budget.
    import pandas as pd

    for p in (C.METADATA_TABLE, C.SPLIT_TABLE):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run `python run_eda.py` first.")

    df = pd.read_csv(C.METADATA_TABLE)
    splits = pd.read_csv(C.SPLIT_TABLE)[["image_id", "split"]]
    df = df.merge(splits, on="image_id", how="inner")

    # `rel` is relative to DATA_ROOT, so the same tables work on any machine.
    if "rel" in df.columns:
        df["path"] = [str(C.DATA_ROOT / r) for r in df["rel"]]
    elif "path" not in df.columns:
        raise KeyError("metadata table has neither 'rel' nor 'path'")

    missing = [c for c in C.PATHOLOGIES if c not in df.columns]
    if missing:
        raise KeyError(f"label columns missing from metadata table: {missing}")
    return df


class ChestXrayDataset(Dataset):
    """One 1024x1024 greyscale PNG -> (3, S, S) tensor + 14 binary labels."""

    def __init__(self, df: pd.DataFrame, tf: transforms.Compose, labels: list[str]):
        self.paths = df["path"].tolist()
        self.ids = df["image_id"].tolist()
        self.y = df[labels].to_numpy(dtype=np.float32)
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        # convert("L") first: a known subset of this dataset is stored RGBA,
        # which the EDA flags. Without it the tensor shape silently varies.
        with Image.open(self.paths[i]) as im:
            img = im.convert("L")
            x = self.tf(img)
        return x, torch.from_numpy(self.y[i]), self.ids[i]


def build_transforms(cfg: C.TrainConfig):
    """Train/eval transforms.

    Augmentation is deliberately mild. Chest films are anatomically
    standardised, so aggressive geometric jitter destroys the very cues
    (cardiac silhouette size, costophrenic angle) the labels depend on.
    """
    norm = transforms.Normalize([cfg.norm_mean] * 3, [cfg.norm_std] * 3)
    s = cfg.image_size

    train_tf = transforms.Compose([
        # scale floor 0.85 keeps the lung fields in frame; the EDA measured a
        # p95 dead-black border, so there is little margin to crop away.
        transforms.RandomResizedCrop(s, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomRotation(10),
        # NOTE: no horizontal flip by default. It mirrors the heart, and
        # Cardiomegaly is judged from the cardiac silhouette. Enable with
        # --hflip if you want the extra invariance.
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        *([transforms.RandomHorizontalFlip(0.5)] if getattr(cfg, "hflip", False) else []),
        transforms.Grayscale(num_output_channels=3),  # ImageNet trunks want 3ch
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(s * 1.14)),
        transforms.CenterCrop(s),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


def build_loaders(cfg: C.TrainConfig, splits=("train", "val")):
    df = load_frame()
    train_tf, eval_tf = build_transforms(cfg)
    out = {}
    for name in splits:
        part = df[df["split"] == name]
        if not len(part):
            raise ValueError(f"split '{name}' is empty")
        is_train = name == "train"
        ds = ChestXrayDataset(part, train_tf if is_train else eval_tf, cfg.labels)
        out[name] = DataLoader(
            ds,
            batch_size=cfg.batch_size if is_train else cfg.val_batch_size,
            shuffle=is_train,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=is_train,
            # Windows respawns workers per epoch otherwise; this is the single
            # biggest dataloader win on win32.
            persistent_workers=cfg.num_workers > 0,
            prefetch_factor=4 if cfg.num_workers > 0 else None,
        )
    return out
