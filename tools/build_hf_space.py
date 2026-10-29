#!/usr/bin/env python
"""Assemble a ready-to-push Hugging Face Space for the web demo.

    python tools/build_hf_space.py
    python tools/build_hf_space.py --checkpoint runs/<run>/best.pt --out dist/hf_space

Writes <out>/ with the Dockerfile, the web app, the chest_train package it
imports, and a slimmed model. If <out> is a clone of your Space repo, its .git
folder is kept, so you can rebuild and push repeatedly.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "deploy" / "hf_space"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "requirements.txt")


def export_model(checkpoint: Path, dest: Path) -> None:
    """Keep only what inference needs: weights, config and epoch.

    Training checkpoints also carry the AdamW state (2x the weights), which is
    only useful for --resume and would triple the download.
    """
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    dest.mkdir(parents=True)
    torch.save({"model": ck["model"], "config": ck["config"], "epoch": ck.get("epoch")},
               dest / "model.pt")
    # Thresholds and metrics are read from files next to the checkpoint.
    for name in ("val_metrics.json", "test_metrics.json"):
        if (checkpoint.parent / name).exists():
            shutil.copy2(checkpoint.parent / name, dest / name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="runs/densenet121_320_wd1e-2/best.pt")
    p.add_argument("--out", default="dist/hf_space")
    a = p.parse_args()

    checkpoint = (ROOT / a.checkpoint).resolve()
    out = (ROOT / a.out).resolve()
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}")
        return 1
    if not (checkpoint.parent / "val_metrics.json").exists():
        print(f"ERROR: {checkpoint.parent / 'val_metrics.json'} is missing; "
              "the app needs it for per-finding thresholds.")
        return 1

    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():          # clean rebuild, but keep the Space's git repo
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    for f in TEMPLATE.iterdir():
        shutil.copy2(f, out / f.name)
    shutil.copytree(ROOT / "webapp", out / "webapp", ignore=IGNORE)
    shutil.copytree(ROOT / "src" / "chest_train", out / "src" / "chest_train", ignore=IGNORE)
    export_model(checkpoint, out / "model")

    size_mb = (out / "model" / "model.pt").stat().st_size / 1e6
    print(f"built {out}")
    print(f"  model: {checkpoint.relative_to(ROOT)} -> model/model.pt ({size_mb:.1f} MB)")
    print(f"  metrics: {', '.join(sorted(f.name for f in (out / 'model').glob('*.json')))}")
    print("next: see 'Deploying to Hugging Face Spaces' in README.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
