#!/usr/bin/env python
"""Train the ChestX-ray14 multi-label classifier.

    python train.py                          # best-known config (320px DenseNet)
    python train.py --arch convnext_tiny     # swap backbone
    python train.py --image-size 224 --batch-size 16   # 8 GB VRAM
    python train.py --limit-batches 3 --epochs 1       # 2-minute smoke test
    python train.py --resume runs/<name>/last.pt

Splits come from the EDA: patient-disjoint train/val, official NIH test held
back entirely for evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from chest_train import config as C          # noqa: E402
from chest_train.data import build_loaders    # noqa: E402
from chest_train.engine import (              # noqa: E402
    _scaler, evaluate, save_checkpoint, train_one_epoch, tune_thresholds,
)
from chest_train.models import BCEWithPosWeight, build_model  # noqa: E402


def parse_args() -> C.TrainConfig:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    d = C.TrainConfig()
    p.add_argument("--arch", default=d.arch)
    p.add_argument("--image-size", type=int, default=d.image_size)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--val-batch-size", type=int, default=d.val_batch_size)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    p.add_argument("--num-workers", type=int, default=d.num_workers)
    p.add_argument("--accum-steps", type=int, default=d.accum_steps)
    p.add_argument("--pos-weight-cap", type=float, default=d.pos_weight_cap)
    p.add_argument("--focal", action="store_true", help="focal BCE instead of plain")
    p.add_argument("--hflip", action="store_true",
                   help="add horizontal flip (mirrors the heart - off by default)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--limit-batches", type=int, default=None,
                   help="stop each epoch after N batches (smoke test)")
    a = p.parse_args()

    cfg = C.TrainConfig(
        arch=a.arch, image_size=a.image_size, batch_size=a.batch_size,
        val_batch_size=a.val_batch_size, epochs=a.epochs, lr=a.lr,
        weight_decay=a.weight_decay, num_workers=a.num_workers,
        accum_steps=a.accum_steps, pos_weight_cap=a.pos_weight_cap,
        use_focal=a.focal, amp=not a.no_amp, pretrained=not a.no_pretrained,
        seed=a.seed, resume=a.resume, limit_batches=a.limit_batches,
        run_name=a.run_name or f"{a.arch}_{a.image_size}",
    )
    cfg.hflip = a.hflip
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # fixed input size -> let cuDNN tune


def main() -> int:
    cfg = parse_args()
    set_seed(cfg.seed)
    cfg.norm_mean, cfg.norm_std = C.load_norm_stats()
    print(f"config: arch={cfg.arch} size={cfg.image_size} batch={cfg.batch_size} "
          f"epochs={cfg.epochs} lr={cfg.lr} weight_decay={cfg.weight_decay} "
          f"patience={cfg.early_stop_patience} norm=({cfg.norm_mean:.4f}, "
          f"{cfg.norm_std:.4f}) run_dir={cfg.run_dir}")

    # A fresh run saves best.pt after its first epoch, so starting one in an
    # existing run dir would silently replace a finished model.
    if not cfg.resume and (cfg.run_dir / "best.pt").exists():
        print(f"ERROR: {cfg.run_dir / 'best.pt'} already exists. Pass --run-name "
              "<new name> for a new run, or --resume to continue this one.")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no CUDA device found - this will be extremely slow.")
    else:
        print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
              f"cuda {torch.version.cuda}")

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_loaders(cfg, splits=("train", "val"))
    print(f"train batches {len(loaders['train'])} | val batches {len(loaders['val'])}")

    model = build_model(cfg.arch, len(cfg.labels), cfg.pretrained, cfg.drop_rate)
    model = model.to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    pw = torch.tensor(C.load_pos_weight(cfg.pos_weight_cap), device=device)
    print("pos_weight (capped):",
          {l: round(float(w), 1) for l, w in zip(cfg.labels, pw)})
    criterion = BCEWithPosWeight(pw, cfg.use_focal, cfg.focal_gamma,
                                 cfg.label_smoothing).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    scaler = _scaler(cfg.amp)

    start_epoch, best, history, step = 0, -1.0, [], 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, history = ck["epoch"] + 1, ck["best"], ck.get("history", [])
        print(f"resumed from {cfg.resume} at epoch {start_epoch} (best {best:.4f})")

    total_steps = cfg.epochs * len(loaders["train"])
    # cosine_lr is driven by the global batch step, so a resume must pick the
    # schedule up where it left off instead of re-running warmup from zero.
    step = start_epoch * len(loaders["train"])
    bad_epochs = 0

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], optimizer, criterion, scaler,
                             device, cfg, epoch, total_steps, step)
        step = tr["step"]
        val, y_true, y_prob = evaluate(model, loaders["val"], device, cfg, criterion)
        mean_auroc = val["auroc_mean"]

        rec = {"epoch": epoch, "train_loss": tr["loss"], "val_loss": val.get("loss"),
               "val_auroc_mean": mean_auroc, "val_auprc_mean": val["auprc_mean"],
               "secs": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"epoch {epoch:02d} | train {tr['loss']:.4f} | val {val.get('loss')} | "
              f"AUROC {mean_auroc:.4f} | AUPRC {val['auprc_mean']:.4f} | "
              f"{rec['secs']}s")

        save_checkpoint(cfg.run_dir / "last.pt", model, optimizer, scaler, cfg,
                        epoch, best, history)

        if mean_auroc > best:
            best, bad_epochs = mean_auroc, 0
            save_checkpoint(cfg.run_dir / "best.pt", model, optimizer, scaler, cfg,
                            epoch, best, history)
            thr = tune_thresholds(y_true, y_prob, cfg.labels)
            (cfg.run_dir / "val_metrics.json").write_text(
                json.dumps({"epoch": epoch, **val, "thresholds": thr}, indent=2),
                encoding="utf-8")
            print(f"  new best -> {cfg.run_dir / 'best.pt'}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"early stop: no val AUROC gain in {bad_epochs} epochs")
                break

    (cfg.run_dir / "history.json").write_text(json.dumps(history, indent=2),
                                              encoding="utf-8")
    print(f"\nbest val AUROC {best:.4f}\nrun dir: {cfg.run_dir}")
    print("next: python evaluate.py --checkpoint "
          f"{cfg.run_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    # Windows spawns dataloader workers; this guard is mandatory there.
    raise SystemExit(main())
