"""Train / validate loops, metrics, checkpointing."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

# torch>=2.4 moved autocast/GradScaler to torch.amp; keep both working.
try:
    from torch.amp import GradScaler, autocast

    def _autocast(enabled: bool):
        return autocast("cuda", enabled=enabled)

    def _scaler(enabled: bool):
        return GradScaler("cuda", enabled=enabled)
except (ImportError, TypeError):  # pragma: no cover - older torch
    from torch.cuda.amp import GradScaler, autocast  # type: ignore

    def _autocast(enabled: bool):
        return autocast(enabled=enabled)

    def _scaler(enabled: bool):
        return GradScaler(enabled=enabled)


def cosine_lr(step: int, total: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> dict:
    """Per-class AUROC and AUPRC.

    AUROC is the headline number the ChestX-ray14 literature reports, but with
    prevalences from 0.2% to 17.7% it flatters a model badly on the rare
    classes, so AUPRC is reported beside it.
    """
    per_class, aurocs, auprcs = {}, [], []
    for i, name in enumerate(labels):
        yt, yp = y_true[:, i], y_prob[:, i]
        if yt.min() == yt.max():          # class absent from this split
            per_class[name] = {"auroc": None, "auprc": None, "n_pos": int(yt.sum())}
            continue
        au = float(roc_auc_score(yt, yp))
        ap = float(average_precision_score(yt, yp))
        per_class[name] = {"auroc": round(au, 4), "auprc": round(ap, 4),
                           "n_pos": int(yt.sum())}
        aurocs.append(au)
        auprcs.append(ap)
    return {
        "auroc_mean": round(float(np.mean(aurocs)), 4) if aurocs else 0.0,
        "auprc_mean": round(float(np.mean(auprcs)), 4) if auprcs else 0.0,
        "per_class": per_class,
    }


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                    labels: list[str]) -> dict[str, float]:
    """Per-class decision threshold maximising F1 on the validation split.

    The web platform needs a binary call per finding. A flat 0.5 is wrong here
    because pos_weight training shifts the operating point, and it shifts by a
    different amount for each class.
    """
    out = {}
    grid = np.arange(0.05, 0.96, 0.01)
    for i, name in enumerate(labels):
        yt = y_true[:, i]
        if yt.min() == yt.max():
            out[name] = 0.5
            continue
        scores = [f1_score(yt, (y_prob[:, i] >= t).astype(int), zero_division=0)
                  for t in grid]
        out[name] = round(float(grid[int(np.argmax(scores))]), 3)
    return out


@torch.no_grad()
def evaluate(model, loader, device, cfg, criterion=None) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    probs, trues, losses = [], [], []
    bar = tqdm(loader, desc="  eval", unit="b", dynamic_ncols=True, leave=False)
    for bi, (x, y, _) in enumerate(bar):
        if cfg.limit_batches and bi >= cfg.limit_batches:
            break
        x = x.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with _autocast(cfg.amp):
            logits = model(x)
            if criterion is not None:
                losses.append(float(criterion(logits, y)))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        trues.append(y.cpu().numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(trues)
    m = compute_metrics(y_true, y_prob, cfg.labels)
    if losses:
        m["loss"] = round(float(np.mean(losses)), 5)
    return m, y_true, y_prob


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, cfg,
                    epoch: int, total_steps: int, step0: int) -> dict:
    model.train()
    running, n, step = 0.0, 0, step0
    bar = tqdm(loader, desc=f"epoch {epoch:02d}", unit="b", dynamic_ncols=True,
               smoothing=0.1)
    optimizer.zero_grad(set_to_none=True)

    for bi, (x, y, _) in enumerate(bar):
        if cfg.limit_batches and bi >= cfg.limit_batches:
            break
        lr = cosine_lr(step, total_steps, cfg.warmup_epochs * len(loader),
                       cfg.lr, cfg.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x = x.to(device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        with _autocast(cfg.amp):
            loss = criterion(model(x), y) / cfg.accum_steps
        scaler.scale(loss).backward()

        if (bi + 1) % cfg.accum_steps == 0:
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running += float(loss) * cfg.accum_steps
        n += 1
        step += 1
        if bi % 20 == 0:
            bar.set_postfix_str(f"loss={running / max(1, n):.4f} lr={lr:.2e}")
    return {"loss": round(running / max(1, n), 5), "step": step}


def save_checkpoint(path: Path, model, optimizer, scaler, cfg, epoch: int,
                    best: float, history: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg.to_dict(),
        "epoch": epoch,
        "best": best,
        "history": history,
    }, path)
