#!/usr/bin/env python
"""Evaluate a checkpoint on the official NIH test split and export for serving.

    python evaluate.py --checkpoint runs/densenet121_320/best.pt
    python evaluate.py --checkpoint ... --export     # + TorchScript/ONNX bundle

The official test split is patient-disjoint from train/val (EDA stage 5
confirmed 0 overlap), so these are the numbers to quote.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from chest_train import config as C            # noqa: E402
from chest_train.data import build_loaders      # noqa: E402
from chest_train.engine import evaluate, tune_thresholds  # noqa: E402
from chest_train.models import build_model      # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--export", action="store_true",
                   help="write TorchScript + ONNX + model_card.json for serving")
    p.add_argument("--limit-batches", type=int, default=None)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    saved = ck["config"]

    cfg = C.TrainConfig(**{k: v for k, v in saved.items()
                           if k in C.TrainConfig.__dataclass_fields__})
    cfg.val_batch_size = a.batch_size
    cfg.num_workers = a.num_workers
    cfg.limit_batches = a.limit_batches

    model = build_model(cfg.arch, len(cfg.labels), False, cfg.drop_rate).to(device)
    model.load_state_dict(ck["model"])
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    loaders = build_loaders(cfg, splits=(a.split,))
    m, y_true, y_prob = evaluate(model, loaders[a.split], device, cfg)

    print(f"\n{a.split} split | mean AUROC {m['auroc_mean']:.4f} | "
          f"mean AUPRC {m['auprc_mean']:.4f}\n")
    print(f"{'label':<20}{'AUROC':>8}{'AUPRC':>8}{'n_pos':>8}")
    for name, v in m["per_class"].items():
        print(f"{name:<20}{str(v['auroc']):>8}{str(v['auprc']):>8}{v['n_pos']:>8}")

    out_dir = Path(a.checkpoint).parent
    (out_dir / f"{a.split}_metrics.json").write_text(json.dumps(m, indent=2),
                                                     encoding="utf-8")

    if a.export:
        # Thresholds must come from val, never from test - tuning them on the
        # split you report would leak.
        val_thr_file = out_dir / "val_metrics.json"
        thresholds = (json.loads(val_thr_file.read_text(encoding="utf-8"))
                      .get("thresholds") if val_thr_file.exists()
                      else tune_thresholds(y_true, y_prob, cfg.labels))

        model.eval().to("cpu").to(memory_format=torch.contiguous_format)
        example = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        ts_path = out_dir / "model.torchscript.pt"
        torch.jit.save(torch.jit.trace(model, example), ts_path)

        onnx_path = out_dir / "model.onnx"
        try:
            torch.onnx.export(
                model, example, str(onnx_path), input_names=["image"],
                output_names=["logits"], opset_version=17,
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            )
        except Exception as exc:
            print(f"ONNX export skipped: {exc}")
            onnx_path = None

        # The contract the web platform consumes. Everything it needs to turn
        # a raw PNG into calibrated per-finding calls lives in this one file.
        card = {
            "labels": cfg.labels,
            "arch": cfg.arch,
            "input_size": cfg.image_size,
            "input_channels": 3,
            "preprocessing": {
                "load": "PIL convert('L')  # some files are RGBA",
                "resize": int(cfg.image_size * 1.14),
                "center_crop": cfg.image_size,
                "scale": "ToTensor -> [0,1]",
                "normalize_mean": [cfg.norm_mean] * 3,
                "normalize_std": [cfg.norm_std] * 3,
            },
            "output": "raw logits -> apply sigmoid for per-label probability",
            "thresholds": thresholds,
            "metrics": {a.split: {"auroc_mean": m["auroc_mean"],
                                  "auprc_mean": m["auprc_mean"],
                                  "per_class": m["per_class"]}},
            "artifacts": {"torchscript": ts_path.name,
                          "onnx": onnx_path.name if onnx_path else None},
            "caveats": [
                "Research/triage support only - not a diagnostic device.",
                "Labels are NLP-mined from reports (~90% accurate), so the "
                "ceiling on any metric here is the label noise, not the model.",
                "Official test split is sicker than train: EDA stage 5 measured "
                "Pneumothorax +241%, Consolidation +115%, Effusion +82% "
                "prevalence drift. Expect test AUROC below val AUROC.",
            ],
        }
        (out_dir / "model_card.json").write_text(json.dumps(card, indent=2),
                                                 encoding="utf-8")
        print(f"\nexported: {ts_path.name}, model_card.json"
              + (f", {onnx_path.name}" if onnx_path else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
