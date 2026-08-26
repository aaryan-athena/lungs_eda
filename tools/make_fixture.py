#!/usr/bin/env python
"""Generate a tiny synthetic stand-in for the NIH dataset.

Lets you smoke-test the whole pipeline in seconds, before (or without) the
45 GB download. It mimics the real quirks on purpose: multi-image patients,
ages > 100, RGBA files, a truncated PNG, a blank image, an inverted image.

    python tools/make_fixture.py --out data/raw/fixture --n 400
    python run_eda.py --data-root data/raw/fixture --sample 200
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
          "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
          "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"]
# roughly the real prevalences
PREV = [0.10, 0.025, 0.04, 0.02, 0.12, 0.02, 0.015, 0.002, 0.18, 0.05,
        0.056, 0.03, 0.013, 0.047]


def synth_cxr(size: int, rng: np.random.Generator) -> np.ndarray:
    """A crude chest-like image: bright centre, dark border, some texture."""
    y, x = np.mgrid[0:size, 0:size] / size
    body = np.exp(-(((x - 0.5) / 0.32) ** 2 + ((y - 0.55) / 0.42) ** 2))
    lungs = 0.45 * (np.exp(-(((x - 0.33) / 0.11) ** 2 + ((y - 0.5) / 0.2) ** 2))
                    + np.exp(-(((x - 0.67) / 0.11) ** 2 + ((y - 0.5) / 0.2) ** 2)))
    img = 235 * body - 150 * lungs + rng.normal(0, 8, (size, size))
    img[:int(size * 0.03), :] = 0
    img[-int(size * 0.03):, :] = 0
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/fixture")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--size", type=int, default=256, help="px (real data is 1024)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    random.seed(a.seed)
    root = Path(a.out)
    imgdir = root / "images_001" / "images"
    imgdir.mkdir(parents=True, exist_ok=True)

    rows, names = [], []
    patient = 0
    i = 0
    while i < a.n:
        patient += 1
        for followup in range(rng.integers(1, 5)):     # multi-image patients
            if i >= a.n:
                break
            name = f"{patient:08d}_{followup:03d}.png"
            pos = [l for l, p in zip(LABELS, PREV) if rng.random() < p]
            age = int(rng.normal(48, 16))
            if rng.random() < 0.01:                    # the >100 age bug
                age = int(rng.choice([148, 411, 155]))
            rows.append({
                "Image Index": name,
                "Finding Labels": "|".join(pos) if pos else "No Finding",
                "Follow-up #": followup,
                "Patient ID": patient,
                "Patient Age": max(1, age),
                "Patient Gender": "M" if rng.random() < 0.56 else "F",
                "View Position": "PA" if rng.random() < 0.6 else "AP",
                "OriginalImage[Width": 1024, "Height]": 1024,
                "OriginalImagePixelSpacing[x": 0.143, "y]": 0.143,
            })
            arr = synth_cxr(a.size, rng)
            r = rng.random()
            if r < 0.02:                               # blank
                arr[:] = 12
            elif r < 0.04:                             # inverted
                arr = 255 - arr
            im = Image.fromarray(arr)
            if rng.random() < 0.05:                    # RGBA variant
                im = im.convert("RGBA")
            im.save(imgdir / name)
            names.append(name)
            i += 1

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(root / "Data_Entry_2017.csv", index=False)

    # a deliberately truncated PNG
    victim = imgdir / names[0]
    victim.write_bytes(victim.read_bytes()[: len(victim.read_bytes()) // 2])

    # patient-disjoint official-style split lists
    pids = df["Patient ID"].unique()
    test_pids = set(pids[int(len(pids) * 0.8):])
    is_test = df["Patient ID"].isin(test_pids)
    (root / "test_list.txt").write_text("\n".join(df.loc[is_test, "Image Index"]))
    (root / "train_val_list.txt").write_text("\n".join(df.loc[~is_test, "Image Index"]))

    # a few bounding boxes
    bb = []
    for name in rng.choice(names, size=min(40, len(names)), replace=False):
        w = float(rng.integers(60, 400)); h = float(rng.integers(60, 400))
        bb.append({"Image Index": name,
                   "Finding Label": rng.choice(["Atelectasis", "Mass", "Nodule",
                                                "Effusion", "Cardiomegaly"]),
                   "Bbox [x": float(rng.integers(0, 600)), "y": float(rng.integers(0, 600)),
                   "w": w, "h]": h})
    pd.DataFrame(bb).to_csv(root / "BBox_List_2017.csv", index=False)

    print(f"fixture written: {root.resolve()}  ({len(df)} images, {patient} patients)")


if __name__ == "__main__":
    main()
