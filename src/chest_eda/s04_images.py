"""STAGE 4 - Pixel-level analysis on a random sample of PNGs.

You cannot open 112k 1024x1024 PNGs casually (~45 GB), so this stage works
on a reproducible random sample (default 2000, set CHEST_XRAY_IMG_SAMPLE).
Everything it measures feeds a concrete preprocessing decision.

What it does, and why
  1. Decode check          -> finds truncated/corrupt PNGs that would crash
                              your DataLoader at epoch 3, hour 6.
  2. Shape + channel count -> ChestX-ray14 is nominally 8-bit greyscale
                              1024x1024, but a known subset is stored as
                              4-channel RGBA. Any loader must force a single
                              channel or it will silently break tensor shape.
  3. Intensity statistics  -> the dataset mean/std needed for Normalize().
  4. Degenerate-image scan -> near-constant (blank) images, very low
                              contrast, and suspected photometric inversion.
  5. Border/padding scan   -> how much of the frame is dead black margin,
                              i.e. how much a centre crop can safely remove.
  6. Average-hash          -> exact/near duplicate pairs within the sample.
  7. A contact sheet + intensity histograms to eyeball the sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from tqdm import tqdm

from . import config
from .io_utils import get_logger, save_fig, save_table

ImageFile.LOAD_TRUNCATED_IMAGES = False  # we WANT truncation to raise
log = get_logger(__name__)


def _ahash(a: np.ndarray, size: int = 8) -> str:
    small = np.asarray(
        Image.fromarray(a).resize((size, size), Image.BILINEAR), dtype=np.float32
    )
    bits = (small > small.mean()).flatten()
    return "".join("1" if b else "0" for b in bits)


def _inspect(path: str) -> dict:
    """Open one PNG and return every per-image measurement we need."""
    rec: dict = {"path": path, "ok": False}
    try:
        with Image.open(path) as im:
            rec.update(
                mode=im.mode, width=im.width, height=im.height,
                n_channels=len(im.getbands()),
            )
            arr = np.asarray(im)
        if arr.ndim == 3:                     # RGBA/RGB -> take the first plane
            arr = arr[..., 0]
        arr = arr.astype(np.uint8, copy=False)

        flat = arr.ravel()
        h, w = arr.shape
        centre = arr[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
        corners = np.concatenate([
            arr[:32, :32].ravel(), arr[:32, -32:].ravel(),
            arr[-32:, :32].ravel(), arr[-32:, -32:].ravel(),
        ]).mean()

        rec.update(
            ok=True,
            dtype=str(arr.dtype),
            mean=float(flat.mean()),
            std=float(flat.std()),
            p01=float(np.percentile(flat, 1)),
            p99=float(np.percentile(flat, 99)),
            vmin=int(flat.min()),
            vmax=int(flat.max()),
            # dead black margin -> how much a centre crop can remove safely
            black_frac=float((flat < 5).mean()),
            saturated_frac=float((flat > 250).mean()),
            # a normal CXR is brighter in the middle than in the corners;
            # a negative value suggests photometric inversion
            centre_minus_corner=float(centre - corners),
            ahash=_ahash(arr),
        )
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def analyse_images(df: pd.DataFrame, n_sample: int | None = None):
    have = df.dropna(subset=["path"]) if "path" in df.columns else df.iloc[:0]
    if not len(have):
        log.warning("stage 4 skipped: no image files resolved on disk")
        return pd.DataFrame(), {"skipped": "no image files found"}

    n = min(n_sample or config.IMAGE_SAMPLE_SIZE, len(have))
    sample = have.sample(n, random_state=config.RANDOM_SEED)
    log.info("inspecting %d of %d images", n, len(have))

    recs = [_inspect(p) for p in tqdm(sample["path"].tolist(), unit="img", ncols=80)]
    px = pd.DataFrame(recs).merge(
        sample[["image_id", "path", "view", "n_findings"]], on="path", how="left"
    )
    save_table(px, "04_pixel_stats")

    good = px[px["ok"]].copy()
    dup = good.groupby("ahash")["image_id"].apply(list)
    dup = dup[dup.map(len) > 1]

    on_spec = (good["width"] == config.EXPECTED_SIZE[0]) & (
        good["height"] == config.EXPECTED_SIZE[1]
    )
    shapes = (
        good.groupby(["width", "height"]).size().sort_values(ascending=False)
        .head(10).rename_axis(["w", "h"]).reset_index(name="n").to_dict("records")
    )

    report = {
        "n_sampled": int(n),
        "n_unreadable": int((~px["ok"]).sum()),
        "unreadable_examples": px.loc[~px["ok"], ["image_id", "error"]]
                                 .head(10).to_dict("records"),
        "shapes": shapes,
        "off_spec_shape_share": float((~on_spec).mean()),
        "channel_counts": good["n_channels"].value_counts().to_dict(),
        "pil_modes": good["mode"].value_counts().to_dict(),
        "multi_channel_share": float((good["n_channels"] > 1).mean()),
        # --- the numbers you paste into transforms.Normalize ---------------
        "intensity_mean_0_255": round(float(good["mean"].mean()), 3),
        "intensity_std_0_255": round(float(good["std"].mean()), 3),
        "normalize_mean_0_1": round(float(good["mean"].mean() / 255), 4),
        "normalize_std_0_1": round(float(good["std"].mean() / 255), 4),
        # --- quality flags -------------------------------------------------
        "n_near_blank": int((good["std"] < 10).sum()),
        "n_low_contrast": int(((good["p99"] - good["p01"]) < 60).sum()),
        "n_suspected_inverted": int((good["centre_minus_corner"] < 0).sum()),
        "mean_black_border_frac": round(float(good["black_frac"].mean()), 4),
        "p95_black_border_frac": round(float(good["black_frac"].quantile(0.95)), 4),
        "n_duplicate_hash_groups": int(len(dup)),
        "duplicate_examples": list(dup.head(5)),
        "est_full_dataset_gb": round(len(have) * 1024 * 1024 / 1e9, 1),
    }

    _plot_intensity(good)
    _plot_contact_sheet(good)
    return px, report


def _plot_intensity(good: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(good["mean"], bins=50, color="#3b7dd8")
    axes[0].set_title("Per-image mean intensity")
    axes[1].hist(good["std"], bins=50, color="#d8853b")
    axes[1].set_title("Per-image std (contrast)")
    axes[2].scatter(good["mean"], good["std"], s=6, alpha=0.35, color="#5aa469")
    axes[2].set_xlabel("mean")
    axes[2].set_ylabel("std")
    axes[2].set_title("mean vs std - outliers are bad scans")
    save_fig(fig, "04_intensity_stats")


def _plot_contact_sheet(good: pd.DataFrame, k: int = 12) -> None:
    rows = good.sample(min(k, len(good)), random_state=config.RANDOM_SEED)
    cols = 4
    nrows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3 * cols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes, rows.iterrows()):
        with Image.open(row["path"]) as im:
            ax.imshow(np.asarray(im.convert("L").resize((256, 256))), cmap="gray")
        ax.set_title(f"{row['n_findings']} finding(s) / {row['view']}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(rows):]:
        ax.axis("off")
    save_fig(fig, "04_contact_sheet")
