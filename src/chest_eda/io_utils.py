"""Small shared helpers: logging, saving tables/figures, image indexing."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
import pandas as pd

from . import config

_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT, stream=sys.stdout)
    return logging.getLogger(name)


log = get_logger(__name__)


def save_table(df: pd.DataFrame, name: str, index: bool = False) -> Path:
    """Persist any analysis table as CSV so results are auditable/diffable."""
    path = config.TBL_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    log.info("table -> %s  (%d rows)", path.name, len(df))
    return path


def save_fig(fig: plt.Figure, name: str) -> Path:
    path = config.FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    log.info("figure -> %s", path.name)
    return path


def save_json(obj: dict, name: str) -> Path:
    path = config.REPORT_DIR / f"{name}.json"
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    log.info("json -> %s", path.name)
    return path


def index_images(force: bool = False) -> pd.DataFrame:
    """Build (and cache) a map of Image Index -> absolute path on disk.

    Kaggle splits the PNGs over images_001..images_012, so the metadata CSV
    alone cannot tell you where a file lives. This walk is the bridge.
    """
    cache = config.INTERIM_DIR / "image_paths.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    seen: dict[str, str] = {}
    for pattern in config.IMAGE_GLOBS:
        for p in config.DATA_ROOT.glob(pattern):
            seen.setdefault(p.name, str(p))
        if seen:  # first pattern that matches anything wins; avoids a full ** walk
            break

    df = pd.DataFrame({"Image Index": list(seen), "path": list(seen.values())})
    log.info("indexed %d PNG files under %s", len(df), config.DATA_ROOT)
    if len(df):
        try:
            df.to_parquet(cache, index=False)
        except Exception as exc:  # pyarrow missing -> fall back to csv
            log.warning("parquet cache failed (%s); using csv", exc)
            df.to_csv(cache.with_suffix(".csv"), index=False)
    return df
