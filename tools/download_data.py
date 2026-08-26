#!/usr/bin/env python
"""Fetch the NIH ChestX-ray14 dataset from Kaggle (~45 GB).

Two options, both need Kaggle credentials (see README.md).

    # A) kagglehub - downloads to its own cache, then you symlink/point at it
    python tools/download_data.py --method kagglehub

    # B) kaggle CLI - downloads straight into data/raw/nih-chest-xrays
    python tools/download_data.py --method cli --dest data/raw/nih-chest-xrays

Both are resumable-ish: re-running skips what is already there. Prefer a
drive with >=60 GB free (45 GB extracted + the zip during transfer).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SLUG = "nih-chest-xrays/data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["kagglehub", "cli"], default="kagglehub")
    ap.add_argument("--dest", default="data/raw/nih-chest-xrays")
    args = ap.parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if args.method == "kagglehub":
        try:
            import kagglehub
        except ImportError:
            print("pip install kagglehub", file=sys.stderr)
            return 1
        path = Path(kagglehub.dataset_download(SLUG))
        print(f"\ndownloaded to kagglehub cache:\n  {path}")
        print("\nEither point the pipeline at it directly:")
        print(f'  $env:CHEST_XRAY_DATA = "{path}"')
        print(f"  python run_eda.py")
        print("\nor link it into the project (run an elevated shell on Windows):")
        print(f'  New-Item -ItemType SymbolicLink -Path "{dest}" -Target "{path}"')
        return 0

    if not shutil.which("kaggle"):
        print("pip install kaggle, then place kaggle.json in %USERPROFILE%\\.kaggle\\",
              file=sys.stderr)
        return 1
    cmd = ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(dest), "--unzip"]
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
