#!/usr/bin/env python
"""Web demo for the ChestX-ray14 classifier: landing page + upload/Grad-CAM tool.

    python -m webapp.server
    python -m webapp.server --checkpoint runs/densenet121_320/best.pt --port 8000

Then open http://127.0.0.1:8000. Uploaded images are processed in memory and
never written to disk.
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "webapp"))

from fastapi import FastAPI, HTTPException, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import UnidentifiedImageError  # noqa: E402

from inference import XrayModel, load_image  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "new_results" / "runs" / "densenet121_320" / "best.pt"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

FINDINGS = {
    "Atelectasis": "Partial collapse or incomplete inflation of lung tissue.",
    "Cardiomegaly": "An enlarged cardiac silhouette relative to the width of the chest.",
    "Consolidation": "Airspaces filled with fluid, pus or cells instead of air.",
    "Edema": "Fluid in the lung tissue and airspaces, often from heart failure.",
    "Effusion": "Fluid collecting in the pleural space around the lung.",
    "Emphysema": "Damaged alveolar walls leaving over-inflated, hyperlucent lungs.",
    "Fibrosis": "Scarring and stiffening of the lung tissue.",
    "Hernia": "Abdominal contents pushing up into the chest, usually hiatal.",
    "Infiltration": "Ill-defined opacity from material denser than air in the lung.",
    "Mass": "A discrete opacity larger than about 3 cm.",
    "Nodule": "A small, rounded opacity up to about 3 cm.",
    "Pleural_Thickening": "Thickened pleural lining, often after prior inflammation.",
    "Pneumonia": "Infection of the lung, usually seen as consolidation.",
    "Pneumothorax": "Air in the pleural space, letting the lung collapse.",
}

state: dict = {}


def _load_model() -> XrayModel:
    ckpt = Path(os.environ.get("CHEST_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}  (pass --checkpoint)")
    print(f"loading {ckpt} ...")
    model = XrayModel(ckpt, os.environ.get("CHEST_DEVICE", "cpu"))
    print(f"ready: {model.cfg.arch} @ {model.cfg.image_size}px, "
          f"{model.metrics_split} AUROC {model.metrics.get('auroc_mean')}")
    return model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state["model"] = _load_model()
    yield
    state.clear()


app = FastAPI(title="LungLens", lifespan=lifespan, docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def landing():
    return FileResponse(STATIC / "index.html")


@app.get("/analyze", include_in_schema=False)
def analyze_page():
    return FileResponse(STATIC / "analyze.html")


@app.get("/api/model")
def model_info():
    info = state["model"].info()
    info["descriptions"] = FINDINGS
    return info


@app.post("/api/predict")
async def predict(file: UploadFile):
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image is larger than 25 MB.")
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        img = load_image(data)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(415, "Could not read that file as an image. "
                                 "Use PNG, JPEG, BMP, TIFF or WebP.")
    if min(img.size) < 64:
        raise HTTPException(422, "Image is too small (minimum 64 px per side).")
    return await run_in_threadpool(state["model"].predict, img)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--device", default="cuda" if _cuda() else "cpu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    os.environ["CHEST_CHECKPOINT"] = a.checkpoint
    os.environ["CHEST_DEVICE"] = a.device

    import uvicorn
    print(f"open http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port)
    return 0


def _cuda() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    raise SystemExit(main())
