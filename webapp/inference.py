"""Model loading, preprocessing and Grad-CAM for the web demo.

Preprocessing is deliberately not re-implemented here. The eval transform is
taken from chest_train.data.build_transforms using the checkpoint's own saved
config, so an uploaded image is seen exactly the way the validation split was
(same resize/crop geometry, same normalisation stats).
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from chest_train import config as C
from chest_train.data import build_transforms
from chest_train.models import build_model

CAM_GRID = 64        # heatmap side sent to the browser; it upsamples smoothly
DISPLAY_SIZE = 640   # side of the preview image returned to the browser

# Guard against decompression bombs; a 1024x1024 CXR is ~1 MP.
Image.MAX_IMAGE_PIXELS = 60_000_000


def load_image(data: bytes) -> Image.Image:
    """Bytes -> single-channel PIL image, the way training read PNGs."""
    im = Image.open(io.BytesIO(data))
    im.load()
    im = ImageOps.exif_transpose(im)
    # 16-bit greyscale exports (common outside NIH) would be clipped to white
    # by convert("L"); window them to 8 bits on robust percentiles instead.
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(im, dtype=np.float32)
        lo, hi = np.percentile(arr, (0.5, 99.5))
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        return Image.fromarray(arr.astype(np.uint8), "L")
    return im.convert("L")


def _split_model(model: torch.nn.Module, arch: str):
    """(features, head) such that head(features(x)) == model(x).

    Grad-CAM needs the last convolutional feature map. Splitting the forward
    pass avoids hooks, which clash with DenseNet's in-place ReLU.
    """
    if arch == "densenet121":
        def feats(x):
            return F.relu(model.features(x))

        def head(a):
            return model.classifier(torch.flatten(F.adaptive_avg_pool2d(a, 1), 1))
    elif arch == "resnet50":
        m = model

        def feats(x):
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            return m.layer4(m.layer3(m.layer2(m.layer1(x))))

        def head(a):
            return m.fc(torch.flatten(m.avgpool(a), 1))
    elif arch == "convnext_tiny":
        feats = model.features

        def head(a):
            return model.classifier(model.avgpool(a))
    elif arch == "efficientnet_b0":
        feats = model.features

        def head(a):
            return model.classifier(torch.flatten(model.avgpool(a), 1))
    else:
        raise ValueError(f"Grad-CAM not wired for arch {arch!r}")
    return feats, head


def _display_transform(eval_tf: transforms.Compose, size: int) -> transforms.Compose:
    """Same crop geometry as eval_tf, rendered at `size` for a crisp preview."""
    resize = next(t for t in eval_tf.transforms if isinstance(t, transforms.Resize))
    crop = next(t for t in eval_tf.transforms if isinstance(t, transforms.CenterCrop))
    r = resize.size if isinstance(resize.size, int) else resize.size[0]
    ratio = r / crop.size[0]
    return transforms.Compose([transforms.Resize(round(size * ratio)),
                               transforms.CenterCrop(size)])


def _to_data_url(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class XrayModel:
    def __init__(self, checkpoint: Path, device: str = "cpu"):
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        ck = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        saved = ck["config"]
        self.cfg = C.TrainConfig(**{k: v for k, v in saved.items()
                                    if k in C.TrainConfig.__dataclass_fields__})
        self.labels: list[str] = list(self.cfg.labels)
        self.epoch = ck.get("epoch")

        model = build_model(self.cfg.arch, len(self.labels), False, self.cfg.drop_rate)
        model.load_state_dict(ck["model"])
        self.model = model.to(self.device).eval()
        self._feats, self._head = _split_model(self.model, self.cfg.arch)

        _, self.eval_tf = build_transforms(self.cfg)
        self.display_tf = _display_transform(self.eval_tf, DISPLAY_SIZE)

        run_dir = self.checkpoint.parent
        val = _read_json(run_dir / "val_metrics.json")
        test = _read_json(run_dir / "test_metrics.json")
        # Thresholds always come from val - never from the split being reported.
        self.thresholds = {l: float((val or {}).get("thresholds", {}).get(l, 0.5))
                           for l in self.labels}
        self.metrics_split = "test" if test else ("val" if val else None)
        self.metrics = test or val or {}

        self._lock = threading.Lock()

    def info(self) -> dict:
        return {
            "arch": self.cfg.arch,
            "image_size": self.cfg.image_size,
            "epoch": self.epoch,
            "checkpoint": str(self.checkpoint),
            "labels": self.labels,
            "thresholds": self.thresholds,
            "metrics_split": self.metrics_split,
            "auroc_mean": self.metrics.get("auroc_mean"),
            "auprc_mean": self.metrics.get("auprc_mean"),
            "per_class": self.metrics.get("per_class", {}),
        }

    def predict(self, img: Image.Image) -> dict:
        t0 = time.perf_counter()
        x = self.eval_tf(img).unsqueeze(0).to(self.device)

        with self._lock:
            with torch.no_grad():
                a = self._feats(x)
            # Only the head is differentiated: d(logit)/d(feature map) never
            # needs the trunk, so all 14 CAMs cost about one forward pass.
            a = a.detach().requires_grad_(True)
            with torch.enable_grad():
                logits = self._head(a)
                cams = []
                for c in range(len(self.labels)):
                    (g,) = torch.autograd.grad(logits[0, c], a, retain_graph=True)
                    w = g.mean(dim=(2, 3), keepdim=True)
                    cams.append(F.relu((w * a).sum(dim=1, keepdim=True)))
            probs = torch.sigmoid(logits.detach())[0].cpu().numpy()

        cam = torch.cat(cams).detach()                             # (K,1,h,w)
        # Bicubic hides the 10x10 feature grid; it can overshoot below 0.
        cam = F.interpolate(cam, size=(CAM_GRID, CAM_GRID), mode="bicubic",
                            align_corners=False)[:, 0].clamp_min(0)
        peak = cam.flatten(1).max(dim=1).values.clamp_min(1e-8)
        cam = (cam / peak[:, None, None]).clamp(0, 1).cpu().numpy()
        cam_u8 = (cam * 255).round().astype(np.uint8)

        preds = []
        for i, label in enumerate(self.labels):
            p, thr = float(probs[i]), self.thresholds[label]
            preds.append({"label": label, "score": round(p, 4),
                          "threshold": thr, "flagged": p >= thr})

        return {
            "predictions": preds,
            "image": _to_data_url(self.display_tf(img)),
            "cam_size": CAM_GRID,
            "cams": {l: base64.b64encode(cam_u8[i].tobytes()).decode()
                     for i, l in enumerate(self.labels)},
            "input": {"width": img.width, "height": img.height,
                      "model_size": self.cfg.image_size},
            "elapsed_ms": round((time.perf_counter() - t0) * 1000),
        }


def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
