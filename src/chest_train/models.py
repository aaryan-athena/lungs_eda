"""Backbones with a 14-logit multi-label head."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

_FACTORY = {
    # name -> (constructor, weights enum attr, classifier attribute path)
    "densenet121":    (models.densenet121,    "DenseNet121_Weights",    "classifier"),
    "resnet50":       (models.resnet50,       "ResNet50_Weights",       "fc"),
    "convnext_tiny":  (models.convnext_tiny,  "ConvNeXt_Tiny_Weights",  "classifier"),
    "efficientnet_b0": (models.efficientnet_b0, "EfficientNet_B0_Weights", "classifier"),
}


def build_model(arch: str, n_classes: int, pretrained: bool = True,
                drop_rate: float = 0.2) -> nn.Module:
    if arch not in _FACTORY:
        raise ValueError(f"unknown arch {arch!r}; choose from {list(_FACTORY)}")
    ctor, weights_attr, head = _FACTORY[arch]

    weights = None
    if pretrained:
        weights = getattr(models, weights_attr).DEFAULT
    model = ctor(weights=weights)

    # Swap the ImageNet head for a 14-way multi-label head. No sigmoid here:
    # BCEWithLogitsLoss needs raw logits for numerical stability.
    if arch == "densenet121":
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(in_f, n_classes))
    elif arch == "resnet50":
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(in_f, n_classes))
    elif arch == "convnext_tiny":
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, n_classes)
    elif arch == "efficientnet_b0":
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(in_f, n_classes))
    return model


class BCEWithPosWeight(nn.Module):
    """Class-balanced BCE, optionally focal.

    pos_weight comes from the EDA prevalence table (capped in config) because
    this dataset runs from 4.6:1 to 493:1 negative:positive.
    """

    def __init__(self, pos_weight: torch.Tensor, use_focal: bool = False,
                 gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.use_focal = use_focal
        self.gamma = gamma
        self.eps = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.eps > 0:
            target = target * (1 - self.eps) + 0.5 * self.eps
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        if not self.use_focal:
            return bce.mean()
        # focal down-weights the easy negatives that dominate this dataset
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        return (bce * (1 - p_t).pow(self.gamma)).mean()
