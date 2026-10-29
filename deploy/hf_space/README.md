---
title: LungLens
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Chest X-ray classifier for 14 findings, with Grad-CAM
---

# LungLens

Research demo of a DenseNet-121 trained on the NIH ChestX-ray14 dataset.
Upload a frontal chest X-ray to score 14 thoracic findings and view a
Grad-CAM heatmap for each.

**Not a medical device.** Scores are uncalibrated and the heatmaps show where
the model looked, not confirmed pathology. Do not use for clinical decisions.
Uploaded images are processed in memory and never stored.

- `GET /api/model`: model metadata, thresholds and metrics
- `POST /api/predict`: multipart `file` → scores and Grad-CAM maps
- API docs at `/api/docs`
