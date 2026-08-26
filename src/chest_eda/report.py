"""STAGE 6 - Turn the measurements into a written report + preprocessing spec.

This stage does no new analysis. It reads the dicts the earlier stages
returned and renders outputs/report/EDA_REPORT.md, where each
recommendation is traceable to a number the pipeline actually measured.
"""
from __future__ import annotations

from datetime import datetime

from . import config
from .io_utils import get_logger, save_json

log = get_logger(__name__)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "**no**"
    if isinstance(v, float):
        return f"{v:,.4f}".rstrip("0").rstrip(".")
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _kv_table(d: dict, keys=None) -> str:
    keys = keys or list(d)
    rows = ["| metric | value |", "| --- | --- |"]
    for k in keys:
        if k in d and not isinstance(d[k], (dict, list)):
            rows.append(f"| `{k}` | {_fmt(d[k])} |")
    return "\n".join(rows)


def build_report(results: dict) -> str:
    meta = results.get("metadata", {})
    lab = results.get("labels", {})
    pat = results.get("patients", {})
    img = results.get("images", {})
    spl = results.get("splits", {})

    lines: list[str] = []
    A = lines.append

    A("# ChestX-ray14 - Exploratory Data Analysis")
    A("")
    A(f"- generated: {datetime.now().isoformat(timespec='seconds')}")
    A(f"- data root: `{config.DATA_ROOT}`")
    A(f"- images in metadata: {_fmt(meta.get('n_rows', 'n/a'))}")
    A(f"- images sampled for pixel analysis: {_fmt(img.get('n_sampled', 0))}")
    A("")
    A("## 1. Data integrity")
    A("")
    A(_kv_table(meta, [
        "n_rows", "n_rows_expected", "row_count_matches_publication",
        "duplicate_image_ids", "n_patients", "implausible_ages",
        "no_finding_inconsistencies", "n_png_on_disk",
        "metadata_rows_missing_file", "files_not_in_metadata",
    ]))
    if meta.get("implausible_age_values"):
        A("")
        A(f"Implausible age values present in the CSV: "
          f"`{meta['implausible_age_values']}` - a documented dataset bug. "
          "These rows are kept (the image is fine) but `age_clean` is NaN, "
          "so never stratify or filter on raw `age`.")
    if meta.get("unknown_label_tokens"):
        A("")
        A(f"Unexpected label tokens: `{meta['unknown_label_tokens']}`")

    A("")
    A("## 2. Labels (multi-label, heavily imbalanced)")
    A("")
    A(_kv_table(lab, [
        "n_images", "n_normal_images", "normal_share",
        "mean_findings_per_image", "max_findings_per_image",
        "most_common_label", "rarest_label", "rarest_label_positives",
        "worst_imbalance_1_to",
    ]))
    if lab.get("top_cooccurring_pairs"):
        A("")
        A("Most frequent label pairs:")
        A("")
        A("| pair | images |")
        A("| --- | --- |")
        for p in lab["top_cooccurring_pairs"][:8]:
            A(f"| {p['pair']} | {_fmt(p['n_images'])} |")

    A("")
    A("## 3. Patients and confounders")
    A("")
    A(_kv_table(pat, [
        "n_patients", "images_per_patient_mean", "images_per_patient_median",
        "images_per_patient_max", "share_patients_with_multiple_images",
        "share_images_from_multi_image_patients", "followup_max",
    ]))
    leak = pat.get("split_leakage", {})
    if leak:
        A("")
        A("**Leakage check** (why you must group by patient):")
        A("")
        A(_kv_table(leak))
        A("")
        A(f"> {leak.get('verdict', '')}")

    A("")
    A("## 4. Pixels")
    A("")
    if img.get("skipped"):
        A(f"_skipped: {img['skipped']}_")
    else:
        A(_kv_table(img, [
            "n_sampled", "n_unreadable", "off_spec_shape_share",
            "multi_channel_share", "intensity_mean_0_255", "intensity_std_0_255",
            "normalize_mean_0_1", "normalize_std_0_1", "n_near_blank",
            "n_low_contrast", "n_suspected_inverted", "mean_black_border_frac",
            "p95_black_border_frac", "n_duplicate_hash_groups",
            "est_full_dataset_gb",
        ]))
        if img.get("channel_counts"):
            A("")
            A(f"Channel counts in sample: `{img['channel_counts']}`, "
              f"PIL modes: `{img['pil_modes']}`")

    A("")
    A("## 5. Splits and bounding boxes")
    A("")
    off = spl.get("official", {})
    if off.get("skipped"):
        A(f"_official split lists: {off['skipped']}_")
    else:
        A(_kv_table(off))
    prop = spl.get("proposed_val_split")
    if prop:
        A("")
        A("Proposed patient-grouped train/val carve-out of `train_val_list.txt`:")
        A("")
        A(_kv_table(prop))
    bb = spl.get("bbox", {})
    if bb and not bb.get("skipped"):
        A("")
        A("Bounding boxes:")
        A("")
        A(_kv_table(bb, [
            "n_boxes", "n_images_with_boxes", "median_area_frac",
            "p10_area_frac", "median_min_side_at_224px", "p10_min_side_at_224px",
        ]))

    A("")
    A("## 6. Preprocessing decisions this analysis implies")
    A("")
    for i, rec in enumerate(_recommendations(results), 1):
        A(f"{i}. {rec}")

    A("")
    A("## 7. Artifacts")
    A("")
    A(f"- tables: `{config.TBL_DIR.relative_to(config.PROJECT_ROOT)}`")
    A(f"- figures: `{config.FIG_DIR.relative_to(config.PROJECT_ROOT)}`")
    A(f"- machine-readable summary: "
      f"`{(config.REPORT_DIR / 'eda_summary.json').relative_to(config.PROJECT_ROOT)}`")

    text = "\n".join(lines) + "\n"
    out = config.REPORT_DIR / "EDA_REPORT.md"
    out.write_text(text, encoding="utf-8")
    save_json(results, "eda_summary")
    log.info("report -> %s", out)
    return text


def _recommendations(r: dict) -> list[str]:
    img = r.get("images", {})
    lab = r.get("labels", {})
    pat = r.get("patients", {})
    recs: list[str] = []

    recs.append(
        "**Split by `patient_id`, not by image.** "
        f"{_fmt(pat.get('share_images_from_multi_image_patients', 0))} of images come "
        "from patients with more than one study; a random image-level split leaked "
        f"{_fmt(pat.get('split_leakage', {}).get('naive_image_split_leaked_patients', 0))} "
        "patients across the boundary in the measured test above. Use the official "
        "`train_val_list.txt` / `test_list.txt`, then `GroupShuffleSplit` on "
        "`patient_id` for validation."
    )
    recs.append(
        "**Multi-label head, not softmax.** 14 sigmoid outputs + "
        "`BCEWithLogitsLoss`. `No Finding` is not a class - it is all 14 zeros. "
        f"Mean findings per image is {_fmt(lab.get('mean_findings_per_image', 0))}."
    )
    if lab.get("pos_weight"):
        worst = max(lab["pos_weight"], key=lab["pos_weight"].get)
        recs.append(
            f"**Handle imbalance explicitly.** Worst label is `{worst}` at "
            f"1:{_fmt(lab['pos_weight'][worst])} negative:positive. Pass the measured "
            "`pos_weight` vector (in `02_label_prevalence.csv`) to "
            "`BCEWithLogitsLoss`, and report per-label AUROC/AP - never plain "
            "accuracy: a model that predicts 'negative' for every label already "
            f"scores {1 - lab.get('mean_findings_per_image', 0) / 14:.1%} element-wise "
            "accuracy while detecting nothing."
        )
    if not img.get("skipped"):
        recs.append(
            "**Force single-channel decode, then replicate to 3 channels.** "
            f"{img.get('multi_channel_share', 0):.2%} of the sampled PNGs are not "
            "single-channel (`.convert('L')` fixes it); replicate to 3 channels only "
            "if you use ImageNet-pretrained weights."
        )
        recs.append(
            f"**Normalize with dataset statistics, not ImageNet's**: "
            f"mean={img.get('normalize_mean_0_1')}, std={img.get('normalize_std_0_1')} "
            "in [0,1] (single channel). Recompute on the training split only, once you "
            "have fixed the split, to avoid leaking test statistics."
        )
        recs.append(
            f"**Resize deliberately.** Native is 1024x1024 (~{_fmt(img.get('est_full_dataset_gb', 0))} GB). "
            "224x224 trains fast but see the bbox table: the 10th-percentile lesion "
            "short side shrinks to "
            f"{r.get('splits', {}).get('bbox', {}).get('p10_min_side_at_224px', 'n/a')} px. "
            "Prefer 320x320 or 448x448 if nodule/mass performance matters. "
            f"Black margin is {img.get('mean_black_border_frac', 0):.1%} on average, so "
            "resize-then-small-centre-crop is safer than an aggressive crop."
        )
        recs.append(
            "**Pre-resize once to an on-disk cache** (e.g. 320px JPEG/webp or a "
            "single `.npy`/LMDB shard). Decoding 1024x1024 PNGs every epoch makes the "
            "CPU, not the GPU, your bottleneck."
        )
        if img.get("n_unreadable") or img.get("n_near_blank") or img.get("n_suspected_inverted"):
            recs.append(
                f"**Quarantine bad images before training**: "
                f"{_fmt(img.get('n_unreadable', 0))} unreadable, "
                f"{_fmt(img.get('n_near_blank', 0))} near-blank, "
                f"{_fmt(img.get('n_low_contrast', 0))} very low contrast, "
                f"{_fmt(img.get('n_suspected_inverted', 0))} possibly inverted in the "
                "sample. Extrapolate, build an exclusion list, and keep it in version "
                "control."
            )
    recs.append(
        "**Augment anatomically.** Small rotations (<=10 deg), small "
        "translation/scale, mild brightness/contrast (or CLAHE). No horizontal flips - "
        "that mirrors the heart and destroys the cardiomegaly and left/right effusion "
        "signal. No vertical flips."
    )
    vc = pat.get("view_confounding", {})
    if vc:
        recs.append(
            "**Watch the view-position confounder.** Normal rate by view is "
            f"`{vc.get('normal_rate_by_view')}` - AP films are portable exams of sicker "
            "patients, so a CNN can score well by detecting the projection instead of "
            "the pathology. Report metrics per view, and consider view as a stratum."
        )
    recs.append(
        "**Remember the label provenance.** These 14 labels were NLP-mined from "
        "radiology reports with ~90% accuracy, not verified by readers. Treat the "
        "ceiling on achievable AUROC accordingly, and do not chase the last 2%."
    )
    return recs
