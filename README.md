# ChestX-ray14 — data analysis pipeline (pre-training EDA)

An analysis-only pipeline for the NIH Clinical Center chest X-ray dataset
([kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)).
It answers "what is in this data, and what does that force my CNN
preprocessing to look like?" — it does **not** train anything.

Every number in the generated report comes from a measurement in the code, and
every preprocessing recommendation cites the number it came from.

---

## 1. Where to put the data

The pipeline resolves the dataset directory in this order:

1. `--data-root <path>` on the command line
2. the `CHEST_XRAY_DATA` environment variable
3. the default `data/raw/nih-chest-xrays/`

### Recommended layout

```
lungs/
├── data/
│   ├── raw/
│   │   └── nih-chest-xrays/          <-- put the Kaggle download HERE, unzipped
│   │       ├── Data_Entry_2017.csv       (112,120 rows of metadata)
│   │       ├── BBox_List_2017.csv        (984 hand-drawn boxes)
│   │       ├── train_val_list.txt        (86,524 image ids)
│   │       ├── test_list.txt             (25,596 image ids)
│   │       ├── images_001/images/*.png
│   │       ├── images_002/images/*.png
│   │       └── ...  images_012/images/*.png
│   └── interim/                      <-- cache written by the pipeline
├── outputs/{figures,tables,report}   <-- all results land here
├── src/chest_eda/                    <-- the pipeline
├── tools/                            <-- download + synthetic-fixture helpers
└── run_eda.py                        <-- entry point
```

Keep the twelve `images_XXX/` folders exactly as Kaggle ships them — the code
globs `images_*/images/*.png` and doesn't care how they're split, but it does
expect them under one root. Do **not** rename `Data_Entry_2017.csv`
(the `Data_Entry_2017_v2020.csv` variant is also recognised).

### If the 45 GB doesn't fit on C:

Don't copy it into the project. Put it on another drive and point at it:

```powershell
# PowerShell — set for this shell
$env:CHEST_XRAY_DATA = "D:\datasets\nih-chest-xrays"
python run_eda.py

# or persist it for future shells
setx CHEST_XRAY_DATA "D:\datasets\nih-chest-xrays"

# or one-shot
python run_eda.py --data-root "D:\datasets\nih-chest-xrays"
```

A directory symlink also works (needs an elevated shell on Windows):

```powershell
New-Item -ItemType SymbolicLink -Path "data\raw\nih-chest-xrays" -Target "D:\datasets\nih-chest-xrays"
```

### Disk budget

| item | size |
| --- | --- |
| download (zip, transient) | ~42 GB |
| extracted PNGs (112,120 × 1024×1024) | ~45 GB |
| CSVs + list files | < 10 MB |
| pipeline outputs | < 50 MB |
| **a 320px pre-resized training cache (recommended later)** | ~4 GB |

Plan for **~90 GB free** during download+extract, ~45 GB steady state. Put it
on an SSD if you can — epoch time on this dataset is usually I/O-bound.

Never commit `data/` to git. A `.gitignore` is included.

### Getting the download

```powershell
pip install -r requirements.txt

# Option A — kagglehub (needs kaggle.json or an interactive login)
python tools/download_data.py --method kagglehub

# Option B — kaggle CLI, straight into the project layout
#   1. kaggle.com -> Settings -> API -> Create New Token
#   2. move kaggle.json to %USERPROFILE%\.kaggle\kaggle.json
python tools/download_data.py --method cli --dest data/raw/nih-chest-xrays
```

The Kaggle mirror is one huge dataset; the original NIH Box mirror ships the
same PNGs as 12 `images_XXX.tar.gz` archives if you prefer resumable parts.

---

## 2. Try it without the download

A synthetic fixture reproduces the dataset's schema *and* its quirks
(multi-image patients, ages > 100, RGBA files, a truncated PNG, blank and
inverted images) in a few hundred KB:

```powershell
python tools/make_fixture.py --out data/raw/fixture --n 400
python run_eda.py --data-root data/raw/fixture --sample 400
```

Read `outputs/report/EDA_REPORT.md`. Then delete `outputs/` and re-run on the
real data. **The fixture's numbers are fake — never quote them.**

---

## 3. Running it

```powershell
python run_eda.py                      # all six stages
python run_eda.py --sample 5000        # inspect more pixels (slower)
python run_eda.py --no-images          # CSV analysis only; no PNGs required
python run_eda.py --stages 1,2,3       # a subset
```

Runtime on the real dataset: stages 1–3 and 5 are ~30 s (pure pandas). Stage 4
is ~1 ms/image after warm cache, so the 2,000-image default is ~1–3 min; the
first run also walks 112k filenames (~1 min on a cold HDD).

Outputs:

- `outputs/report/EDA_REPORT.md` — the human-readable report
- `outputs/report/eda_summary.json` — every metric, machine-readable
- `outputs/tables/*.csv` — one CSV per analysis (diffable, auditable)
- `outputs/figures/*.png` — prevalence, co-occurrence, demographics, pixels

---

## 4. What each stage does, and why it exists

### Stage 1 — `s01_metadata.py`: load and validate

Reads `Data_Entry_2017.csv`, renames its awkward columns
(`OriginalImage[Width` → `width`) to snake_case, drops the phantom column
pandas creates from the CSV's trailing comma, then **validates**:

- row count against the published 112,120 and patient count against 30,805 —
  if these don't match, your download is incomplete and nothing downstream is
  trustworthy;
- duplicate `Image Index`, null columns;
- **implausible ages** — the CSV genuinely contains ages like 148, 411 and 413.
  The rows are kept (the *image* is fine) but a parallel `age_clean` column is
  NaN there, so you never stratify or plot on the corrupt value;
- **unknown label tokens** — guards against a re-released CSV adding a class;
- **cross-reference with disk** — how many metadata rows have no PNG, and how
  many PNGs aren't in the metadata. This is how you catch a half-finished
  download before you waste a GPU-day.

It also does the single most important transform: **one-hot encoding the
pipe-delimited `Finding Labels` string into 14 binary columns**. That matrix
*is* your CNN target. `No Finding` deliberately becomes "all 14 zeros" rather
than a 15th class, and the code asserts that the text and the encoding agree.

### Stage 2 — `s02_labels.py`: label structure

- **Prevalence and imbalance per label.** Infiltration is ~18% of images,
  Hernia ~0.2% — a ~100× spread. Reported alongside the exact
  `pos_weight = n_negative / n_positive` you paste into
  `BCEWithLogitsLoss(pos_weight=...)`.
- **Cardinality** — how many findings co-occur per image. ~54% of images are
  `No Finding`; the rest average well over one label. This is *proof* the
  problem is multi-label, which decides sigmoid-vs-softmax before you touch a
  layer.
- **Co-occurrence matrix**, raw and as conditional P(B|A). Pairs like
  Effusion+Atelectasis dominate; those are exactly the labels your model will
  confuse, and the pairs to look at when per-label AUROC disappoints.

### Stage 3 — `s03_patients.py`: patients and confounders

The stage that saves the project. 112,120 images come from only ~30,805
patients (~3.6 each, one patient has 180+ follow-ups).

- **Images per patient** quantifies the leakage exposure.
- **The leakage demonstration** actually runs both splits and counts the
  damage: a naive random image-level 80/20 split leaves thousands of patients
  with images on both sides — the *same chest*, days apart, in train and test.
  Your AUROC then partly measures "have I memorised this ribcage". The same
  code shows `GroupShuffleSplit(groups=patient_id)` reduces the overlap to
  zero.
- **Demographics** — age (post-cleaning), sex, and view position.
- **Confounder scan** — prevalence broken down by sex and by view. AP films are
  portable exams of sicker, often supine patients, so "AP" correlates with
  disease. A CNN can score respectably by learning to detect the *projection*
  instead of the pathology. Knowing the size of that gap up front is what lets
  you report metrics per view later and catch the shortcut.

### Stage 4 — `s04_images.py`: pixels (on a sample)

Opening 112k PNGs is 45 GB of I/O, so this runs on a seeded random sample
(`--sample`). Each measurement maps to one decision:

| measurement | decision it drives |
| --- | --- |
| decode succeeds? | the exclusion list, so a truncated PNG doesn't kill epoch 3 |
| shape, channel count, PIL mode | ChestX-ray14 is *nominally* 8-bit greyscale 1024², but a known subset is stored **4-channel RGBA**. Force `.convert("L")` or your tensors silently change shape |
| mean / std intensity | the actual `Normalize(mean, std)` constants — the dataset is much darker than ImageNet, so ImageNet's constants are wrong here |
| p1/p99 spread, std ≈ 0 | blank and near-zero-contrast scans to drop |
| centre-vs-corner brightness | suspected photometric inversion (a negative image) |
| fraction of near-black pixels | how much dead margin exists, i.e. how aggressive a centre crop can safely be |
| 8×8 average hash | exact/near duplicates within the sample |

Plus an intensity histogram and a 12-image contact sheet — always look at the
pictures.

### Stage 5 — `s05_splits.py`: official splits and boxes

- Verifies `train_val_list.txt` + `test_list.txt` cover the metadata exactly
  once and are **patient-disjoint**. Using these lists is what makes your
  numbers comparable to published ChestX-ray14 results.
- **Prevalence drift** train_val vs test, per label. Big drift on a rare label
  means that label's test AUROC will be noisy — worth knowing before you read
  meaning into a 0.02 difference.
- Carves a **patient-grouped validation set** out of `train_val` and reports
  the resulting drift, writing the full assignment to
  `outputs/tables/05_split_assignment.csv` so training uses a fixed, auditable
  split.
- **`BBox_List_2017.csv`** — 984 boxes over 8 classes. It reports lesion size
  as a fraction of the frame *and* what the short side becomes at 224×224.
  This is the concrete, quantitative argument about input resolution: a small
  nodule that is 30 px across at 1024² is ~7 px at 224², i.e. gone after two
  pooling layers.

### Stage 6 — `report.py`: render

No new analysis. It renders `EDA_REPORT.md` and `eda_summary.json`, turning
the measured dicts into recommendations that quote their own evidence — so
when someone asks "why 320×320?", the answer is a number, not a vibe.

---

## 5. What the analysis concludes (for the training pipeline you'll build next)

These come out of the report with your dataset's actual numbers filled in:

1. **Split by patient, never by image** — use the official lists, then
   `GroupShuffleSplit` on `patient_id` for validation.
2. **14 sigmoids + `BCEWithLogitsLoss`**, not softmax. `No Finding` is the
   all-zero vector, not a class.
3. **Weight the loss** with the measured `pos_weight`; evaluate with per-label
   AUROC/AP, never accuracy.
4. **Force single-channel decode**, then replicate to 3 channels only if you
   use ImageNet-pretrained weights.
5. **Normalize with this dataset's mean/std**, computed on the training split
   only.
6. **Choose resolution against the bbox evidence** — 224² is cheap but costs
   small-lesion recall; 320²–448² is the honest compromise.
7. **Pre-resize once to an on-disk cache** — decoding 1024² PNGs every epoch
   makes the CPU your bottleneck, not the GPU.
8. **Quarantine the bad images** in a version-controlled exclusion list.
9. **Augment anatomically** — small rotations/translations, mild
   brightness/contrast or CLAHE. **No horizontal flips**: mirroring the chest
   destroys the cardiomegaly and left/right effusion signal.
10. **Report per-view metrics** to catch the AP/PA shortcut.
11. **Remember the labels are NLP-mined from reports at ~90% accuracy** — there
    is a ceiling here, and it isn't 1.0.

---

## 6. Module map

| file | role |
| --- | --- |
| [config.py](src/chest_eda/config.py) | all paths, the 14 label names, knobs — change the dataset location in one place |
| [io_utils.py](src/chest_eda/io_utils.py) | logging, table/figure/json saving, the image-path index + cache |
| [s01_metadata.py](src/chest_eda/s01_metadata.py) | load, validate, one-hot |
| [s02_labels.py](src/chest_eda/s02_labels.py) | prevalence, imbalance, co-occurrence |
| [s03_patients.py](src/chest_eda/s03_patients.py) | demographics, confounders, leakage |
| [s04_images.py](src/chest_eda/s04_images.py) | pixel statistics and quality flags |
| [s05_splits.py](src/chest_eda/s05_splits.py) | official splits, drift, bounding boxes |
| [report.py](src/chest_eda/report.py) | renders the report |
| [run_eda.py](run_eda.py) | CLI entry point |
| [tools/download_data.py](tools/download_data.py) | Kaggle download helper |
| [tools/make_fixture.py](tools/make_fixture.py) | synthetic dataset for smoke tests |

---

## 7. Web demo (upload + Grad-CAM)

A FastAPI app in [webapp/](webapp/) serves a landing page and an analyzer
that scores an uploaded X-ray for all 14 findings and overlays a Grad-CAM
heatmap per finding.

```powershell
pip install -r webapp/requirements.txt
python -m webapp.server                      # http://127.0.0.1:8000
python -m webapp.server --checkpoint runs/densenet121_320/best.pt --port 8080
```

- Preprocessing reuses `chest_train.data.build_transforms` with the
  checkpoint's saved config, so uploads are seen exactly as validation was.
- Thresholds and metrics are read from `val_metrics.json` next to the
  checkpoint. If `test_metrics.json` exists (`python evaluate.py ...`), the
  landing page shows test metrics instead.
- Uploads are processed in memory and never written to disk. The server binds
  to `127.0.0.1` by default; pass `--host 0.0.0.0` to expose it on a network.
- API: `GET /api/model`, `POST /api/predict` (multipart `file`); docs at
  `/api/docs`.

### Deploying to Hugging Face Spaces

[tools/build_hf_space.py](tools/build_hf_space.py) assembles everything the
Space needs into `dist/hf_space/`: the Dockerfile and Space README from
[deploy/hf_space/](deploy/hf_space/), `webapp/`, `src/chest_train/`, and a
slimmed model (weights only, 28 MB instead of 84 MB) with its
`val_metrics.json`.

**First deploy**

1. On huggingface.co: **New Space** → SDK **Docker** → **Blank** template →
   hardware **CPU basic (free)**. Then create an access token with **write**
   permission (Settings → Access Tokens).
2. Clone the empty Space into `dist/hf_space` and build into it:
   ```powershell
   git lfs install
   Remove-Item -Recurse -Force dist\hf_space -ErrorAction SilentlyContinue
   git clone https://huggingface.co/spaces/<username>/<space-name> dist\hf_space
   python tools\build_hf_space.py            # or --checkpoint runs\<run>\best.pt
   ```
3. Push (username = your HF username, password = the token):
   ```powershell
   cd dist\hf_space
   git add .
   git commit -m "Deploy LungLens"
   git push
   ```
4. The Space builds the image (about 5–10 minutes the first time) and then
   serves at `https://huggingface.co/spaces/<username>/<space-name>`.

**Updating** (new model or app changes): run `python tools\build_hf_space.py`
again. It keeps `dist\hf_space\.git`, so just commit and push from there.

Free Spaces sleep after about 48 hours without traffic; the next visit wakes
them in about a minute. To show test-set metrics on the landing page, run
`evaluate.py` on the checkpoint first; the build copies `test_metrics.json`
when it exists.

---

## 8. Licence / use note

ChestX-ray14 is released by the NIH Clinical Center for research use; the
images are de-identified but the terms on the Kaggle page still apply. The
labels are NLP-mined and not radiologist-verified — this data supports
research, not clinical decisions.
