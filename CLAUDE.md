# CLAUDE.md — Pothole & Road Surface Damage Severity Mapping

## Project overview

Automated road damage detection and severity estimation from dashcam/phone images.
Research contribution: cross-country generalization study (6 countries) + severity scoring
comparison (area-based vs depth-assisted).

## Stack

- Python 3.11
- PyTorch (MPS backend — Apple M1)
- Ultralytics YOLOv8
- Depth Anything (HuggingFace transformers)
- OpenCV, albumentations, pandas, matplotlib

## Environment

```bash
conda activate pothole
export PYTORCH_ENABLE_MPS_FALLBACK=1  # set in ~/.zshrc, not per-script
```

Always use `device="mps"` in Ultralytics calls. Never hardcode `"cuda"` or `"cpu"`.

Real training/evaluation runs happen on Kaggle (GPU + the RDD2022 data), via
`notebooks/kaggle_pipeline.ipynb`. Local MPS is for iterating on script logic only —
there's no data or GPU locally. `scripts/utils.py` has `get_device()` (cuda → mps → cpu)
and `DATA_ROOT`/`RUNS_ROOT`/`OUTPUTS_ROOT`, which resolve to `/kaggle/...` when
`/kaggle/input` exists and to the local repo paths otherwise — use these instead of
hardcoding device or paths in scripts.

## Project structure

```
pothole_cv/
├── CLAUDE.md
├── data/
│   └── RDD_SPLIT/                       # Kaggle "RDD 2022" dataset, as-is
│       ├── train/images  labels         # YOLO format (.txt), all 6 countries mixed
│       ├── val/images    labels
│       └── test/images   labels
├── scripts/
│   ├── utils.py                         # device + path resolution, country_of()
│   ├── build_country_lists.py           # filters RDD_SPLIT by country prefix into
│   │                                     #   outputs/splits/<country>_<split>.txt
│   ├── train.py                         # YOLOv8 training entry point
│   ├── evaluate.py                      # multi-country generalization eval
│   ├── severity.py                      # severity scoring (v1 area, v2 depth)
│   └── visualize.py                     # annotated output images + route map
├── notebooks/
│   └── kaggle_pipeline.ipynb            # actual run environment — GPU + real data
├── runs/                                # YOLOv8 auto-saves checkpoints here
│   └── japan_baseline/weights/best.pt  # primary model
└── outputs/
    ├── splits/                          # per-country image list .txt files + yamls
    ├── generalization_results.csv       # 6-country × 4-class mAP table
    └── severity_comparison.csv          # area vs depth severity scores
```

## Dataset

**RDD2022** — 47k road images, 6 countries, 4 damage classes.
On Kaggle as "RDD 2022" → `RDD_SPLIT/{train,val,test}/{images,labels}`. Unlike the raw
Zenodo release, this copy is **already YOLO-formatted and already split 70/15/15**, with
all 6 countries mixed together within each split. Filenames carry a country prefix
(`Japan_000123.jpg`, `India_000045.jpg`, ...) — that prefix is the only way to recover
per-country subsets, via `country_of()` in `scripts/utils.py`.
Zenodo original (XML, per-country folders): https://zenodo.org/record/7504400 — not what
this pipeline consumes, but useful for cross-referencing label quality/class definitions.

Damage classes (YOLO class index order):

- 0 = D00 Longitudinal crack
- 1 = D10 Transverse crack
- 2 = D20 Alligator crack
- 3 = D40 Pothole

This order is an **assumption about the Kaggle dataset's pre-conversion**, not something
we control — verify it (e.g. inspect a few label files, check D40/pothole AP isn't near
zero) before trusting per-class AP or severity results. `notebooks/kaggle_pipeline.ipynb`
has a sanity-check cell for this in the data prep section.

Label quality by country (best → worst): Japan > Czech > Norway > USA > China > India.
Always train on Japan first. India is used for generalization stress-testing only.

dataset yamls must have `names` in exactly this order — any mismatch silently breaks
per-class AP and severity scoring.

## Key scripts

### build_country_lists.py

RDD_SPLIT is pre-converted and pre-split — there's no XML-to-YOLO conversion step and no
train/val/test split to run. This script only filters: for each of train/val/test, it
groups image paths by country prefix and writes `outputs/splits/<country>_<split>.txt`
(one path per line). `train.py` and `evaluate.py` call it automatically if the list files
don't exist yet, so it rarely needs to be run directly.

### train.py

Trains YOLOv8s on one country (Japan by default) by pointing Ultralytics' `train`/`val`
dataset yaml fields at that country's list files from `build_country_lists.py`. Key params:

- `imgsz=640`, `batch=16` (reduce to 8 under memory pressure)
- `epochs=50`, `patience=10` (early stopping)
- `device` — via `get_device()`, never hardcoded

### evaluate.py

For each country, builds a test-only yaml from that country's `outputs/splits/<country>_test.txt`
and runs `model.val()` against it using the Japan-trained model.
Outputs `generalization_results.csv` — this is the core research result table.

### severity.py

Two modes:

- `--mode area` — bounding box area / image area as severity proxy (baseline)
- `--mode depth` — Depth Anything small model estimates depth within each box (advanced)

Usage:

```bash
python scripts/severity.py path/to/image.jpg --mode area
python scripts/severity.py path/to/image.jpg --mode depth
```

## Evaluation metrics

Primary: **mAP@50** (main reported metric)
Secondary: mAP@50-95, per-class AP, precision, recall

Expected Japan baseline (YOLOv8s):

- mAP@50: 0.60 – 0.67
- mAP@50-95: 0.32 – 0.38
- D20 (alligator) will be lowest AP — expected, not a bug

Generalization drop (Japan → India): expect 15–25 mAP points. This IS the finding.

## Ablation studies

Run in this order:

1. Model size: yolov8n vs yolov8s vs yolov8m on Japan
2. Cross-country: Japan-trained model evaluated on all 6 countries
3. Severity method: area-based vs depth-assisted comparison
4. Class analysis: per-class AP across countries (which class transfers worst?)
5. Data mixing: Japan + India training vs Japan-only (does mixing help?)

## Common mistakes to avoid

**Label index mismatch** — if D40 (pothole) AP is near zero, the class index the Kaggle
copy's pre-conversion used doesn't match `CLASS_NAMES` (D00=0, D10=1, D20=2, D40=3).
This is an assumption about someone else's conversion, not something we generate — verify
against actual label files rather than assuming it's correct.

**Country prefix drift** — `country_of()` matches filenames against the exact `COUNTRIES`
list in `utils.py`. If RDD_SPLIT filenames use different casing/spelling (e.g. `USA_` vs
`United_States_`), images silently get dropped from every per-country list with no error.

**Data leakage** — RDD_SPLIT's train/val/test split is already frozen; don't reshuffle or
recombine it. `build_country_lists.py` only filters within each split, it never moves
images across train/val/test.

**MPS fallback** — set `PYTORCH_ENABLE_MPS_FALLBACK=1` before any local training or
inference. Without it, unsupported ops crash instead of falling back to CPU. Not relevant
on Kaggle (CUDA).

**Overfitting Japan** — mAP@50 above 0.80 on Japan test set almost certainly
means train/test overlap. Re-check that `build_country_lists.py` isn't double-counting
an image across splits.

## Research narrative

The project has two claims:

1. **Generalization claim** — a model trained on Japan degrades significantly on
   other countries, and the degradation is class-dependent (D20 transfers worst).

2. **Severity claim** — depth-assisted severity estimation is more informative than
   area-based proxy, particularly for potholes (D40) where depth matters more than
   surface area.

Both claims need a table or figure to support them. generalization_results.csv
supports claim 1. severity_comparison.csv (with qualitative examples) supports claim 2.

## Week-by-week milestones

| Week | Deliverable                                                                     |
| ---- | ------------------------------------------------------------------------------- |
| 1    | Environment set up, Japan country lists built from RDD_SPLIT, first training run complete |
| 2    | Japan baseline mAP reported, confusion matrix analyzed, per-class AP table      |
| 3    | 6-country generalization table complete — this is the core research result      |
| 4    | Severity v1 (area-based) implemented, demo on sample images                     |
| 5    | Severity v2 (Depth Anything) implemented, comparison with v1 documented         |
| 6    | Ablation tables complete, write-up drafted, demo video recorded                 |

## Demo

Final demo should show:

- Input: dashcam image or short video clip
- Output: annotated image with bounding boxes colored by severity
  (green = low, orange = medium, red = high)
- Sidebar: per-detection table (class, confidence, severity score, method used)
- Optional: GPS route map with damage hotspots if metadata available

Run demo (on Kaggle, where the data and trained weights actually are):

```bash
python scripts/severity.py data/RDD_SPLIT/test/images/Japan_000001.jpg --mode depth
```

## External resources

- RDD2022 paper: Arya et al. 2022 (arXiv) — read intro + dataset sections
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com
- Depth Anything model: https://huggingface.co/LiheYoung/depth-anything-small-hf
- RDD2022 dataset: https://zenodo.org/record/7504400
- mAP explained: search "Mean Average Precision mAP Explained Towards Data Science"
