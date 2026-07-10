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
pothole/
├── CLAUDE.md
├── data/
│   ├── Japan/
│   │   ├── images/train  val  test
│   │   ├── labels/train  val  test       # YOLO format (.txt)
│   │   └── annotations/xmls              # original RDD2022 Pascal VOC XML
│   ├── India/
│   ├── Czech/
│   ├── Norway/
│   ├── United_States/
│   ├── China/
│   └── dataset_japan.yaml               # active training config
├── scripts/
│   ├── convert_annotations.py           # XML → YOLO label conversion
│   ├── split_dataset.py                 # train/val/test split (70/15/15)
│   ├── train.py                         # YOLOv8 training entry point
│   ├── evaluate.py                      # multi-country generalization eval
│   ├── severity.py                      # severity scoring (v1 area, v2 depth)
│   └── visualize.py                     # annotated output images + route map
├── runs/                                # YOLOv8 auto-saves checkpoints here
│   └── japan_baseline/weights/best.pt  # primary model
└── outputs/
    ├── generalization_results.csv       # 6-country × 4-class mAP table
    └── severity_comparison.csv          # area vs depth severity scores
```

## Dataset

**RDD2022** — 47k road images, 6 countries, 4 damage classes.
Download: https://zenodo.org/record/7504400

Damage classes (YOLO class index order):

- 0 = D00 Longitudinal crack
- 1 = D10 Transverse crack
- 2 = D20 Alligator crack
- 3 = D40 Pothole

Label quality by country (best → worst): Japan > Czech > Norway > USA > China > India.
Always train on Japan first. India is used for generalization stress-testing only.

dataset.yaml must have `names` in exactly this order — any mismatch silently breaks
per-class AP and severity scoring.

## Key scripts

### convert_annotations.py

Converts Pascal VOC XML (RDD2022 default) to YOLO .txt format.
Input: `data/<Country>/annotations/xmls/*.xml`
Output: `data/<Country>/labels/*.txt`
Assumes image size 600×600 — verify before running on non-Japan subsets.

### split_dataset.py

Random 70/15/15 split. Seed fixed at 42 for reproducibility.
Run once per country. Do not re-run after training has started.

### train.py

Trains YOLOv8s on Japan by default. Key params:

- `imgsz=640`, `batch=16` (reduce to 8 if MPS memory pressure)
- `epochs=50`, `patience=10` (early stopping)
- `device="mps"`

### evaluate.py

Runs `model.val()` on each country's test split using the Japan-trained model.
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

**Label index mismatch** — if D40 (pothole) AP is near zero, the class index in
dataset.yaml is wrong. Must be: D00=0, D10=1, D20=2, D40=3.

**Coordinate normalization** — YOLO format requires coordinates in [0,1].
RDD2022 XML uses absolute pixel coordinates. Divide by image width/height in
convert_annotations.py. Easy to silently get wrong.

**Data leakage** — run split_dataset.py once, then freeze the split. Never re-split
after seeing validation results.

**MPS fallback** — set `PYTORCH_ENABLE_MPS_FALLBACK=1` before any training or
inference. Without it, unsupported ops crash instead of falling back to CPU.

**Overfitting Japan** — mAP@50 above 0.80 on Japan test set almost certainly
means train/test overlap. Re-check split.

**Re-running convert_annotations.py** — safe to re-run but will overwrite labels.
If you have manually corrected any labels, back them up first.

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
| 1    | Environment set up, Japan data converted and split, first training run complete |
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

Run demo:

```bash
python scripts/severity.py data/Japan/images/test/sample.jpg --mode depth
```

## External resources

- RDD2022 paper: Arya et al. 2022 (arXiv) — read intro + dataset sections
- Ultralytics YOLOv8 docs: https://docs.ultralytics.com
- Depth Anything model: https://huggingface.co/LiheYoung/depth-anything-small-hf
- RDD2022 dataset: https://zenodo.org/record/7504400
- mAP explained: search "Mean Average Precision mAP Explained Towards Data Science"
