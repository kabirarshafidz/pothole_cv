import re
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ON_KAGGLE = Path("/kaggle/input").exists()

COUNTRIES = ["Japan", "India", "Czech", "Norway", "United_States", "China"]
# The Kaggle "aliabdelmenam/rdd-2022" copy of RDD_SPLIT has 5 classes, not the 4 in the
# RDD2022 paper — verified against its labels, which use class id 4 for Pothole. Declaring
# only 4 names here (nc=4) made Ultralytics silently drop every class-4 label as "corrupt",
# i.e. every pothole box in the dataset was being discarded during training.
CLASS_NAMES = ["D00", "D10", "D20", "D44", "D40"]  # longitudinal, transverse, alligator, other corruption, pothole

# RDD2022 filenames carry a country prefix, e.g. "Japan_000123.jpg".
_COUNTRY_PREFIX = re.compile(r"^(" + "|".join(COUNTRIES) + r")_")


def _find_data_root() -> Path:
    if not ON_KAGGLE:
        return REPO_ROOT / "data" / "RDD_SPLIT"
    matches = sorted(p for p in Path("/kaggle/input").rglob("RDD_SPLIT") if p.is_dir())
    if not matches:
        raise FileNotFoundError(
            "No RDD_SPLIT folder found under /kaggle/input — attach the RDD2022 dataset. "
            f"Contents of /kaggle/input: {list(Path('/kaggle/input').iterdir())}"
        )
    return matches[0]


# Pre-converted YOLO labels, pre-split 70/15/15 with all 6 countries mixed together —
# do not run a separate convert/split step. Filter by country prefix instead (see country_of).
DATA_ROOT = _find_data_root()
RUNS_ROOT = Path("/kaggle/working/runs") if ON_KAGGLE else REPO_ROOT / "runs"
OUTPUTS_ROOT = Path("/kaggle/working/outputs") if ON_KAGGLE else REPO_ROOT / "outputs"
SPLITS_ROOT = OUTPUTS_ROOT / "splits"


def country_of(filename: str) -> str | None:
    match = _COUNTRY_PREFIX.match(filename)
    return match.group(1) if match else None


def truncate_list(list_path: Path, n: int, suffix: str) -> Path:
    lines = list_path.read_text().splitlines()[:n]
    out_path = list_path.with_name(f"{list_path.stem}_{suffix}.txt")
    out_path.write_text("\n".join(lines))
    return out_path


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def verify_label_classes(data_root: Path, class_names: list[str], split: str = "train", sample: int = 200) -> None:
    """Fail fast if label class ids don't match class_names — catches the exact
    silent-drop bug from ed65d4a (nc=4 declared against 5-class labels) before
    a training run burns GPU time on it."""
    label_dir = data_root / split / "labels"
    label_files = sorted(label_dir.glob("*.txt"))[:sample]
    if not label_files:
        raise FileNotFoundError(f"No label files found in {label_dir} — check DATA_ROOT ({data_root})")

    max_id = len(class_names) - 1
    seen_ids: set[int] = set()
    for label_file in label_files:
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            class_id = int(line.split()[0])
            seen_ids.add(class_id)
            if class_id > max_id:
                raise ValueError(
                    f"{label_file} has class id {class_id}, but CLASS_NAMES only declares "
                    f"{len(class_names)} classes (0-{max_id}): {class_names}. Labels would be "
                    f"silently dropped as 'corrupt' by Ultralytics — see ed65d4a."
                )

    pothole_id = class_names.index("D40") if "D40" in class_names else None
    if pothole_id is not None and pothole_id not in seen_ids:
        raise ValueError(
            f"No labels with class id {pothole_id} (D40/Pothole) found in the first "
            f"{len(label_files)} {split} label files — verify CLASS_NAMES order still "
            f"matches this Kaggle copy's pre-conversion."
        )
