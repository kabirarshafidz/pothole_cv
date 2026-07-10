import re
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ON_KAGGLE = Path("/kaggle/input").exists()

COUNTRIES = ["Japan", "India", "Czech", "Norway", "United_States", "China"]
CLASS_NAMES = ["D00", "D10", "D20", "D40"]  # longitudinal, transverse, alligator, pothole

# RDD2022 filenames carry a country prefix, e.g. "Japan_000123.jpg".
_COUNTRY_PREFIX = re.compile(r"^(" + "|".join(COUNTRIES) + r")_")


def _find_data_root() -> Path:
    if not ON_KAGGLE:
        return REPO_ROOT / "data" / "RDD_SPLIT"
    matches = sorted(Path("/kaggle/input").glob("*/RDD_SPLIT"))
    return matches[0] if matches else Path("/kaggle/input/rdd-2022/RDD_SPLIT")


# Pre-converted YOLO labels, pre-split 70/15/15 with all 6 countries mixed together —
# do not run a separate convert/split step. Filter by country prefix instead (see country_of).
DATA_ROOT = _find_data_root()
RUNS_ROOT = Path("/kaggle/working/runs") if ON_KAGGLE else REPO_ROOT / "runs"
OUTPUTS_ROOT = Path("/kaggle/working/outputs") if ON_KAGGLE else REPO_ROOT / "outputs"
SPLITS_ROOT = OUTPUTS_ROOT / "splits"


def country_of(filename: str) -> str | None:
    match = _COUNTRY_PREFIX.match(filename)
    return match.group(1) if match else None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
