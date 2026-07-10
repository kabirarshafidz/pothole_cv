from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
ON_KAGGLE = Path("/kaggle/input").exists()

DATA_ROOT = Path("/kaggle/input/rdd2022") if ON_KAGGLE else REPO_ROOT / "data"
RUNS_ROOT = Path("/kaggle/working/runs") if ON_KAGGLE else REPO_ROOT / "runs"
OUTPUTS_ROOT = Path("/kaggle/working/outputs") if ON_KAGGLE else REPO_ROOT / "outputs"

COUNTRIES = ["Japan", "India", "Czech", "Norway", "United_States", "China"]
CLASS_NAMES = ["D00", "D10", "D20", "D40"]  # longitudinal, transverse, alligator, pothole


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
