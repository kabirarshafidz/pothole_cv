import argparse
import random
import shutil
from pathlib import Path

from utils import DATA_ROOT

SEED = 42
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def split_country(country: str) -> None:
    country_dir = DATA_ROOT / country
    image_dir = country_dir / "images"
    label_dir = country_dir / "labels"

    images = sorted(image_dir.glob("*.jpg"))
    random.Random(SEED).shuffle(images)

    n_train = int(len(images) * SPLIT_RATIOS[0])
    n_val = int(len(images) * SPLIT_RATIOS[1])
    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split, files in splits.items():
        (image_dir / split).mkdir(parents=True, exist_ok=True)
        (label_dir / split).mkdir(parents=True, exist_ok=True)
        for img_path in files:
            shutil.move(str(img_path), str(image_dir / split / img_path.name))
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.move(str(label_path), str(label_dir / split / label_path.name))

    print(f"{country}: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")


def main() -> None:
    parser = argparse.ArgumentParser(description="70/15/15 train/val/test split, run once per country")
    parser.add_argument("--country", required=True)
    args = parser.parse_args()
    split_country(args.country)


if __name__ == "__main__":
    main()
