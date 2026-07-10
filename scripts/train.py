import argparse

import yaml
from ultralytics import YOLO

from build_country_lists import build_lists
from utils import CLASS_NAMES, RUNS_ROOT, SPLITS_ROOT, get_device, truncate_list


def make_dataset_yaml(country: str, limit: int | None = None) -> str:
    train_list = SPLITS_ROOT / f"{country}_train.txt"
    val_list = SPLITS_ROOT / f"{country}_val.txt"
    if not train_list.exists() or not val_list.exists():
        build_lists()

    if limit:
        train_list = truncate_list(train_list, limit, suffix="dryrun")
        val_list = truncate_list(val_list, max(1, limit // 4), suffix="dryrun")

    yaml_path = SPLITS_ROOT / f"{country}{'_dryrun' if limit else ''}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "train": str(train_list),
                "val": str(val_list),
                "names": {i: name for i, name in enumerate(CLASS_NAMES)},
            }
        )
    )
    return str(yaml_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on one country's subset of RDD_SPLIT")
    parser.add_argument("--country", default="Japan")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--name", default="japan_baseline")
    args = parser.parse_args()

    data_yaml = make_dataset_yaml(args.country)
    model = YOLO(args.model)
    model.train(
        data=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        device=get_device(),
        project=str(RUNS_ROOT),
        name=args.name,
    )


if __name__ == "__main__":
    main()
