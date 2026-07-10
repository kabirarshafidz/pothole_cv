import argparse

from ultralytics import YOLO

from utils import RUNS_ROOT, get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on a dataset yaml")
    parser.add_argument("--data", default="data/dataset_japan.yaml")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--name", default="japan_baseline")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
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
