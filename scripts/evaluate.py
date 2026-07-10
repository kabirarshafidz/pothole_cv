import argparse
import csv

from ultralytics import YOLO

from utils import COUNTRIES, DATA_ROOT, OUTPUTS_ROOT, get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-country generalization evaluation")
    parser.add_argument("--weights", default="runs/japan_baseline/weights/best.pt")
    parser.add_argument("--countries", nargs="+", default=COUNTRIES)
    args = parser.parse_args()

    model = YOLO(args.weights)
    device = get_device()
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_ROOT / "generalization_results.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["country", "map50", "map50_95", "precision", "recall"])
        for country in args.countries:
            data_yaml = DATA_ROOT / country / "dataset.yaml"
            metrics = model.val(data=str(data_yaml), device=device)
            writer.writerow([country, metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
