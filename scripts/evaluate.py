import argparse
import csv

import yaml
from ultralytics import YOLO

from build_country_lists import build_lists
from utils import CLASS_NAMES, COUNTRIES, OUTPUTS_ROOT, SPLITS_ROOT, get_device


def make_test_yaml(country: str) -> str:
    test_list = SPLITS_ROOT / f"{country}_test.txt"
    if not test_list.exists():
        build_lists()

    yaml_path = SPLITS_ROOT / f"{country}_test.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "val": str(test_list),
                "names": {i: name for i, name in enumerate(CLASS_NAMES)},
            }
        )
    )
    return str(yaml_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-country generalization evaluation on the RDD_SPLIT test set")
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
            metrics = model.val(data=make_test_yaml(country), device=device)
            writer.writerow([country, metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
