import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from utils import CLASS_NAMES, DATA_ROOT

IMG_SIZE = 600  # RDD2022 default; verify before running on non-Japan subsets


def convert_one(xml_path: Path, out_path: Path) -> None:
    root = ET.parse(xml_path).getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name not in CLASS_NAMES:
            continue
        cls_id = CLASS_NAMES.index(name)
        box = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(box.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
        x_center = (xmin + xmax) / 2 / IMG_SIZE
        y_center = (ymin + ymax) / 2 / IMG_SIZE
        w = (xmax - xmin) / IMG_SIZE
        h = (ymax - ymin) / IMG_SIZE
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RDD2022 Pascal VOC XML annotations to YOLO format")
    parser.add_argument("--country", required=True, help="Country folder under data/, e.g. Japan")
    args = parser.parse_args()

    country_dir = DATA_ROOT / args.country
    xml_dir = country_dir / "annotations" / "xmls"
    label_dir = country_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob("*.xml"))
    for xml_path in xml_files:
        convert_one(xml_path, label_dir / f"{xml_path.stem}.txt")

    print(f"Converted {len(xml_files)} annotations for {args.country} -> {label_dir}")


if __name__ == "__main__":
    main()
