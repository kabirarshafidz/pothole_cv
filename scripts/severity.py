import argparse

import numpy as np
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO

from utils import get_device


def area_severity(box, image_shape) -> float:
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)
    image_area = image_shape[0] * image_shape[1]
    return box_area / image_area


def depth_severity(box, depth_map) -> float:
    x1, y1, x2, y2 = (int(v) for v in box)
    region = depth_map[y1:y2, x1:x2]
    return float(region.mean()) if region.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Severity scoring for detected road damage")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--mode", choices=["area", "depth"], default="area")
    parser.add_argument("--weights", default="runs/japan_baseline/weights/best.pt")
    args = parser.parse_args()

    device = get_device()
    model = YOLO(args.weights)
    image = Image.open(args.image).convert("RGB")
    results = model.predict(image, device=device)[0]

    depth_map = None
    if args.mode == "depth":
        depth_pipe = pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)
        depth_map = np.array(depth_pipe(image)["depth"])

    for box in results.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        score = (
            area_severity(xyxy, image.size[::-1])
            if args.mode == "area"
            else depth_severity(xyxy, depth_map)
        )
        print(f"class={cls_id} conf={float(box.conf[0]):.2f} severity[{args.mode}]={score:.4f}")


if __name__ == "__main__":
    main()
