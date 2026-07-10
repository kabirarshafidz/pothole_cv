import argparse

import cv2
from ultralytics import YOLO

from severity import area_severity
from utils import get_device

SEVERITY_COLORS = {
    "low": (0, 200, 0),
    "medium": (0, 165, 255),
    "high": (0, 0, 255),
}


def bucket(score: float) -> str:
    if score < 0.02:
        return "low"
    if score < 0.06:
        return "medium"
    return "high"


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw severity-colored detections on an image")
    parser.add_argument("image")
    parser.add_argument("--weights", default="runs/japan_baseline/weights/best.pt")
    parser.add_argument("--out", default="outputs/annotated.jpg")
    args = parser.parse_args()

    device = get_device()
    model = YOLO(args.weights)
    image = cv2.imread(args.image)
    results = model.predict(image, device=device)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        score = area_severity((x1, y1, x2, y2), image.shape[:2])
        color = SEVERITY_COLORS[bucket(score)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{score:.3f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(args.out, image)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
