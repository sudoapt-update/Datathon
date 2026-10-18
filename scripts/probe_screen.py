"""Laptop-screen brightness when the cashier is away from the opening position."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402

# stem, second to sample
SAMPLES = [
    ("X-11", 19),
    ("X-15", 16),
    ("X-22", 16),
    ("X-24", 10),
    ("X-26", 16),
    ("X-28", 12),
    ("X-12", 21),
    ("X-23", 16),
    ("X-25", 27),
    ("X-27", 21),
    ("X-29", 12),
    ("X-3", 8),
    ("X-5", 30),
]


def main() -> None:
    model = YOLO(str(ROOT / "yolo11m.pt"))
    model.to("cuda")
    files = {
        _normalize_stem(p.stem): p
        for p in Path(settings.video_dir).iterdir()
        if p.suffix.lower() == ".mov"
    }
    for stem, t in SAMPLES:
        cap = cv2.VideoCapture(str(files[stem]))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        cap.release()
        if not ok:
            print(stem, "read fail")
            continue
        h, w = img.shape[:2]
        result = model(img, verbose=False, conf=0.35)[0]
        laptops = []
        for box in result.boxes:
            if result.names[int(box.cls[0])] != "laptop":
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            crop = img[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            laptops.append(f"conf={float(box.conf[0]):.2f} mean={float(gray.mean()):.0f} p90={float(cv2.minMaxLoc(gray)[1]):.0f}")
        print(stem, "t", t, laptops or "NO_LAPTOP")


if __name__ == "__main__":
    main()
