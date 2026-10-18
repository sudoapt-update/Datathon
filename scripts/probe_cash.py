"""Color blobs on the desk, relative to the laptop anchor."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402


def laptop_center(result, w: int, h: int):
    best = None
    for box in result.boxes:
        name = result.names[int(box.cls[0])]
        if name != "laptop":
            continue
        conf = float(box.conf[0])
        if best and conf <= best[0]:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        best = (conf, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h)
    return best


def cash_blobs(img, lx: float, ly: float):
    h, w = img.shape[:2]
    x1 = int(max(0, (lx - 0.42) * w))
    x2 = int(min(w, (lx - 0.02) * w))
    y1 = int(max(0, (ly - 0.10) * h))
    y2 = int(min(h, (ly + 0.14) * h))
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 0, 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    hue = hsv[:, :, 0]
    # Banknotes are saturated; wood is not. Drop the blue laptop-screen hue.
    mask = ((sat > 80) & (val > 70) & ~((hue > 90) & (hue < 140))).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    blobs = 0
    area = 0
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if 40 <= a <= 2500:
            blobs += 1
            area += a
    return blobs, area


def main() -> None:
    model = YOLO(str(ROOT / "yolo11m.pt"))
    model.to("cuda")
    labels = json.loads((ROOT / "data" / "labels.json").read_text(encoding="utf-8"))["videos"]
    files = {
        _normalize_stem(p.stem): p
        for p in Path(settings.video_dir).iterdir()
        if p.suffix.lower() == ".mov"
    }
    for stem, lab in labels.items():
        cap = cv2.VideoCapture(str(files[stem]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = n / fps if fps else 0
        ts = lab["timestamp_sec"] if lab["timestamp_sec"] is not None else min(5.0, dur / 2)
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
        ok, img = cap.read()
        cap.release()
        if not ok:
            continue
        h, w = img.shape[:2]
        result = model(img, verbose=False, conf=0.25)[0]
        lap = laptop_center(result, w, h)
        if lap is None:
            print(f"{stem} {lab.get('code')} t={ts} NO_LAPTOP")
            continue
        blobs, area = cash_blobs(img, lap[1], lap[2])
        print(f"{stem} {str(lab.get('code')):5} t={ts:5} blobs={blobs} area={area} lap={lap[1]:.2f},{lap[2]:.2f}")


if __name__ == "__main__":
    main()
