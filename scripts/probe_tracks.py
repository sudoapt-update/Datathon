"""Print person tracks for a few train videos. Gold is not used for detection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402

STEMS = ["X-12", "X-15", "X-23", "X-25", "X-27", "X-29", "X-31", "X-35", "X-39"]


def main() -> None:
    model = YOLO(str(ROOT / "yolo11m.pt"))
    model.to("cuda")
    labels = json.loads((ROOT / "data" / "labels.json").read_text(encoding="utf-8"))["videos"]
    files = {
        _normalize_stem(p.stem): p
        for p in Path(settings.video_dir).iterdir()
        if p.suffix.lower() == ".mov"
    }
    for stem in STEMS:
        cap = cv2.VideoCapture(str(files[stem]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = n / fps
        print("---", stem, labels[stem].get("code"), f"dur={dur:.0f}")
        t = 0.0
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            result = model(img, verbose=False, conf=0.35)[0]
            people = []
            extra = []
            for box in result.boxes:
                name = result.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                area = (x2 - x1) * (y2 - y1) / (w * h)
                if name == "person" and area > 0.01:
                    people.append(f"{cx:.2f},{cy:.2f}")
                elif name in {"laptop", "handbag", "backpack", "cell phone", "suitcase"}:
                    extra.append(f"{name[:4]}:{float(box.conf[0]):.2f}@{cx:.2f},{cy:.2f}")
            print(f"  t={t:5.1f} p=[{' '.join(people)}] {' '.join(extra)}")
            t += 3.0
        cap.release()


if __name__ == "__main__":
    main()
