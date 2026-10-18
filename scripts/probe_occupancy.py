from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402


def find(stem: str) -> Path | None:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    return None


def main() -> None:
    labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))["videos"]
    model = YOLO(str(settings.yolo_weights))
    if torch.cuda.is_available():
        model.to("cuda")
    samples = ["X-1", "X-4", "X-11", "X-19", "X-30", "X-40", "X-18", "X-3"]
    for stem in samples:
        lab = labels[stem]
        path = find(stem)
        if path is None:
            print(stem, "missing")
            continue
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = n / fps if fps else 0
        print(f"\n=== {stem} gold={lab.get('code')} t={lab.get('timestamp_sec')} dur={duration:.1f} ===")
        t = 0.0
        while t < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                break
            height, width = img.shape[:2]
            result = model(img, verbose=False)[0]
            people = []
            others = []
            for box in result.boxes:
                cls = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                if conf < 0.45:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
                area = (x2 - x1) * (y2 - y1) / (width * height)
                if cls == "person":
                    people.append((round(cx, 2), round(cy, 2), round(area, 3), round(conf, 2)))
                elif cls in {"cell phone", "backpack", "handbag", "suitcase", "laptop"}:
                    others.append((cls, round(cx, 2), round(cy, 2), round(conf, 2)))
            desk = [p for p in people if p[1] >= 0.32 and p[2] >= 0.02]
            gold = lab.get("timestamp_sec")
            mark = " <GOLD" if gold is not None and abs(t - float(gold)) < 0.8 else ""
            print(f" t={t:5.1f} people={len(people)} desk={len(desk)} {others}{mark}")
            t += 1.0
        cap.release()


if __name__ == "__main__":
    main()
