from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLOWorld

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
    model = YOLOWorld(str(ROOT / "yolov8s-worldv2.pt"))
    model.set_classes(
        [
            "person",
            "banknote",
            "stack of cash",
            "paper money",
            "cell phone",
            "backpack",
            "handbag",
            "open door",
            "safe",
        ]
    )
    if torch.cuda.is_available():
        model.to("cuda")
    labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))["videos"]
    for stem in ["X-1", "X-4", "X-11", "X-40", "X-18", "X-19"]:
        lab = labels[stem]
        path = find(stem)
        ts = float(lab["timestamp_sec"]) if lab.get("timestamp_sec") is not None else 5.0
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ok, img = cap.read()
        cap.release()
        if not ok:
            print(stem, "read fail")
            continue
        result = model(img, verbose=False, conf=0.15)[0]
        dets = []
        for box in result.boxes:
            name = result.names[int(box.cls[0])]
            conf = float(box.conf[0])
            dets.append(f"{name}:{conf:.2f}")
        print(stem, lab.get("code"), f"t={ts}", dets)


if __name__ == "__main__":
    main()
