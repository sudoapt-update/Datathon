"""Compare the glass-door region on open-door videos and clean videos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402


def main() -> None:
    labels = json.loads((ROOT / "data" / "labels.json").read_text(encoding="utf-8"))["videos"]
    files = {
        _normalize_stem(p.stem): p
        for p in Path(settings.video_dir).iterdir()
        if p.suffix.lower() == ".mov"
    }
    for stem in ["X-10", "X-31", "X-30", "X-32", "X-33", "X-5", "X-3"]:
        cap = cv2.VideoCapture(str(files[stem]))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = n / fps
        vals = []
        t = 1.0
        while t < dur:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            roi = img[int(0.08 * h) : int(0.42 * h), int(0.02 * w) : int(0.28 * w)]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            vals.append((float(hsv[:, :, 2].mean()), float(hsv[:, :, 1].mean())))
            t += 2.0
        cap.release()
        v = np.mean([a for a, _ in vals])
        s = np.mean([b for _, b in vals])
        print(f"{stem} {labels[stem].get('code')} doorV={v:.1f} doorS={s:.1f}")


if __name__ == "__main__":
    main()
