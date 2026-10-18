"""Wrist-to-hip distance around a few gold timestamps."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402


def main() -> None:
    model = YOLO("yolo11n-pose.pt")
    model.to("cuda")
    files = {
        _normalize_stem(p.stem): p
        for p in Path(settings.video_dir).iterdir()
        if p.suffix.lower() == ".mov"
    }
    # X-1 pocket at 22s, X-3 clean, X-40 bag
    for stem, times in {"X-1": [8, 16, 20, 22, 24], "X-3": [4, 8, 12], "X-18": [8, 16, 20]}.items():
        cap = cv2.VideoCapture(str(files[stem]))
        print("---", stem)
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            result = model(img, verbose=False, conf=0.4)[0]
            if result.keypoints is None or result.keypoints.xy is None:
                print(f"  t={t} no kpts")
                continue
            xy = result.keypoints.xy.cpu().numpy()
            for person in xy:
                if person.shape[0] < 13:
                    continue
                lw, rw, lh, rh = person[9], person[10], person[11], person[12]
                def dist(a, b):
                    if a[0] <= 1 or b[0] <= 1:
                        return -1
                    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 / max(h, w)
                print(
                    f"  t={t:4.0f} hip=({lh[0]/w:.2f},{lh[1]/h:.2f}) "
                    f"dL={dist(lw, lh):.3f} dR={dist(rw, rh):.3f}"
                )
        cap.release()


if __name__ == "__main__":
    main()
