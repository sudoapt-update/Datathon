"""Compare teammate gold labels vs COCO YOLO at the labeled timestamp.

Does not walk the whole video (that is slow). Prints what YOLO sees on the
gold frame so misses like X-1 / X-4 are explainable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.rules.codes import CODE_TO_CATALOG_ID  # noqa: E402
from app.services.analyzer import YoloCVAnalyzer, _normalize_stem  # noqa: E402
from app.services.video import frame_at  # noqa: E402


def _find_video(stem: str) -> Path | None:
    directory = Path(settings.video_dir)
    if not directory.exists():
        return None
    wanted = _normalize_stem(stem)
    for path in directory.iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    return None


def main() -> None:
    labels = json.loads(settings.labels_path.read_text(encoding="utf-8")).get("videos", {})
    yolo = YoloCVAnalyzer()
    rows: list[str] = []
    tp = fp = fn = tn = 0
    print(f"labels={settings.labels_path} videos={settings.video_dir}")
    print(f"{'file':<8} {'gold':<6} {'code':<6} {'yolo':<6} {'coco@gold':<40} match")

    for stem, label in sorted(labels.items(), key=lambda kv: kv[0]):
        path = _find_video(stem)
        if path is None:
            print(f"{stem:<8} MISSING_FILE")
            continue
        gold_pos = bool(label.get("has_violation"))
        ts = float(label["timestamp_sec"]) if gold_pos and label.get("timestamp_sec") is not None else 5.0
        frame = frame_at(path, ts)
        img = cv2.imdecode(np.frombuffer(frame.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        dets = yolo.detect_classes(img) if img is not None else []
        names = [name for name, _conf in dets]
        rule_id = yolo.classes_to_rule_id(names)
        yolo_pos = rule_id is not None
        if gold_pos and yolo_pos:
            tp += 1
            tag = "TP"
        elif (not gold_pos) and yolo_pos:
            fp += 1
            tag = "FP"
        elif gold_pos and not yolo_pos:
            fn += 1
            tag = "FN"
        else:
            tn += 1
            tag = "TN"
        coco = ", ".join(f"{n}:{c:.2f}" for n, c in dets[:6]) or "-"
        code = label.get("code") or "-"
        print(f"{stem:<8} {str(gold_pos):<6} {code:<6} {str(yolo_pos):<6} {coco:<40} {tag}")
        rows.append(tag)

    print()
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  n={len(rows)}")
    print("Gold types YOLO can theoretically hit: V-16 (bag), V-17 (phone).")
    print("V-01/V-04/V-05/V-11/V-18 need other rules — FN here is expected.")
    known = {CODE_TO_CATALOG_ID[c] for c in ("V-16", "V-17") if c in CODE_TO_CATALOG_ID}
    _ = known


if __name__ == "__main__":
    main()
