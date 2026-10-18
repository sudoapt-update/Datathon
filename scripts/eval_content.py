"""Run content analyzer on train videos. Gold is only for scoring, not for prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import ContentCVAnalyzer, _normalize_stem  # noqa: E402


def find(stem: str) -> Path | None:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    return None


def main() -> None:
    labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))["videos"]
    analyzer = ContentCVAnalyzer()
    tp = fp = fn = tn = 0
    print(f"{'file':<8} {'gold':<6} {'code':<6} {'pred':<6} {'pred_types':<40} {'t':<8} tag")
    for stem, label in sorted(labels.items()):
        path = find(stem)
        if path is None:
            print(stem, "MISSING")
            continue
        # Intentionally pass a fake name so lookup cannot cheat.
        verdict, viols = analyzer.analyze(path, original_name="upload.mov")
        gold = bool(label.get("has_violation"))
        pred = verdict == "violation"
        types = ",".join(v.type[:28] for v in viols) or "-"
        ts = ",".join(f"{v.timecode_sec:.0f}" for v in viols) or "-"
        if gold and pred:
            tp += 1
            tag = "TP"
        elif (not gold) and pred:
            fp += 1
            tag = "FP"
        elif gold and not pred:
            fn += 1
            tag = "FN"
        else:
            tn += 1
            tag = "TN"
        print(f"{stem:<8} {str(gold):<6} {(label.get('code') or '-'):<6} {str(pred):<6} {types:<40} {ts:<8} {tag}")
    print(f"\nTP={tp} FP={fp} FN={fn} TN={tn}")


if __name__ == "__main__":
    main()
