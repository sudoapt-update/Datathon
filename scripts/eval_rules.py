"""Score detector+rules on the train videos. Gold is used only after prediction."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.rules.codes import CODE_TO_CATALOG_ID  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402
from app.services.rules_engine import collect_samples, decide, load_zones  # noqa: E402


def find(stem: str) -> Path | None:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    return None


def main() -> None:
    labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))["videos"]
    zones = load_zones()
    tp = fp = fn = tn = 0
    type_ok = 0
    rows = []
    print(f"{'file':<8} {'gold':<6} {'code':<6} {'pred':<6} {'rules':<42} {'t':<8} tag")
    for stem, label in sorted(labels.items(), key=lambda item: item[0]):
        path = find(stem)
        if path is None:
            print(stem, "MISSING")
            continue
        started = time.perf_counter()
        samples = collect_samples(path, fps=settings.rules_fps)
        hits = decide(samples, zones)
        elapsed = time.perf_counter() - started
        gold = bool(label.get("has_violation"))
        pred = bool(hits)
        rules = ",".join(hit.rule_id for hit in hits) or "-"
        ts = ",".join(f"{hit.t:.0f}" for hit in hits) or "-"
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
        expected = CODE_TO_CATALOG_ID.get(label.get("code") or "")
        if expected and any(hit.rule_id == expected for hit in hits):
            type_ok += 1
            tag += "+type"
        print(f"{stem:<8} {str(gold):<6} {(label.get('code') or '-'):<6} {str(pred):<6} {rules:<42} {ts:<8} {tag} {elapsed:.1f}s")
        rows.append(
            {
                "stem": stem,
                "gold": gold,
                "code": label.get("code"),
                "expected": expected,
                "pred": pred,
                "rules": [hit.rule_id for hit in hits],
                "times": [round(hit.t, 2) for hit in hits],
                "tag": tag,
                "sec": round(elapsed, 2),
            }
        )
    summary = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "type_ok": type_ok,
        "rows": rows,
    }
    out = ROOT / "reports" / "eval_rules.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTP={tp} FP={fp} FN={fn} TN={tn} type_ok={type_ok}")
    print("saved", out)


if __name__ == "__main__":
    main()
