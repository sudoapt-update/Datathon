"""Score Qwen2.5-VL-7B-AWQ on all gold train videos. Filename is not in the prompt."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.config import settings  # noqa: E402
from smoke_vlm import (  # noqa: E402
    N_FRAMES,
    find_video,
    infer,
    load_model,
    parse_json,
    sample_pils,
)
import torch  # noqa: E402

OUT_PATH = ROOT / "reports" / "eval_vlm.json"


def _pred_violation(parsed: dict | None) -> bool:
    if not parsed:
        return False
    verdict = str(parsed.get("verdict") or "").lower()
    return verdict == "violation"


def _pred_ts(parsed: dict | None) -> float | None:
    if not parsed:
        return None
    viols = parsed.get("violations") or []
    if not viols:
        return None
    try:
        return float(viols[0].get("timecode_sec"))
    except (TypeError, ValueError, AttributeError):
        return None


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))["videos"]
    stems = sorted(labels.keys(), key=lambda s: (len(s), s))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, dict] = {}
    if OUT_PATH.exists():
        prev = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        for row in prev.get("rows") or []:
            if row.get("stem") and row.get("tag") not in (None, "ERR"):
                done[row["stem"]] = row
        print("resume", len(done), "already scored")

    processor, model = load_model()
    print("VRAM after load, GB", round(torch.cuda.memory_allocated() / 1024**3, 2))
    print(f"{'file':<8} {'gold':<6} {'code':<6} {'pred':<6} {'t':<8} {'wall':<6} tag")

    rows: list[dict] = []
    tp = fp = fn = tn = err = 0

    def flush() -> None:
        summary = {
            "n_frames": N_FRAMES,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "err": err,
            "rows": rows,
        }
        OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for stem in stems:
        gold = labels[stem]
        gold_v = bool(gold.get("has_violation"))
        if stem in done:
            row = done[stem]
            rows.append(row)
            tag = row["tag"]
            if tag == "TP":
                tp += 1
            elif tag == "FP":
                fp += 1
            elif tag == "FN":
                fn += 1
            elif tag == "TN":
                tn += 1
            print(
                f"{stem:<8} {str(gold_v):<6} {(gold.get('code') or '-'):<6} "
                f"{str(row.get('pred')):<6} {str(row.get('pred_ts') or '-'):<8} "
                f"{str(row.get('wall_sec')):<6} {tag} (cached)"
            )
            continue

        try:
            path = find_video(stem)
        except FileNotFoundError:
            err += 1
            row = {"stem": stem, "tag": "ERR", "error": "missing video"}
            rows.append(row)
            print(stem, "MISSING")
            flush()
            continue

        started = time.perf_counter()
        try:
            frames = sample_pils(path, N_FRAMES)
            raw = infer(processor, model, frames)
            parsed = parse_json(raw)
            elapsed = round(time.perf_counter() - started, 1)
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.perf_counter() - started, 1)
            err += 1
            row = {
                "stem": stem,
                "file": path.name,
                "gold": gold_v,
                "code": gold.get("code"),
                "gold_ts": gold.get("timestamp_sec"),
                "pred": None,
                "pred_ts": None,
                "wall_sec": elapsed,
                "tag": "ERR",
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-800:],
            }
            rows.append(row)
            print(stem, "ERR", elapsed, exc)
            torch.cuda.empty_cache()
            flush()
            continue

        pred = _pred_violation(parsed)
        pred_ts = _pred_ts(parsed)
        if gold_v and pred:
            tp += 1
            tag = "TP"
        elif (not gold_v) and pred:
            fp += 1
            tag = "FP"
        elif gold_v and not pred:
            fn += 1
            tag = "FN"
        else:
            tn += 1
            tag = "TN"

        viols = (parsed or {}).get("violations") or []
        row = {
            "stem": stem,
            "file": path.name,
            "gold": gold_v,
            "code": gold.get("code"),
            "gold_ts": gold.get("timestamp_sec"),
            "gold_desc": gold.get("description"),
            "pred": pred,
            "pred_ts": pred_ts,
            "wall_sec": elapsed,
            "tag": tag,
            "parsed": parsed,
            "viol_types": [v.get("type") for v in viols if isinstance(v, dict)],
        }
        rows.append(row)
        print(
            f"{stem:<8} {str(gold_v):<6} {(gold.get('code') or '-'):<6} "
            f"{str(pred):<6} {str(pred_ts if pred_ts is not None else '-'):<8} {elapsed:<6} {tag}"
        )
        torch.cuda.empty_cache()
        flush()

    print(f"\nTP={tp} FP={fp} FN={fn} TN={tn} ERR={err}")
    denom = tp + fp
    rec_d = tp + fn
    print("precision", round(tp / denom, 3) if denom else None)
    print("recall", round(tp / rec_d, 3) if rec_d else None)
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
