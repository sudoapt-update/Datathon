"""Send the train videos through the running API and score the job results."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.rules.codes import CODE_TO_CATALOG_ID  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402

BASE = "http://127.0.0.1:8000"
KEY = settings.api_key


def find(stem: str) -> Path | None:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    return None


def post(path: Path) -> str:
    boundary = "----datathon"
    filename = path.name.encode("utf-8")
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: video/quicktime\r\n\r\n"
    ).encode("utf-8")
    tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
    body = head + path.read_bytes() + tail
    request = urllib.request.Request(
        f"{BASE}/api/v1/analyze",
        data=body,
        headers={
            "X-API-Key": KEY,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["job_id"]


def poll(job_id: str, timeout: float = 240.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            f"{BASE}/api/v1/jobs/{job_id}",
            headers={"X-API-Key": KEY},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload["status"] != "processing":
            return payload
        time.sleep(1.0)
    raise TimeoutError(job_id)


def main() -> None:
    catalog = json.loads((ROOT / "app" / "rules" / "catalog.json").read_text(encoding="utf-8"))
    title_to_id = {item["title"]: item["id"] for item in catalog["violation_types"]}
    labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))["videos"]
    tp = fp = fn = tn = type_ok = 0
    rows = []
    print(f"{'file':<8} {'gold':<6} {'pred':<6} {'rules':<42} {'frames':<8} tag")
    for stem, label in sorted(labels.items(), key=lambda item: item[0]):
        path = find(stem)
        if path is None:
            print(stem, "MISSING")
            continue
        started = time.perf_counter()
        job_id = post(path)
        result = poll(job_id)
        elapsed = time.perf_counter() - started
        titles = [item["type"] for item in result.get("violations") or []]
        rules = [title_to_id.get(title, title) for title in titles]
        frames = [len(item.get("evidence_frames") or []) for item in result.get("violations") or []]
        gold = bool(label.get("has_violation"))
        pred = result.get("verdict") == "violation"
        if result.get("status") != "done":
            tag = "FAIL"
        elif gold and pred:
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
        if expected and expected in rules:
            type_ok += 1
            tag += "+type"
        shown = ",".join(rules) or "-"
        print(
            f"{stem:<8} {str(gold):<6} {str(pred):<6} {shown:<42} {','.join(map(str, frames)) or '-':<8} {tag} {elapsed:.1f}s {result.get('error') or ''}",
            flush=True,
        )
        rows.append(
            {
                "stem": stem,
                "gold": gold,
                "code": label.get("code"),
                "expected": expected,
                "status": result.get("status"),
                "verdict": result.get("verdict"),
                "rules": rules,
                "times": [item.get("timecode_sec") for item in result.get("violations") or []],
                "frames": frames,
                "error": result.get("error"),
                "tag": tag,
                "sec": round(elapsed, 2),
            }
        )
    summary = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "type_ok": type_ok, "rows": rows}
    out = ROOT / "reports" / "service_score.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn} type_ok={type_ok}")
    print(out)


if __name__ == "__main__":
    main()
