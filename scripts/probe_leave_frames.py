"""Save the new leave evidence frames."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402
from app.services.rules_engine import collect_samples, decide, hits_to_violations, load_zones  # noqa: E402
from app.services.video import evidence_frames  # noqa: E402
import json  # noqa: E402


def find(stem: str) -> Path:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    raise FileNotFoundError(stem)


def main() -> None:
    zones = load_zones()
    catalog = json.loads((ROOT / "app" / "rules" / "catalog.json").read_text(encoding="utf-8"))
    rules = {item["id"]: item for item in catalog["violation_types"]}
    out = ROOT / "data" / "preview" / "leave_evidence"
    out.mkdir(parents=True, exist_ok=True)
    for stem in ("X-28", "X-11", "X-22", "X-24", "X-26"):
        path = find(stem)
        samples = collect_samples(path, fps=settings.rules_fps)
        hits = [hit for hit in decide(samples, zones) if hit.rule_id == "cashier_leaves_during_service"]
        for hit in hits:
            print(stem, round(hit.t, 1), [round(t, 1) for t in (hit.evidence_at or [])])
            print(" ", hit.observation)
            frames = evidence_frames(path, hit.t, times=hit.evidence_at)
            for index, frame in enumerate(frames):
                name = f"{stem}_{index}_{frame.timestamp_sec:.1f}.jpg"
                (out / name).write_bytes(frame.jpeg_bytes)
                print(" ", name)
        if not hits:
            print(stem, "no leave")
        _ = hits_to_violations


if __name__ == "__main__":
    main()
