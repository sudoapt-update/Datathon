"""Save door evidence frames for a few clips. Does not call Qwen."""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402
from app.services.rules_engine import (  # noqa: E402
    RulesAnalyzer,
    collect_samples,
    decide,
    hits_to_violations,
)


def find(stem: str) -> Path:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    raise FileNotFoundError(stem)


def main() -> None:
    analyzer = RulesAnalyzer()
    out = ROOT / "data" / "preview" / "door_evidence"
    out.mkdir(parents=True, exist_ok=True)
    for stem in ("X-33", "X-30", "X-32", "X-26", "X-11"):
        path = find(stem)
        samples = collect_samples(path, fps=settings.rules_fps)
        hits = decide(samples, analyzer.zones)
        violations = hits_to_violations(path, hits, analyzer.rules)
        print(stem, [(hit.rule_id, round(hit.t, 1), [round(t, 1) for t in (hit.evidence_at or [])]) for hit in hits])
        for index, violation in enumerate(violations):
            for frame_index, frame in enumerate(violation.evidence_frames):
                name = f"{stem}_{index}_{frame_index}_{frame.timestamp_sec:.1f}.jpg"
                (out / name).write_bytes(base64.b64decode(frame.image_base64))
                print(" ", name, violation.type[:24])


if __name__ == "__main__":
    main()
