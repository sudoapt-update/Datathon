"""Check the 3-second document hold on a few clips, using the v4 weights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402
from app.services import rules_engine as rules  # noqa: E402


def find(stem: str) -> Path:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    raise FileNotFoundError(stem)


def hold_seconds(samples, zones) -> float:
    best = run = gap = 0.0
    for index, sample in enumerate(samples):
        step = rules._dt(samples, index)
        if rules._document_in_hand(sample, zones):
            run += step
            gap = 0.0
            best = max(best, run)
        else:
            gap += step
            if gap > 0.8:
                run = 0.0
                gap = 0.0
    return best


def main() -> None:
    rules._CASHIER = rules._to_cuda(YOLO(str(ROOT / "runs" / "detect" / "cashier4" / "weights" / "best.pt")))
    zones = rules.load_zones()
    for stem in ("X-2", "X-16", "X-17", "X-3", "X-5", "X-38", "X-34", "X-10"):
        samples = rules.collect_samples(find(stem), fps=settings.rules_fps)
        hits = rules.decide(samples, zones)
        shown = [(hit.rule_id, round(hit.t, 1)) for hit in hits]
        print(stem, "hold", round(hold_seconds(samples, zones), 1), "hits", shown)


if __name__ == "__main__":
    main()
