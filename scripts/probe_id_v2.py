"""Hold time with the production weights."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402
from app.services import rules_engine as rules  # noqa: E402


def find(stem: str) -> Path:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    raise FileNotFoundError(stem)


def main() -> None:
    zones = rules.load_zones()
    for stem in ("X-16", "X-17", "X-2"):
        samples = rules.collect_samples(find(stem), fps=settings.rules_fps)
        hits = [hit for hit in rules.decide(samples, zones) if hit.rule_id == "no_client_identification"]
        run = best = gap = 0.0
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
        print(stem, "hold", round(best, 1), "id", [(round(hit.t, 1)) for hit in hits])


if __name__ == "__main__":
    main()
