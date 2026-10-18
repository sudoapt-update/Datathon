"""Print the leave timeline so evidence times can be chosen by eye."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402
from app.services.rules_engine import (  # noqa: E402
    _desk_in_service,
    _near,
    _objs,
    _seen,
    _stepped_away,
    collect_samples,
    decide,
    load_zones,
)


def find(stem: str) -> Path:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    raise FileNotFoundError(stem)


def main() -> None:
    zones = load_zones()
    for stem in ("X-28", "X-11", "X-22", "X-24", "X-15"):
        samples = collect_samples(find(stem), fps=settings.rules_fps)
        hits = [hit for hit in decide(samples, zones) if hit.rule_id == "cashier_leaves_during_service"]
        anchor = None
        from app.services.rules_engine import _cashier_anchor

        anchor = _cashier_anchor(samples, zones)
        print(f"\n{stem} hits={[(round(h.t, 1)) for h in hits]} anchor={None if anchor is None else (round(anchor[0], 2), round(anchor[1], 2))}")
        if anchor is None:
            continue
        for index, sample in enumerate(samples):
            if sample.t < 2 or sample.t > 28:
                continue
            here = _near(sample, zones, anchor, zones["cashier_radius"])
            stepped = _stepped_away(sample, zones, anchor)
            people = "no" if not any(d.name == "person" for d in sample.dets) else "yes"
            screen = "on" if _seen(sample, "screen_on", 0.45) else "-"
            laptop = "lap" if _seen(sample, "laptop", 0.45) else "-"
            lock = "lock" if _seen(sample, "screen_locked", 0.45) else "-"
            cash = "cash" if _objs(sample, "cash_stack", 0.4) else "-"
            bright = "-" if sample.screen_brightness is None else f"{sample.screen_brightness:.0f}"
            open_desk = "OPEN" if _desk_in_service(sample, samples[: index + 1], zones) else "shut"
            if here or stepped or sample.t % 2 < 0.4:
                print(
                    f"  {sample.t:5.1f} {'HERE' if here else 'away':4} {'step' if stepped else '    '} p={people:3} {screen:3} {laptop:3} {lock:4} {cash:4} br={bright:>4} {open_desk}"
                )


if __name__ == "__main__":
    main()
