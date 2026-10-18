"""Ask Qwen to confirm or reject a cashier-left hypothesis on a few clips."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.rules_engine import Hit  # noqa: E402
from app.services.vlm_verify import _asks_confirm, _frames, _load  # noqa: E402
from app.config import settings  # noqa: E402
from app.services.analyzer import _normalize_stem  # noqa: E402

CASES = [
    ("X-22", 16.0, "violation, left the desk"),
    ("X-11", 19.0, "violation, left devices on"),
    ("X-23", 16.0, "CLEAN but walks to the door"),
    ("X-25", 27.0, "CLEAN brief step toward the door"),
    ("X-5", 30.0, "CLEAN cashier stays"),
]

RULE = {
    "title": "Кассир покинул рабочее место во время обслуживания клиента",
    "visual_cues": [
        "Кассир выходит из зоны кассы, пока клиент ещё у окна или на столе остаются деньги, документы или включённый ноутбук"
    ],
    "false_positive": "Кассир подошёл к двери или сейфу в конце уже завершённой операции, стол пуст",
}


def find(stem: str) -> Path:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    raise FileNotFoundError(stem)


def main() -> None:
    processor, model = _load()
    for stem, ts, note in CASES:
        path = find(stem)
        frames = _frames(path, ts)
        answer = _asks_confirm(processor, model, frames, RULE)
        print(stem, note, "confirm=", answer)


if __name__ == "__main__":
    main()
