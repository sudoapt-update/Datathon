from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings
from app.services.analyzer import ContentCVAnalyzer, _normalize_stem


def find(stem: str) -> Path:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    raise FileNotFoundError(stem)


def main() -> None:
    a = ContentCVAnalyzer()
    for stem in ["X-1", "X-4", "X-15", "X-19"]:
        path = find(stem)
        v, viols = a.analyze(path, original_name="totally_other_name.mov")
        types = [x.type for x in viols]
        ts = [x.timecode_sec for x in viols]
        print(stem, v, types, ts)


if __name__ == "__main__":
    main()
