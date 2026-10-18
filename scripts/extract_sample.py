"""Sanity-check frame extraction against the training .mov files."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.video import evidence_frames, iter_frames, video_meta  # noqa: E402


def _find_video(stem: str) -> Path | None:
    directory = Path(settings.video_dir)
    if not directory.exists():
        return None
    wanted = stem.upper()
    for path in directory.iterdir():
        if not path.is_file():
            continue
        name = path.stem.translate(str.maketrans("Хх", "Xx")).upper()
        if name == wanted:
            return path
    return None


def _run_one(stem: str, sample_iter: bool) -> None:
    path = _find_video(stem)
    if path is None:
        print(f"[skip] {stem}: file not found in {settings.video_dir}")
        return

    duration, native_fps, frame_count = video_meta(path)
    print(f"\n=== {path.name} ===")
    print(f"size_mb={path.stat().st_size / 1e6:.1f} duration={duration:.2f}s fps={native_fps:.2f} frames={frame_count}")

    if sample_iter:
        started = time.perf_counter()
        sampled = iter_frames(path)
        elapsed = time.perf_counter() - started
        first = sampled[0].timestamp_sec if sampled else None
        last = sampled[-1].timestamp_sec if sampled else None
        print(f"iter_frames fps={settings.frame_fps}: n={len(sampled)} first={first} last={last} wall={elapsed:.2f}s")

    ts = 22.0
    if stem.upper() != "X-1":
        ts = min(5.0, max(0.0, duration / 2 if duration else 5.0))

    out_dir = settings.data_dir / "jobs" / "_phase2_sample" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    evidence = evidence_frames(path, ts, n=3, span_sec=settings.evidence_span_sec)
    elapsed = time.perf_counter() - started
    print(f"evidence around {ts}s: {len(evidence)} frames wall={elapsed:.2f}s")
    for i, frame in enumerate(evidence, start=1):
        dest = out_dir / f"{i}_{frame.timestamp_sec:.2f}s.jpg"
        dest.write_bytes(frame.jpeg_bytes)
        print(f"  wrote {dest} ({len(frame.jpeg_bytes)} bytes, b64={len(frame.image_base64)})")


def main() -> None:
    print(f"VIDEO_DIR={settings.video_dir}")
    _run_one("X-1", sample_iter=True)
    _run_one("X-14", sample_iter=True)
    print("\nDone. Open data/jobs/_phase2_sample/ and check the JPEGs.")


if __name__ == "__main__":
    main()
