from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.config import settings


@dataclass(frozen=True)
class Frame:
    timestamp_sec: float
    jpeg_bytes: bytes
    image_base64: str


class VideoOpenError(RuntimeError):
    pass


def _open_capture(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise VideoOpenError(f"Cannot open video: {video_path}")
    return cap


def video_meta(video_path: Path) -> tuple[float, float, int]:
    """Return (duration_sec, native_fps, frame_count)."""
    cap = _open_capture(video_path)
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if native_fps > 1e-3 and frame_count > 0:
            return frame_count / native_fps, native_fps, frame_count
        return 0.0, native_fps or 30.0, frame_count
    finally:
        cap.release()


def _clock(timestamp_sec: float) -> str:
    total = max(0.0, float(timestamp_sec))
    minutes = int(total // 60)
    seconds = total - minutes * 60
    return f"{minutes:02d}:{seconds:04.1f}"


def _stamp_timecode(bgr: np.ndarray, timestamp_sec: float) -> np.ndarray:
    image = bgr.copy()
    text = _clock(timestamp_sec)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 12, 16 + height
    cv2.rectangle(image, (x - 6, y - height - 6), (x + width + 6, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def _encode_jpeg(bgr: np.ndarray, timestamp_sec: float | None = None) -> bytes:
    max_width = settings.evidence_max_width
    quality = settings.evidence_jpeg_quality
    height, width = bgr.shape[:2]
    if width > max_width > 0:
        scale = max_width / width
        bgr = cv2.resize(
            bgr,
            (max_width, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if timestamp_sec is not None:
        bgr = _stamp_timecode(bgr, timestamp_sec)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def to_frame(bgr: np.ndarray, timestamp_sec: float, stamp: bool = False) -> Frame:
    jpeg = _encode_jpeg(bgr, timestamp_sec if stamp else None)
    return Frame(
        timestamp_sec=round(float(timestamp_sec), 3),
        jpeg_bytes=jpeg,
        image_base64=base64.b64encode(jpeg).decode("ascii"),
    )


def iter_bgr_frames(video_path: Path, fps: float | None = None):
    """Yield (timestamp_sec, BGR ndarray) without JPEG encode — for the detector."""
    sample_fps = settings.frame_fps if fps is None else fps
    if sample_fps <= 0:
        raise ValueError("fps must be positive")
    cap = _open_capture(video_path)
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        interval = 1.0 / sample_fps
        next_t = 0.0
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            timestamp = msec / 1000.0 if msec > 0 else index / native_fps
            if timestamp + 1e-4 >= next_t:
                yield round(float(timestamp), 3), bgr
                next_t += interval
            index += 1
    finally:
        cap.release()


def iter_frames(video_path: Path, fps: float | None = None) -> list[Frame]:
    """Uniform sample along the timeline. Decodes sequentially (reliable for .mov)."""
    sample_fps = settings.frame_fps if fps is None else fps
    if sample_fps <= 0:
        raise ValueError("fps must be positive")

    cap = _open_capture(video_path)
    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        interval = 1.0 / sample_fps
        frames: list[Frame] = []
        next_t = 0.0
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            timestamp = msec / 1000.0 if msec > 0 else index / native_fps
            if timestamp + 1e-4 >= next_t:
                frames.append(to_frame(bgr, timestamp))
                next_t += interval
            index += 1
        return frames
    finally:
        cap.release()


def frame_at(video_path: Path, timestamp_sec: float, stamp: bool = False) -> Frame:
    duration, native_fps, frame_count = video_meta(video_path)
    target = max(0.0, float(timestamp_sec))
    if duration > 0:
        # Stay inside the last readable frame.
        target = min(target, max(0.0, duration - 1.0 / max(native_fps, 1.0)))

    cap = _open_capture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
        ok, bgr = cap.read()
        if not ok and frame_count > 0:
            frame_idx = min(frame_count - 1, int(round(target * native_fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame at {timestamp_sec}s from {video_path}")
        msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        actual = msec / 1000.0 if msec > 0 else target
        return to_frame(bgr, actual, stamp=stamp)
    finally:
        cap.release()


def evidence_frames(
    video_path: Path,
    timestamp_sec: float,
    n: int = 3,
    span_sec: float = 1.0,
    times: list[float] | None = None,
) -> list[Frame]:
    """1–3 JPEGs. Explicit times win; otherwise frames sit around the event."""
    duration, _, _ = video_meta(video_path)
    if times:
        wanted = [float(item) for item in times[:3]]
    else:
        count = max(1, min(3, int(n)))
        t = max(0.0, float(timestamp_sec))
        if duration > 0:
            t = min(t, duration)
        if count == 1:
            wanted = [t]
        elif count == 2:
            wanted = [t, t + span_sec]
        else:
            wanted = [t - span_sec, t, t + span_sec]

    picked: list[Frame] = []
    seen_ts: set[float] = set()
    for ts in wanted:
        if duration > 0:
            ts = min(max(0.0, ts), duration)
        else:
            ts = max(0.0, ts)
        frame = frame_at(video_path, ts, stamp=True)
        key = round(frame.timestamp_sec, 2)
        if key in seen_ts:
            continue
        seen_ts.add(key)
        picked.append(frame)
    return picked
