"""Content-based cashier-discipline detector. Ignores the file name."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.schemas import EvidenceFrame, Violation
from app.services.video import iter_bgr_frames, to_frame


DESK_Y_MIN = 0.32
DESK_AREA_MIN = 0.02
EMPTY_PERSIST_SEC = 1.0
OCCUPIED_BEFORE_LEAVE_SEC = 3.0
IGNORE_HEAD_SEC = 2.0
IGNORE_TAIL_SEC = 2.0
CROWD_N = 3
CROWD_PERSIST_SEC = 2.0
BAG_CONF = 0.55
BAG_PERSIST_SEC = 1.0
BAG_CX_MIN, BAG_CX_MAX = 0.22, 0.82
PHONE_CONF = 0.55
PHONE_PERSIST_SEC = 1.0
BAG_CLASSES = {"backpack", "handbag", "suitcase"}


@dataclass
class Box:
    name: str
    conf: float
    cx: float
    cy: float
    area: float


@dataclass
class Track:
    had_cashier: bool = False
    occupied_sec: float = 0.0
    last_ts: float = 0.0
    empty_since: float | None = None
    crowd_since: float | None = None
    bag_since: float | None = None
    phone_since: float | None = None
    fired: set[str] = field(default_factory=set)


def _desk_people(boxes: list[Box]) -> list[Box]:
    return [b for b in boxes if b.name == "person" and b.cy >= DESK_Y_MIN and b.area >= DESK_AREA_MIN]


def _parse_boxes(result) -> list[Box]:
    names = result.names
    height, width = result.orig_shape
    out: list[Box] = []
    for box in result.boxes:
        conf = float(box.conf[0])
        name = names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append(
            Box(
                name=name,
                conf=conf,
                cx=(x1 + x2) / 2 / width,
                cy=(y1 + y2) / 2 / height,
                area=(x2 - x1) * (y2 - y1) / (width * height),
            )
        )
    return out


class ContentAnalyzer:
    """YOLO boxes + temporal rules. Does not read labels.json or original_name."""

    def __init__(self, model, rules: dict) -> None:
        self.model = model
        self.rules = rules

    def analyze(self, video_path: Path, original_name: str | None = None) -> tuple[str, list[Violation]]:
        _ = original_name
        track = Track()
        found: list[Violation] = []
        recent: deque[tuple[float, object]] = deque(maxlen=5)

        for timestamp, bgr in iter_bgr_frames(video_path):
            recent.append((timestamp, bgr))
            result = self.model(bgr, verbose=False)[0]
            boxes = _parse_boxes(result)
            dt = timestamp - track.last_ts if track.last_ts else 0.0
            self._update(timestamp, dt, boxes, track, found, recent)
            track.last_ts = timestamp

        duration = track.last_ts
        found = [v for v in found if v.timecode_sec <= max(0.0, duration - IGNORE_TAIL_SEC)]
        if found:
            return "violation", found
        return "no_violation", []

    def _update(
        self,
        timestamp: float,
        dt: float,
        boxes: list[Box],
        track: Track,
        found: list[Violation],
        recent: deque,
    ) -> None:
        desk = _desk_people(boxes)
        if desk:
            track.had_cashier = True
            track.occupied_sec += dt
            track.empty_since = None
        elif track.had_cashier:
            if track.empty_since is None:
                track.empty_since = timestamp
            elif (
                timestamp - track.empty_since >= EMPTY_PERSIST_SEC
                and track.occupied_sec >= OCCUPIED_BEFORE_LEAVE_SEC
                and track.empty_since >= IGNORE_HEAD_SEC
            ):
                self._fire(
                    track,
                    found,
                    recent,
                    "cashier_leaves_during_service",
                    track.empty_since,
                    "Сотрудник зоны кассы пропадает из кадра после того, как рабочее место было занято.",
                )

        if len(desk) >= CROWD_N:
            if track.crowd_since is None:
                track.crowd_since = timestamp
            elif timestamp - track.crowd_since >= CROWD_PERSIST_SEC and track.crowd_since >= IGNORE_HEAD_SEC:
                self._fire(
                    track,
                    found,
                    recent,
                    "unauthorized_person_in_cash_desk",
                    track.crowd_since,
                    f"В зоне кассы одновременно видно не менее {CROWD_N} человек.",
                )
        else:
            track.crowd_since = None

        bags = [
            b
            for b in boxes
            if b.name in BAG_CLASSES and b.conf >= BAG_CONF and BAG_CX_MIN <= b.cx <= BAG_CX_MAX and b.cy >= DESK_Y_MIN
        ]
        if bags:
            if track.bag_since is None:
                track.bag_since = timestamp
            elif timestamp - track.bag_since >= BAG_PERSIST_SEC:
                self._fire(
                    track,
                    found,
                    recent,
                    "personal_items_in_cash_desk",
                    track.bag_since,
                    f"В зоне кассы видна личная сумка ({bags[0].name}, conf={bags[0].conf:.2f}).",
                )
        else:
            track.bag_since = None

        phones = [b for b in boxes if b.name == "cell phone" and b.conf >= PHONE_CONF]
        if phones:
            if track.phone_since is None:
                track.phone_since = timestamp
            elif timestamp - track.phone_since >= PHONE_PERSIST_SEC:
                self._fire(
                    track,
                    found,
                    recent,
                    "phone_with_camera_in_cash_desk",
                    track.phone_since,
                    f"На кадре виден смартфон (conf={phones[0].conf:.2f}).",
                )
        else:
            track.phone_since = None

    def _fire(
        self,
        track: Track,
        found: list[Violation],
        recent: deque,
        rule_id: str,
        timestamp: float,
        observation: str,
    ) -> None:
        if rule_id in track.fired:
            return
        rule = self.rules.get(rule_id)
        if not rule:
            return
        track.fired.add(rule_id)
        evidence = []
        for ts, bgr in list(recent)[-3:]:
            frame = to_frame(bgr, ts)
            evidence.append(EvidenceFrame(timestamp_sec=frame.timestamp_sec, image_base64=frame.image_base64))
        if not evidence:
            return
        found.append(
            Violation(
                type=rule["title"],
                timecode_sec=round(timestamp, 2),
                description=f"{observation} {rule['visual_cues'][0]} Это подсказка по видео, не юридический вывод.",
                evidence_frames=evidence,
                rule_document=rule["legal_source"]["document"],
                rule_section=rule["legal_source"]["section"],
            )
        )
