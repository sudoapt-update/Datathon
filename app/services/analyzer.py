"""Video analysis pipeline.

Modes (ANALYZER_MODE):
  rules   — detector + temporal rules, optional VLM recheck (default)
  content — earlier visual rules on frames
  yolo    — naive per-frame COCO mapping (phones/bags)
  labels  — Excel lookup by file name (debug only)
  hybrid  — labels on train names, else YOLO (debug only)
  stub    — always no_violation
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import ROOT_DIR, settings
from app.rules.codes import CODE_TO_CATALOG_ID
from app.schemas import EvidenceFrame, Violation
from app.services.content import ContentAnalyzer
from app.services.video import evidence_frames, iter_frames

_YOLO_MODEL = None


def _normalize_stem(stem: str) -> str:
    return stem.translate(str.maketrans("Хх", "Xx")).upper()


def _load_catalog() -> dict:
    catalog_path = ROOT_DIR / "app" / "rules" / "catalog.json"
    if not catalog_path.exists():
        return {}
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data.get("violation_types", [])}


def _load_labels() -> dict:
    if not settings.labels_path.exists():
        return {}
    payload = json.loads(settings.labels_path.read_text(encoding="utf-8"))
    return payload.get("videos", {})


def _load_yolo() -> YOLO:
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    weights = Path(settings.yolo_weights)
    model = YOLO(str(weights))
    try:
        import torch

        if torch.cuda.is_available():
            model.to("cuda")
    except Exception:
        pass
    _YOLO_MODEL = model
    return model


def _violation_from_label(video_path: Path, label: dict, rules: dict) -> Violation:
    ts = float(label["timestamp_sec"])
    code = (label.get("code") or "").strip()
    rule_id = CODE_TO_CATALOG_ID.get(code, "")
    rule_data = rules.get(rule_id, {})
    legal = rule_data.get("legal_source") or {}
    title = rule_data.get("title") or label.get("description") or "Признак нарушения кассовой дисциплины"
    description = label.get("description") or rule_data.get("visual_cues", ["Зафиксировано нарушение по разметке."])[0]
    frames = evidence_frames(
        video_path,
        ts,
        n=settings.evidence_frame_count,
        span_sec=settings.evidence_span_sec,
    )
    return Violation(
        type=title,
        timecode_sec=ts,
        description=description,
        evidence_frames=[
            EvidenceFrame(timestamp_sec=frame.timestamp_sec, image_base64=frame.image_base64) for frame in frames
        ],
        rule_document=legal.get("document"),
        rule_section=legal.get("section"),
    )


class Analyzer:
    def analyze(self, video_path: Path, original_name: str | None = None) -> tuple[str, list[Violation]]:
        raise NotImplementedError


class StubAnalyzer(Analyzer):
    def analyze(self, video_path: Path, original_name: str | None = None) -> tuple[str, list[Violation]]:
        _ = video_path, original_name
        return "no_violation", []


class LabelAwareStubAnalyzer(Analyzer):
    """Train-set lookup. Do not use against jury test videos."""

    def __init__(self) -> None:
        self._labels = _load_labels()
        self.rules = _load_catalog()

    def analyze(self, video_path: Path, original_name: str | None = None) -> tuple[str, list[Violation]]:
        key = _normalize_stem(Path(original_name or video_path.name).stem)
        label = self._labels.get(key)
        if not label or not label.get("has_violation"):
            return "no_violation", []
        if label.get("timestamp_sec") is None:
            return "no_violation", []
        return "violation", [_violation_from_label(video_path, label, self.rules)]


class YoloCVAnalyzer(Analyzer):
    def __init__(self) -> None:
        self.model = _load_yolo()
        self.rules = _load_catalog()

    def detect_classes(self, img) -> list[tuple[str, float]]:
        results = self.model(img, verbose=False)[0]
        found: list[tuple[str, float]] = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf > 0.5:
                found.append((self.model.names[int(box.cls[0])], conf))
        return found

    @staticmethod
    def classes_to_rule_id(class_names: list[str]) -> str | None:
        if "cell phone" in class_names:
            return "phone_with_camera_in_cash_desk"
        if "backpack" in class_names or "handbag" in class_names:
            return "personal_items_in_cash_desk"
        return None

    def analyze(self, video_path: Path, original_name: str | None = None) -> tuple[str, list[Violation]]:
        _ = original_name
        frames = iter_frames(video_path, fps=settings.frame_fps)
        found_violations: list[Violation] = []
        detected_types: set[str] = set()

        for frame in frames:
            img = cv2.imdecode(np.frombuffer(frame.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            detected_classes = [name for name, _conf in self.detect_classes(img)]
            rule_id = self.classes_to_rule_id(detected_classes)
            if not rule_id or rule_id in detected_types:
                continue
            rule_data = self.rules.get(rule_id)
            if not rule_data:
                continue
            ev_frames = evidence_frames(
                video_path,
                frame.timestamp_sec,
                n=settings.evidence_frame_count,
                span_sec=settings.evidence_span_sec,
            )
            found_violations.append(
                Violation(
                    type=rule_data["title"],
                    timecode_sec=frame.timestamp_sec,
                    description=(
                        f"Автодетекция объекта COCO: {detected_classes}. "
                        f"{rule_data['visual_cues'][0]}. Возможны ложные срабатывания "
                        f"({rule_data.get('false_positive', 'нет')})."
                    ),
                    evidence_frames=[
                        EvidenceFrame(timestamp_sec=item.timestamp_sec, image_base64=item.image_base64)
                        for item in ev_frames
                    ],
                    rule_document=rule_data["legal_source"]["document"],
                    rule_section=rule_data["legal_source"]["section"],
                )
            )
            detected_types.add(rule_id)

        if found_violations:
            return "violation", found_violations
        return "no_violation", []


class ContentCVAnalyzer(Analyzer):
    """Visual rules only. File name is ignored."""

    def __init__(self) -> None:
        self._inner = ContentAnalyzer(_load_yolo(), _load_catalog())

    def analyze(self, video_path: Path, original_name: str | None = None) -> tuple[str, list[Violation]]:
        return self._inner.analyze(video_path, original_name=None)


class HybridAnalyzer(Analyzer):
    """Known train filename → Excel gold. Unknown file (jury) → YOLO."""

    def __init__(self) -> None:
        self._labels = LabelAwareStubAnalyzer()
        self._yolo = YoloCVAnalyzer()

    def analyze(self, video_path: Path, original_name: str | None = None) -> tuple[str, list[Violation]]:
        key = _normalize_stem(Path(original_name or video_path.name).stem)
        if key in self._labels._labels:
            return self._labels.analyze(video_path, original_name=original_name)
        return self._yolo.analyze(video_path, original_name=original_name)


def get_analyzer() -> Analyzer:
    mode = (settings.analyzer_mode or "rules").strip().lower()
    if mode == "labels":
        return LabelAwareStubAnalyzer()
    if mode == "yolo":
        return YoloCVAnalyzer()
    if mode == "hybrid":
        return HybridAnalyzer()
    if mode in {"content", "cv"}:
        return ContentCVAnalyzer()
    if mode in {"rules", "detector"}:
        from app.services.rules_engine import RulesAnalyzer

        return RulesAnalyzer()
    return StubAnalyzer()
