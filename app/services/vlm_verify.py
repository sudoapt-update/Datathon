"""Qwen checks a short window around a rule hit. It cannot invent a violation type.

Wording and the legal reference stay in catalog.json. If the model is not
ready or the time budget runs out, the rule hit is kept.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.services.rules_engine import Hit
from app.services.video import frame_at

# These rules are stable on the train set; the VLM only rechecks the rest.
STRONG_RULES = {
    "cashier_leaves_during_service",
    "unauthorized_person_in_cash_desk",
    "personal_items_in_cash_desk",
    "recount_not_in_client_presence",
    "cash_left_unsecured",
    "cash_in_pocket",
    "no_client_identification",
    "no_authenticity_check",
}

_PROCESSOR = None
_MODEL = None

PROMPT = """Кадры кассы банка по времени. Детектор подозревает событие, но часто ошибается.
По умолчанию отвечай {{"confirm": false}}.

Событие: {title}
Подтверждай только если видно именно это: {cue}
Не подтверждать, если: {false_positive}

Ответь ТОЛЬКО JSON: {{"confirm": true}} или {{"confirm": false}}
confirm=true только когда действие видно ясно. Если сомневаешься — false.
Не добавляй других типов и не цитируй законы.
"""


def _load():
    global _PROCESSOR, _MODEL
    if _MODEL is not None:
        return _PROCESSOR, _MODEL
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    model_dir = Path(settings.vlm_model_dir)
    processor = AutoProcessor.from_pretrained(
        str(model_dir),
        min_pixels=16 * 28 * 28,
        max_pixels=256 * 28 * 28,
        trust_remote_code=True,
    )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir),
        quantization_config=quant,
        device_map="cuda",
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.eval()
    _PROCESSOR = processor
    _MODEL = model
    return processor, model


def _frames(video_path: Path, timestamp: float) -> list[tuple[float, Image.Image]]:
    out = []
    for offset in (-1.5, 0.0, 1.5):
        frame = frame_at(video_path, max(0.0, timestamp + offset))
        bgr = cv2.imdecode(np.frombuffer(frame.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out.append((frame.timestamp_sec, Image.fromarray(rgb)))
    return out


def _asks_confirm(processor, model, frames, rule: dict) -> bool | None:
    import torch

    cue = (rule.get("visual_cues") or [""])[0]
    text = PROMPT.format(
        title=rule.get("title", ""),
        cue=cue,
        false_positive=rule.get("false_positive", ""),
    )
    content = [{"type": "image", "image": img} for _, img in frames]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    batch = batch.to(model.device)
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=40, do_sample=False)
    trimmed = out[:, batch.input_ids.shape[1] :]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if "confirm" not in payload:
        return None
    return bool(payload["confirm"])


def confirm_hits(video_path: Path, hits: list[Hit], rules: dict) -> list[Hit]:
    """Drop a weak hit only when the VLM clearly rejects it inside the budget."""
    strong = [hit for hit in hits if hit.rule_id in STRONG_RULES]
    weak = [hit for hit in hits if hit.rule_id not in STRONG_RULES]
    if not weak:
        return strong
    deadline = time.monotonic() + settings.vlm_budget_sec
    try:
        processor, model = _load()
    except Exception:
        return hits
    kept = list(strong)
    for hit in weak[:3]:
        if time.monotonic() > deadline:
            kept.append(hit)
            continue
        rule = rules.get(hit.rule_id) or {}
        try:
            frames = _frames(video_path, hit.t)
            if len(frames) < 2:
                kept.append(hit)
                continue
            answer = _asks_confirm(processor, model, frames, rule)
        except Exception:
            kept.append(hit)
            continue
        if answer is False:
            continue
        kept.append(hit)
    return kept
