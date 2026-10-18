"""Smoke Qwen3.5-4B (official VLM, Apache-2.0) on 3 train clips. Gold after, not in prompt."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.video import frame_at, video_meta  # noqa: E402

MODEL_DIR = ROOT / "models" / "Qwen3.5-4B"
N_FRAMES = 4
MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 16 * 28 * 28

SAMPLES = [
    ("X-1", "V-18 pocket / cash to self"),
    ("X-15", "V-04 cashier leaves"),
    ("X-19", "NO_VIOLATION"),
]

PROMPT = """Ты проверяешь видео кассы банка по кадрам сверху. Звук и банковская система недоступны.
Каждый кадр помечен временем от начала ролика.

Ответь ТОЛЬКО JSON без markdown:
{"verdict":"violation"|"no_violation","violations":[{"type":"кратко","timecode_sec":0.0,"description":"что видно на кадре и почему это может быть нарушением"}]}

Правила:
- Не утверждай проводки, сверку и документы, которых не видно.
- Если сомневаешься — verdict no_violation и пустой violations.
- Ложные срабатывания хуже для банка, чем осторожный пропуск сомнительного.
- timecode_sec бери из подписи ближайшего кадра, где видно событие.
"""


def _normalize_stem(stem: str) -> str:
    return stem.translate(str.maketrans("Хх", "Xx")).upper()


def find_video(stem: str) -> Path:
    wanted = _normalize_stem(stem)
    for path in Path(settings.video_dir).iterdir():
        if path.is_file() and _normalize_stem(path.stem) == wanted:
            return path
    raise FileNotFoundError(stem)


def sample_pils(video_path: Path, n: int) -> list[tuple[float, Image.Image]]:
    duration, _, _ = video_meta(video_path)
    if duration <= 0:
        duration = 10.0
    times = [duration * i / max(n, 1) for i in range(n)]
    times = [min(t, max(0.0, duration - 0.15)) for t in times]
    out = []
    for ts in times:
        frame = frame_at(video_path, ts)
        bgr = cv2.imdecode(np.frombuffer(frame.jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out.append((frame.timestamp_sec, Image.fromarray(rgb)))
    return out


def parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    return {"raw": text}


def load_model():
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

    print("Loading", MODEL_DIR)
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        quantization_config=quant,
        device_map="cuda",
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.eval()
    return processor, model


def infer(processor, model, frames: list[tuple[float, Image.Image]]) -> str:
    lines = [f"Кадр {i + 1}: {ts:.1f} с" for i, (ts, _) in enumerate(frames)]
    content = [{"type": "image", "image": img} for _, img in frames]
    content.append({"type": "text", "text": PROMPT + "\n" + "\n".join(lines)})
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
        out = model.generate(**batch, max_new_tokens=160, do_sample=False)
    trimmed = out[:, batch.input_ids.shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    print(
        "GPU",
        torch.cuda.get_device_name(0),
        "VRAM",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
        "torch",
        torch.__version__,
    )
    processor, model = load_model()
    print("VRAM after load, GB", round(torch.cuda.memory_allocated() / 1024**3, 2))
    labels = json.loads(settings.labels_path.read_text(encoding="utf-8"))["videos"]
    report = []

    for stem, note in SAMPLES:
        path = find_video(stem)
        gold = labels.get(_normalize_stem(stem), {})
        print("\n" + "=" * 60)
        print(stem, note, path.name)
        started = time.perf_counter()
        frames = sample_pils(path, N_FRAMES)
        print("frames", [(round(ts, 1), im.size) for ts, im in frames])
        raw = infer(processor, model, frames)
        elapsed = time.perf_counter() - started
        print("wall_sec", round(elapsed, 1))
        print("RAW:", raw)
        try:
            parsed = parse_json(raw)
            print("PARSED:", json.dumps(parsed, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            parsed = None
            print("JSON parse failed")
        print("GOLD:", json.dumps(gold, ensure_ascii=False))
        report.append(
            {
                "stem": stem,
                "note": note,
                "file": path.name,
                "wall_sec": round(elapsed, 1),
                "parsed": parsed,
                "gold": gold,
            }
        )
        torch.cuda.empty_cache()

    out_path = ROOT / "reports" / "smoke_qwen35.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out_path)


if __name__ == "__main__":
    main()
