"""Offline smoke test: Qwen2.5-VL-7B-AWQ on 3 train clips. Gold is printed after, not in the prompt."""

from __future__ import annotations

import importlib.machinery
import json
import re
import sys
import time
import types
from pathlib import Path

# autoawq imports `datasets` only for calibration; HF hub pins conflict on Windows.
_datasets = types.ModuleType("datasets")
_datasets.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
_datasets.__path__ = []
_datasets.load_dataset = lambda *args, **kwargs: None  # noqa: ARG005
sys.modules["datasets"] = _datasets

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.services.video import frame_at, video_meta  # noqa: E402

MODEL_DIR = ROOT / "models" / "Qwen2.5-VL-7B-Instruct-AWQ"
N_FRAMES = 6
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
        rgb = cv2.cvtColor(cv2.imdecode(__import__("numpy").frombuffer(frame.jpeg_bytes, __import__("numpy").uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
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
    import torch.nn as nn
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
    import awq.modules.linear.gemm as awq_gemm
    from awq.modules.linear.gemm import WQLinear_GEMM

    print("Loading", MODEL_DIR, "triton", awq_gemm.TRITON_AVAILABLE, "awq_ext", awq_gemm.awq_ext is not None)
    processor = AutoProcessor.from_pretrained(
        str(MODEL_DIR),
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
        use_fast=False,
    )
    config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    qc = dict(config.quantization_config)
    skip = list(qc.get("modules_to_not_convert") or [])
    if "lm_head" not in skip:
        skip.append("lm_head")
    qc["modules_to_not_convert"] = skip
    config.quantization_config = qc

    # Keep packed qweight/qzeros as int32. torch_dtype=float16 was casting them
    # to fp16, which broke Triton bit-shifts and forced the naive dequant path.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        config=config,
        torch_dtype="auto",
        device_map="cuda",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    n_int, n_bad = 0, 0
    for mod in model.modules():
        if isinstance(mod, WQLinear_GEMM):
            if mod.qweight.dtype != torch.int32 or mod.qzeros.dtype != torch.int32:
                n_bad += 1
            else:
                n_int += 1
    print("AWQ layers int32", n_int, "bad_dtype", n_bad)

    # Triton GEMM needs activations and dequant scales in the same dtype.
    # Checkpoint stores bf16 floats; qweight must stay int32.
    for name, param in model.named_parameters():
        if param.dtype in (torch.bfloat16, torch.float32):
            param.data = param.data.to(torch.float16)
    for name, buf in model.named_buffers():
        if buf.dtype in (torch.bfloat16, torch.float32):
            buf.data = buf.data.to(torch.float16)
    if isinstance(model.lm_head, WQLinear_GEMM):
        weight = None
        for shard in sorted(MODEL_DIR.glob("*.safetensors")):
            tensors = load_file(str(shard), device="cpu")
            if "lm_head.weight" in tensors:
                weight = tensors["lm_head.weight"]
                break
        if weight is None:
            raise RuntimeError("lm_head.weight missing in AWQ shards")
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        linear.weight = nn.Parameter(weight.to(dtype=torch.float16))
        model.lm_head = linear.to(device=model.device, dtype=torch.float16)
        print("Restored fp16 lm_head", tuple(weight.shape))
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    model.eval()
    return processor, model


def infer(processor, model, frames: list[tuple[float, Image.Image]]) -> str:
    from qwen_vl_utils import process_vision_info

    lines = [f"Кадр {i + 1}: {ts:.1f} с" for i, (ts, _) in enumerate(frames)]
    content = [{"type": "image", "image": img} for _, img in frames]
    content.append({"type": "text", "text": PROMPT + "\n" + "\n".join(lines)})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    kwargs = dict(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    if video_inputs:
        kwargs["videos"] = video_inputs
    batch = processor(**kwargs)
    batch = batch.to(model.device)
    with torch.inference_mode():
        out = model.generate(**batch, max_new_tokens=250, do_sample=False, temperature=None, top_p=None)
    trimmed = out[:, batch.input_ids.shape[1] :]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    print("GPU", torch.cuda.get_device_name(0), "VRAM", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
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

    out_path = ROOT / "reports" / "smoke_vlm.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out_path)


if __name__ == "__main__":
    main()
