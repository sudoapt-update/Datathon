# Фаза 2 — бриф (можно делать параллельно с фазой 1)

Цель: видео → кадры с таймкодом → JPEG/base64. **Без моделей и без угадывания типов нарушений.**

Фаза 1 ещё не готова: каталог типов и акты РК вам не нужны. Детектор (фаза 3) потом вызовет ваши функции.

## Контракт, который надо выдать

Новый модуль `app/services/video.py` (имя можно другое, сигнатуры лучше не ломать):

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Frame:
    timestamp_sec: float   # секунды от начала ролика
    jpeg_bytes: bytes      # JPEG, не PNG
    image_base64: str      # тот же JPEG в base64 без data:-префикса

def iter_frames(video_path: Path, fps: float = 2.0) -> list[Frame]:
    """Равномерная нарезка. Для ролика 145 с при 2 fps ≈ 290 кадров."""

def frame_at(video_path: Path, timestamp_sec: float) -> Frame:
    """Ближайший кадр к таймкоду. Нужен для evidence и для отладки UI."""

def evidence_frames(video_path: Path, timestamp_sec: float, n: int = 3, span_sec: float = 1.0) -> list[Frame]:
    """1–3 кадра вокруг события: t, и при n>1 ещё t±span. Как в API жюри."""
```

Ограничения:

- Вход только `.mov`, звук игнорировать.
- Таймкод должен совпадать с тем, что потом уйдёт в `timecode_sec` / `timestamp_sec`.
- JPEG разумного размера (ширина ~640–960, quality 70–85), чтобы 3 кадра в JSON не раздували ответ.
- На самом длинном ролике (~145 с, `X-14` / `X-47` около 14–28 МБ) уложиться **с запасом** в 5 минут. Цель фазы 2: нарезка **десятки секунд**, не минуты.
- Без сети, без облачных API, лица не детектить и не кропать «для распознавания».
- Временные файлы писать в `data/jobs/{job_id}/` и чистить. Сейчас `_run_job` в `app/main.py` уже удаляет upload и `data/jobs/{job_id}` после анализа.

## Что подключить в сервис

1. `opencv-python-headless` (или вызов **уже установленного FFmpeg**). OpenCV часто лучше читает `.mov`, чем Pillow.
2. Добавить зависимость в `requirements.txt`.
3. Настройки в `app/config.py`: `frame_fps: float = 2.0`, `evidence_jpeg_quality: int = 80`, `evidence_max_width: int = 960`.

## Как проверить, что фаза 2 готова

Скрипт `scripts/extract_sample.py`:

- Берёт 1 ролик из `VIDEO_DIR` (`.env`: `C:/Users/User/Downloads/train_videos/videos`).
- Печатает число кадров, первый/последний `timestamp_sec`, длительность.
- Сохраняет 3 JPEG в `data/jobs/_phase2_sample/` для `X-1` на **22-й секунде** (таймкод из `data/labels.json`).
- Замеряет wall time на длинном ролике (`X-14` или `X-47`).

Плюс ручная проверка UI:

- В `get_analyzer()` временно вернуть `LabelAwareStubAnalyzer`.
- **Заменить** `PLACEHOLDER_JPEG_B64` на реальный кадр через `evidence_frames(video_path, ts, n=1)`.
- Загрузить `X-1.mov` в UI: вердикт «есть признаки», картинка с кассы, не белый квадрат 1×1.
- Для ролика без нарушения (`X-3`) — пустой список и `no_violation`.
- Перед сдачей жюри этот analyzer **выключить**. Имена файлов у жюри другие, lookup по `labels.json` на тесте бесполезен и опасен.

Нормализация имени уже есть: `_normalize_stem` в `app/services/analyzer.py` (кириллица `Х-3` → `X-3`). Upload в API сейчас называется `{job_id}_{rand}.mov`, **не** `X-1.mov`, поэтому lookup по stem upload-файла не сработает. Для отладки: либо гонять скрипт по `VIDEO_DIR`, либо прокидывать original filename в job (это мелкая правка `main.py`, можно сделать в фазе 2).

## Что не делать в этой фазе

- Не писать YOLO, не классифицировать типы, не трогать `app/rules/catalog.json`.
- Не менять JSON жюри: `verdict`, `violations[]`, `evidence_frames[].image_base64`.
- Не хранить все кадры всех роликов в git.
- Не блокироваться, пока фаза 1 не закончит типы: вам достаточно любого `timestamp_sec`.

## Стык с фазой 3

Следующий человек импортирует `iter_frames` / `frame_at` / `evidence_frames` и гоняет детектор по списку `Frame`. Если сигнатуры стабильные, им не придётся переписывать нарезку.
