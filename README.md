# Контроль кассовой дисциплины (ForteBank Datathon)

Локальный веб-сервис: загрузка видео работы кассира, анализ, вердикт и кадры-доказательства.

Режимы `ANALYZER_MODE` в `.env`:

- `content` — **по кадрам**, имя файла не используется (режим сдачи).
- `yolo` — наивный COCO (телефон/сумка на одном кадре).
- `labels` / `hybrid` — подстановка разметки по имени, только отладка UI.

Залить Excel тиммейта и сверить YOLO с gold:

```powershell
python scripts/import_labels.py
python scripts/eval_gold_vs_yolo.py
```

## Ограничения кейса

- Только локальные модели, без облачных LLM/VLM.
- Анализ строго визуальный, без звука и без идентификации лиц.
- Формат входа: `.mov`, до ~29 МБ, ответ жюри не дольше 5 минут.

## API жюри

| Метод | Путь | Назначение |
| --- | --- | --- |
| `POST` | `/api/v1/analyze` | поле `file`, заголовок `X-API-Key` → `{ job_id }` |
| `GET` | `/api/v1/jobs/{job_id}` | `processing` / `done` / `failed` и JSON результата |

Ключ по умолчанию: `datathon-jury-key-change-me` (см. `.env.example`).

## Запуск (когда установлен Python 3.12+)

```powershell
cd C:\Users\User\Documents\Datathon
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

UI: http://localhost:8000

Проверка нарезки:

```powershell
python scripts/extract_sample.py
```

Кадры появятся в `data/jobs/_phase2_sample/`.

Обучающие ролики лежат отдельно: `C:\Users\User\Downloads\train_videos`. В репозиторий их не копируем.

Очередность работ — в `TASKS.md`.
