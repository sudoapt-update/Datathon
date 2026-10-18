from __future__ import annotations

import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.auth import require_api_key
from app.config import ROOT_DIR, settings
from app.schemas import AnalyzeAccepted, JobResult
from app.services.analyzer import get_analyzer
from app.services.job_store import job_store

STATIC_DIR = ROOT_DIR / "app" / "static"
_JOB_LOCK = threading.Lock()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Models stay in memory for the whole process. Jobs take the lock one by one.
    if settings.analyzer_mode in {"rules", "detector", "content", "yolo", "hybrid"}:
        get_analyzer()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "local"}


@app.post("/api/v1/analyze", response_model=AnalyzeAccepted, dependencies=[Depends(require_api_key)])
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> AnalyzeAccepted:
    filename = file.filename or "upload.mov"
    suffix = Path(filename).suffix.lower()
    if suffix != ".mov":
        raise HTTPException(status_code=400, detail="Only .mov files are accepted")

    contents = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB")

    job = job_store.create()
    dest = settings.data_dir / "uploads" / f"{job.job_id}_{uuid.uuid4().hex[:6]}{suffix}"
    dest.write_bytes(contents)
    background_tasks.add_task(_run_job, job.job_id, dest, filename)
    return AnalyzeAccepted(job_id=job.job_id, status="processing")


@app.get("/api/v1/jobs/{job_id}", response_model=JobResult, dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> JobResult:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _run_job(job_id: str, video_path: Path, original_name: str) -> None:
    try:
        with _JOB_LOCK:
            analyzer = get_analyzer()
            verdict, violations = analyzer.analyze(video_path, original_name=original_name)
        job_store.complete(job_id, verdict, violations)
    except Exception as exc:  # noqa: BLE001 — surface any pipeline crash as failed job
        job_store.set_status(job_id, "failed", error=str(exc))
    finally:
        if video_path.exists():
            try:
                video_path.unlink()
            except OSError:
                pass
        shutil.rmtree(settings.data_dir / "jobs" / job_id, ignore_errors=True)
