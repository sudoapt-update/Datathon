from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone

from app.schemas import JobResult, JobStatus, Verdict, Violation


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobResult] = {}

    def create(self) -> JobResult:
        job_id = uuid.uuid4().hex[:8]
        job = JobResult(job_id=job_id, status="processing")
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> JobResult | None:
        with self._lock:
            return self._jobs.get(job_id)

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(update={"status": status, "error": error})

    def complete(self, job_id: str, verdict: Verdict, violations: list[Violation]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(
                update={
                    "status": "done",
                    "verdict": verdict,
                    "violations": violations,
                    "error": None,
                }
            )


job_store = JobStore()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
