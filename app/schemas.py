from typing import Literal

from pydantic import BaseModel, Field


JobStatus = Literal["processing", "done", "failed"]
Verdict = Literal["no_violation", "violation"]


class EvidenceFrame(BaseModel):
    timestamp_sec: float
    image_base64: str


class Violation(BaseModel):
    type: str
    timecode_sec: float
    description: str
    evidence_frames: list[EvidenceFrame] = Field(default_factory=list)
    rule_document: str | None = None
    rule_section: str | None = None


class AnalyzeAccepted(BaseModel):
    job_id: str
    status: JobStatus = "processing"


class JobResult(BaseModel):
    job_id: str
    status: JobStatus
    verdict: Verdict | None = None
    violations: list[Violation] = Field(default_factory=list)
    error: str | None = None
