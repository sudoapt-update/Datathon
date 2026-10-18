from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "ForteBank Cash Discipline"
    api_key: str = "datathon-jury-key-change-me"
    max_upload_mb: int = 40
    job_ttl_sec: int = 3600
    data_dir: Path = ROOT_DIR / "data"
    video_dir: Path = Path(r"C:\Users\User\Downloads\train_videos\videos")
    labels_path: Path = ROOT_DIR / "data" / "labels.json"
    local_only: bool = True
    # rules = detector + temporal rules. labels/hybrid = train-name lookup, not for jury.
    analyzer_mode: str = "rules"
    yolo_weights: Path = ROOT_DIR / "yolo11m.pt"
    rules_fps: float = 3.0
    vlm_enabled: bool = True
    vlm_model_dir: Path = ROOT_DIR / "models" / "Qwen3.5-4B"
    vlm_budget_sec: float = 90.0
    analyze_budget_sec: float = 240.0
    frame_fps: float = 2.0
    evidence_jpeg_quality: int = 80
    evidence_max_width: int = 960
    evidence_frame_count: int = 3
    evidence_span_sec: float = 1.0


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
(settings.data_dir / "uploads").mkdir(parents=True, exist_ok=True)
(settings.data_dir / "jobs").mkdir(parents=True, exist_ok=True)
