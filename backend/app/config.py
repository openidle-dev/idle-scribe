from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="IDLE_SCRIBE_", env_file=".env", extra="ignore"
    )

    storage_dir: Path = _BACKEND_DIR / "storage"
    models_dir: Path = _BACKEND_DIR.parent / "models"

    # Audio normalization target (Whisper/pyannote expect 16 kHz mono).
    sample_rate: int = 16000
    channels: int = 1

    # Cap upload size to avoid filling the disk with a single request.
    max_upload_bytes: int = 2 * 1024 * 1024 * 1024  # 2 GiB

    # Transcription engine. `whisper_model` is a named size ("large-v3") or a
    # path to a converted CTranslate2 model (the Afrikaans fine-tune seam).
    transcription_engine: str = "faster_whisper"
    whisper_model: str = "large-v3"
    whisper_device: str = "cuda"
    # float16 fits large-v3 in ~4.7 GB (within the 8 GB RTX 2070) and is both
    # faster and more accurate than int8 on Turing. Drop to int8_float16 only if
    # transcription + diarization together exhaust VRAM (milestone 4).
    whisper_compute_type: str = "float16"
    whisper_language: str | None = None  # None = autodetect; set "af"/"en" to pin

    # OpenAI engine (fallback / comparison). Only whisper-* models return word
    # timestamps. Key falls back to the standard OPENAI_API_KEY env var.
    openai_model: str = "whisper-1"
    openai_api_key: str | None = None
    openai_language: str | None = None

    # ElevenLabs Scribe (cloud, strong on Afrikaans). Key falls back to
    # ELEVENLABS_API_KEY env var.
    elevenlabs_model: str = "scribe_v1"
    elevenlabs_api_key: str | None = None

    # Diarization (pyannote, CPU). Optional and degrades gracefully — a job still
    # completes (transcribe-only) if the model can't load. hf_token is optional;
    # None uses the token cached by `huggingface-cli login`.
    enable_diarization: bool = True
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_device: str = "cpu"
    hf_token: str | None = None

    # Frontend dev server origins allowed via CORS (comma-separated to override).
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def normalized_dir(self) -> Path:
        return self.storage_dir / "normalized"

    @property
    def transcripts_dir(self) -> Path:
        return self.storage_dir / "transcripts"

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "idle_scribe.db"

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
