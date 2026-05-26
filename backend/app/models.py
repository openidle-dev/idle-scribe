from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime, timezone


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    NORMALIZING = "normalizing"
    # Stages below are wired in later milestones (transcription, diarization, merge).
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"


TERMINAL_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.FAILED})


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Job:
    id: str
    original_filename: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    original_path: str
    engine: str | None = None
    language: str | None = None  # requested hint; None = autodetect
    normalized_path: str | None = None
    transcript_path: str | None = None  # original ASR output (immutable)
    edited_transcript_path: str | None = None  # user-corrected version
    error: str | None = None
