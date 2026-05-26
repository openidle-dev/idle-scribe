from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from fastapi.responses import Response

from . import db, export
from .audio import ffmpeg_available
from .config import Settings, get_settings
from .engines import KNOWN_ENGINES, EngineProvider, available_engines
from .models import Job, JobStatus, utcnow
from .worker import JobWorker

logging.basicConfig(level=logging.INFO)

_UPLOAD_CHUNK = 1024 * 1024  # 1 MiB


class JobOut(BaseModel):
    id: str
    original_filename: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    engine: str | None
    language: str | None
    normalized_path: str | None
    transcript_path: str | None
    error: str | None

    @classmethod
    def from_job(cls, job: Job) -> "JobOut":
        return cls(
            id=job.id,
            original_filename=job.original_filename,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            engine=job.engine,
            language=job.language,
            normalized_path=job.normalized_path,
            transcript_path=job.transcript_path,
            error=job.error,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    settings.ensure_dirs()
    db.init_db(settings.db_path)
    if not ffmpeg_available():
        logging.getLogger("idle_scribe").warning(
            "ffmpeg not found on PATH — uploads will fail to normalize."
        )
    engines = EngineProvider(settings)
    diarizer = None
    if settings.enable_diarization:
        from .diarize.pyannote_diarizer import PyannoteDiarizer

        diarizer = PyannoteDiarizer(
            model=settings.diarization_model,
            device=settings.diarization_device,
            token=settings.hf_token,
        )
    worker = JobWorker(settings, engines, diarizer)
    worker.start()
    app.state.settings = settings
    app.state.worker = worker
    try:
        yield
    finally:
        await worker.stop()


app = FastAPI(title="idle-scribe", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    settings: Settings = app.state.settings
    return {
        "status": "ok",
        "ffmpeg": ffmpeg_available(),
        "default_engine": settings.transcription_engine,
        "available_engines": available_engines(settings),
        "diarization": settings.enable_diarization,
    }


@app.post("/jobs", response_model=JobOut, status_code=202)
async def create_job(
    file: UploadFile,
    engine: str | None = Form(default=None),
    language: str | None = Form(default=None),
) -> JobOut:
    settings: Settings = app.state.settings
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if engine is not None:
        if engine not in KNOWN_ENGINES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown engine {engine!r}. Known: {list(KNOWN_ENGINES)}.",
            )
        if engine not in available_engines(settings):
            raise HTTPException(
                status_code=400,
                detail=f"Engine {engine!r} is not configured (missing API key).",
            )

    job_id = uuid4().hex
    suffix = Path(file.filename).suffix
    dst = settings.uploads_dir / f"{job_id}{suffix}"

    size = 0
    try:
        with dst.open("wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Upload too large.")
                out.write(chunk)
    except HTTPException:
        dst.unlink(missing_ok=True)
        raise

    if size == 0:
        dst.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload.")

    now = utcnow()
    job = Job(
        id=job_id,
        original_filename=file.filename,
        status=JobStatus.PENDING,
        created_at=now,
        updated_at=now,
        original_path=str(dst),
        engine=engine or settings.transcription_engine,
        language=language or None,
    )
    db.create_job(settings.db_path, job)
    app.state.worker.enqueue(job_id)
    return JobOut.from_job(job)


@app.get("/jobs", response_model=list[JobOut])
async def list_jobs() -> list[JobOut]:
    settings: Settings = app.state.settings
    return [JobOut.from_job(j) for j in db.list_jobs(settings.db_path)]


@app.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(job_id: str) -> JobOut:
    settings: Settings = app.state.settings
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobOut.from_job(job)


def _latest_transcript_path(job: Job) -> str | None:
    """Prefer the user-edited transcript; fall back to the original ASR output."""
    return job.edited_transcript_path or job.transcript_path


def _load_transcript_or_409(job: Job) -> dict:
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=409, detail=f"Job failed: {job.error}")
    path = _latest_transcript_path(job)
    if path is None:
        raise HTTPException(
            status_code=409, detail=f"Transcript not ready (status: {job.status.value})."
        )
    return json.loads(Path(path).read_text(encoding="utf-8"))


@app.get("/jobs/{job_id}/transcript")
async def get_transcript(job_id: str) -> JSONResponse:
    settings: Settings = app.state.settings
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(content=_load_transcript_or_409(job))


class WordIn(BaseModel):
    start: float
    end: float
    word: str
    probability: float | None = None
    speaker: str | None = None


class SegmentIn(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[WordIn] = []


class TranscriptIn(BaseModel):
    language: str | None = None
    language_probability: float | None = None
    duration: float | None = None
    speakers: list[str] = []
    segments: list[SegmentIn]


@app.put("/jobs/{job_id}/transcript")
async def save_transcript(job_id: str, payload: TranscriptIn) -> dict:
    settings: Settings = app.state.settings
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.transcript_path is None:
        raise HTTPException(status_code=409, detail="Nothing transcribed to edit yet.")

    # Always diff against the immutable original ASR output, so corrections are
    # (ASR baseline -> human-approved text) training pairs.
    original = json.loads(Path(job.transcript_path).read_text(encoding="utf-8"))
    orig_segments = original.get("segments", [])
    edited = payload.model_dump()

    corrections: list[dict] = []
    for i, seg in enumerate(edited["segments"]):
        if i >= len(orig_segments):
            break
        orig = orig_segments[i]
        if seg["text"].strip() != orig.get("text", "").strip():
            corrections.append(
                {
                    "segment_start": orig["start"],
                    "segment_end": orig["end"],
                    "original_text": orig.get("text", ""),
                    "corrected_text": seg["text"],
                    "speaker": seg.get("speaker"),
                }
            )

    edited_path = settings.transcripts_dir / f"{job_id}.edited.json"
    edited_path.write_text(json.dumps(edited, ensure_ascii=False), encoding="utf-8")
    db.update_job(settings.db_path, job_id, edited_transcript_path=str(edited_path))
    n = db.replace_corrections(settings.db_path, job_id, corrections)
    return {"saved": True, "corrections_recorded": n}


@app.get("/jobs/{job_id}/export")
async def export_transcript(job_id: str, format: str = "txt") -> Response:
    settings: Settings = app.state.settings
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    data = _load_transcript_or_409(job)
    try:
        content, media_type = export.render(data, format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    filename = f"{job_id}.{format}"
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/jobs/{job_id}/audio")
async def get_audio(job_id: str) -> FileResponse:
    settings: Settings = app.state.settings
    job = db.get_job(settings.db_path, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if not job.normalized_path or not Path(job.normalized_path).exists():
        raise HTTPException(status_code=409, detail="Audio not available yet.")
    return FileResponse(job.normalized_path, media_type="audio/wav")
