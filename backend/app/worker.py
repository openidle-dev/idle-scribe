from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from . import db
from .audio import normalize_to_wav
from .config import Settings
from .diarize.base import Diarizer
from .engines import EngineProvider
from .merge import merge
from .models import JobStatus

logger = logging.getLogger("idle_scribe.worker")


class JobWorker:
    """Single-consumer queue: jobs are processed one at a time.

    Serial processing is deliberate — later milestones run Whisper large-v3 and
    pyannote on an 8 GB GPU, where two concurrent jobs would exhaust VRAM.
    """

    def __init__(
        self,
        settings: Settings,
        engines: EngineProvider,
        diarizer: Diarizer | None = None,
    ) -> None:
        self._settings = settings
        self._engines = engines
        self._diarizer = diarizer
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="job-worker")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def enqueue(self, job_id: str) -> None:
        self._queue.put_nowait(job_id)

    async def _run(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._process(job_id)
            except Exception:  # noqa: BLE001 - worker must survive any job failure
                logger.exception("Unhandled error processing job %s", job_id)
            finally:
                self._queue.task_done()

    async def _process(self, job_id: str) -> None:
        db_path = self._settings.db_path
        job = db.get_job(db_path, job_id)
        if job is None:
            logger.warning("Job %s vanished before processing", job_id)
            return

        try:
            db.update_job(db_path, job_id, status=JobStatus.NORMALIZING)
            wav = self._settings.normalized_dir / f"{job_id}.wav"
            await normalize_to_wav(
                Path(job.original_path),
                wav,
                sample_rate=self._settings.sample_rate,
                channels=self._settings.channels,
            )
            db.update_job(
                db_path, job_id, status=JobStatus.TRANSCRIBING, normalized_path=str(wav)
            )

            # transcribe() is blocking (and GPU-bound); keep it off the event loop.
            engine = self._engines.get(job.engine)
            result = await asyncio.to_thread(engine.transcribe, wav, job.language)

            # Skip pyannote when the engine already diarized (e.g. ElevenLabs):
            # no point redoing speaker assignment, and it's the slow CPU stage.
            # Diarization is optional and must never fail the job: on any error we
            # log and ship a transcribe-only transcript (PLAN.md §6).
            if self._diarizer is not None and not result.diarized:
                try:
                    db.update_job(db_path, job_id, status=JobStatus.DIARIZING)
                    turns = await asyncio.to_thread(self._diarizer.diarize, wav)
                    merge(result.segments, turns)
                except Exception:
                    logger.exception(
                        "Diarization failed for %s; completing transcribe-only", job_id
                    )

            data = result.to_dict()
            data["speakers"] = sorted(
                {s.speaker for s in result.segments if s.speaker is not None}
            )
            transcript_path = self._settings.transcripts_dir / f"{job_id}.json"
            transcript_path.write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            db.update_job(
                db_path,
                job_id,
                status=JobStatus.COMPLETED,
                transcript_path=str(transcript_path),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Job %s failed", job_id)
            db.update_job(db_path, job_id, status=JobStatus.FAILED, error=str(exc))
