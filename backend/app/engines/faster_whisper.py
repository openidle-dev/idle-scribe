from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..cuda_setup import register_cuda_dll_dirs
from .base import Segment, TranscriptionResult, Word

logger = logging.getLogger("idle_scribe.engine.faster_whisper")


class FasterWhisperEngine:
    """Local CTranslate2 Whisper engine.

    `model` is either a named size ("large-v3", "tiny", ...) or a path to a
    converted CTranslate2 model directory — the latter is how the Afrikaans
    fine-tune (or a future in-house model) gets dropped in.

    The model is loaded lazily and once: construction is cheap, the first
    transcribe() pays the load cost. Loads are guarded by a lock because the
    worker calls transcribe() from a thread.
    """

    def __init__(
        self,
        model: str = "large-v3",
        *,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        download_root: str | None = None,
        language: str | None = None,
        beam_size: int = 5,
    ) -> None:
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._download_root = download_root
        self._language = language
        self._beam_size = beam_size
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    register_cuda_dll_dirs()
                    from faster_whisper import WhisperModel

                    logger.info(
                        "Loading Whisper model=%s device=%s compute_type=%s",
                        self._model_name,
                        self._device,
                        self._compute_type,
                    )
                    self._model = WhisperModel(
                        self._model_name,
                        device=self._device,
                        compute_type=self._compute_type,
                        download_root=self._download_root,
                    )
        return self._model

    def transcribe(
        self, wav_path: Path, language: str | None = None
    ) -> TranscriptionResult:
        model = self._ensure_model()
        # VAD filtering + no conditioning on prior text mitigates large-v3's
        # tendency to hallucinate on silence/music (PLAN.md §6).
        segments_iter, info = model.transcribe(
            str(wav_path),
            language=language or self._language,
            beam_size=self._beam_size,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
        )

        segments: list[Segment] = []
        for seg in segments_iter:  # generator — transcription happens here
            words = [
                Word(start=w.start, end=w.end, word=w.word, probability=w.probability)
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(start=seg.start, end=seg.end, text=seg.text, words=words)
            )

        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
        )
