from __future__ import annotations

import logging
from pathlib import Path

from .base import Segment, TranscriptionResult, Word

logger = logging.getLogger("idle_scribe.engine.openai")

# OpenAI's transcription endpoint rejects files larger than 25 MB.
_MAX_BYTES = 25 * 1024 * 1024


class OpenAIEngine:
    """OpenAI transcription engine (zero-setup fallback / comparison).

    Only the Whisper models ("whisper-1") support verbose_json + word/segment
    timestamps. The gpt-4o-transcribe family produces better text but returns no
    timestamps, so diarization merge (milestone 4) degrades to no word-level
    speaker assignment on that path — surfaced here, not silently.
    """

    def __init__(
        self,
        model: str = "whisper-1",
        *,
        api_key: str | None = None,
        language: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._language = language
        self._client = None

    @property
    def supports_word_timestamps(self) -> bool:
        return self._model.startswith("whisper")

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            # OpenAI() also reads OPENAI_API_KEY from env if api_key is None.
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def transcribe(
        self, wav_path: Path, language: str | None = None
    ) -> TranscriptionResult:
        lang = language or self._language
        size = wav_path.stat().st_size
        if size > _MAX_BYTES:
            raise ValueError(
                f"{wav_path.name} is {size / 1e6:.1f} MB; OpenAI limit is 25 MB. "
                "Use the local engine or chunk the audio (milestone 6)."
            )

        client = self._ensure_client()
        with wav_path.open("rb") as fh:
            if self.supports_word_timestamps:
                resp = client.audio.transcriptions.create(
                    model=self._model,
                    file=fh,
                    response_format="verbose_json",
                    timestamp_granularities=["word", "segment"],
                    language=lang or None,
                )
                return self._from_verbose(resp)

            logger.warning(
                "Model %s returns no word timestamps; diarization merge will be "
                "segment-only.",
                self._model,
            )
            resp = client.audio.transcriptions.create(
                model=self._model, file=fh, language=self._language or None
            )
            return TranscriptionResult(
                language=None,
                language_probability=None,
                duration=None,
                segments=[Segment(start=0.0, end=0.0, text=resp.text, words=[])],
            )

    @staticmethod
    def _from_verbose(resp) -> TranscriptionResult:
        all_words = [
            Word(start=w.start, end=w.end, word=w.word) for w in (resp.words or [])
        ]
        segments: list[Segment] = []
        for seg in resp.segments or []:
            mid = lambda w: (w.start + w.end) / 2  # noqa: E731
            seg_words = [w for w in all_words if seg.start <= mid(w) <= seg.end]
            segments.append(
                Segment(start=seg.start, end=seg.end, text=seg.text, words=seg_words)
            )
        # Fall back to a single segment if the API returned words but no segments.
        if not segments and all_words:
            segments = [
                Segment(
                    start=all_words[0].start,
                    end=all_words[-1].end,
                    text=resp.text,
                    words=all_words,
                )
            ]
        return TranscriptionResult(
            language=getattr(resp, "language", None),
            language_probability=None,
            duration=getattr(resp, "duration", None),
            segments=segments,
        )
