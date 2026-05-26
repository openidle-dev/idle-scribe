from __future__ import annotations

import logging
from pathlib import Path

from .base import Segment, TranscriptionResult, Word

logger = logging.getLogger("idle_scribe.engine.elevenlabs")

# Close a segment after sentence-final punctuation or a pause longer than this.
_PAUSE_GAP = 1.0


class ElevenLabsEngine:
    """ElevenLabs Scribe transcription (cloud, pay-as-you-go).

    Strong multilingual accuracy including Afrikaans. Returns text + word
    timestamps; speaker labels are layered on separately by the diarization
    stage (consistent with the other engines), so we don't use Scribe's own
    diarization here.
    """

    def __init__(
        self,
        model: str = "scribe_v1",
        *,
        api_key: str | None = None,
        language: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._language = language
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from elevenlabs import ElevenLabs

            self._client = ElevenLabs(api_key=self._api_key)
        return self._client

    def transcribe(
        self, wav_path: Path, language: str | None = None
    ) -> TranscriptionResult:
        client = self._ensure_client()
        with wav_path.open("rb") as fh:
            resp = client.speech_to_text.convert(
                model_id=self._model,
                file=fh,
                language_code=(language or self._language) or None,
                timestamps_granularity="word",
                diarize=False,
                tag_audio_events=False,
            )
        return self._to_result(resp)

    @staticmethod
    def _to_result(resp) -> TranscriptionResult:
        words = [
            Word(start=w.start, end=w.end, word=w.text)
            for w in (resp.words or [])
            if getattr(w, "type", "word") == "word"
        ]
        return TranscriptionResult(
            language=getattr(resp, "language_code", None),
            language_probability=getattr(resp, "language_probability", None),
            duration=getattr(resp, "audio_duration_secs", None),
            segments=_group_into_segments(words),
        )


def _group_into_segments(words: list[Word]) -> list[Segment]:
    """Scribe returns a flat word list; group it into sentence-ish segments on
    sentence-final punctuation or pauses, so the editor shows readable chunks."""
    segments: list[Segment] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            segments.append(
                Segment(
                    start=current[0].start,
                    end=current[-1].end,
                    text=" ".join(w.word for w in current).strip(),
                    words=list(current),
                )
            )
            current.clear()

    for i, w in enumerate(words):
        current.append(w)
        ends_sentence = w.word.rstrip().endswith((".", "?", "!"))
        gap_next = (
            i + 1 < len(words) and (words[i + 1].start - w.end) > _PAUSE_GAP
        )
        if ends_sentence or gap_next:
            flush()
    flush()
    return segments
