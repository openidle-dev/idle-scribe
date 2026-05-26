from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float | None = None
    speaker: str | None = None  # filled by merge() in the diarization stage

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "word": self.word,
            "probability": self.probability,
            "speaker": self.speaker,
        }


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None  # majority speaker of the segment's words

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class TranscriptionResult:
    language: str | None
    language_probability: float | None
    duration: float | None
    segments: list[Segment] = field(default_factory=list)
    # True when the engine already assigned speakers (e.g. ElevenLabs), so the
    # pipeline can skip the separate diarization stage.
    diarized: bool = False

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "segments": [s.to_dict() for s in self.segments],
        }


@runtime_checkable
class TranscriptionEngine(Protocol):
    """A transcription engine turns a normalized WAV into timestamped segments.

    Speaker labels are layered on separately (see diarize + merge), so engines
    only ever produce text + word timestamps. This is the seam that keeps the
    local model and the OpenAI API interchangeable.
    """

    def transcribe(
        self, wav_path: Path, language: str | None = None
    ) -> TranscriptionResult: ...
