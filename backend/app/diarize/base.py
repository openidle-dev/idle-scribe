from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str  # e.g. "SPEAKER_00"


@runtime_checkable
class Diarizer(Protocol):
    """A diarizer turns a normalized WAV into speaker turns (who spoke when).

    It knows nothing about the words — speaker labels are layered onto the
    transcript afterwards by the pure merge() function. This is what keeps
    transcription and diarization independent (PLAN.md §2/§4).
    """

    def diarize(self, wav_path: Path) -> list[SpeakerTurn]: ...
