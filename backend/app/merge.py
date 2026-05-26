from __future__ import annotations

from .diarize.base import SpeakerTurn
from .engines.base import Segment


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _best_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """The speaker whose turn overlaps [start, end] the most, or None if no turn
    overlaps (a diarization gap — honest to leave unlabeled)."""
    best: str | None = None
    best_overlap = 0.0
    for turn in turns:
        ov = _overlap(start, end, turn.start, turn.end)
        if ov > best_overlap:
            best_overlap = ov
            best = turn.speaker
    return best


def merge(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    """Layer speaker labels onto a transcript. Pure: mutates and returns the
    given segments, does no I/O.

    Each word gets the speaker whose turn it overlaps most — so a speaker change
    mid-segment is captured at word granularity. Each segment gets the majority
    speaker among its words (falling back to its own time-range overlap when it
    has no words).
    """
    for seg in segments:
        counts: dict[str, int] = {}
        for word in seg.words:
            speaker = _best_speaker(word.start, word.end, turns)
            word.speaker = speaker
            if speaker is not None:
                counts[speaker] = counts.get(speaker, 0) + 1
        if counts:
            seg.speaker = max(counts, key=counts.__getitem__)
        else:
            seg.speaker = _best_speaker(seg.start, seg.end, turns)
    return segments
