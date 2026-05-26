from app.diarize.base import SpeakerTurn
from app.engines.base import Segment, Word
from app.merge import merge


def _w(start, end, text="x"):
    return Word(start=start, end=end, word=text)


def test_speaker_switch_mid_segment():
    # One Whisper segment spanning a speaker change at t=5.
    seg = Segment(
        start=0.0,
        end=8.0,
        text="a b c d",
        words=[_w(0.5, 1.0, "a"), _w(1.5, 2.0, "b"), _w(5.5, 6.0, "c"), _w(6.5, 7.0, "d")],
    )
    turns = [
        SpeakerTurn(0.0, 5.0, "SPEAKER_00"),
        SpeakerTurn(5.0, 8.0, "SPEAKER_01"),
    ]
    merge([seg], turns)
    assert [w.speaker for w in seg.words] == [
        "SPEAKER_00", "SPEAKER_00", "SPEAKER_01", "SPEAKER_01"
    ]
    # 2 vs 2 tie -> first-seen speaker wins (SPEAKER_00).
    assert seg.speaker == "SPEAKER_00"


def test_segment_majority():
    seg = Segment(0.0, 8.0, "a b c", words=[_w(0.5, 1.0), _w(1.5, 2.0), _w(6.0, 6.5)])
    turns = [SpeakerTurn(0.0, 5.0, "SPEAKER_00"), SpeakerTurn(5.0, 8.0, "SPEAKER_01")]
    merge([seg], turns)
    assert seg.speaker == "SPEAKER_00"  # 2 of 3 words


def test_no_overlap_leaves_word_unlabeled():
    seg = Segment(0.0, 2.0, "a", words=[_w(0.5, 1.0)])
    turns = [SpeakerTurn(10.0, 20.0, "SPEAKER_00")]  # no overlap
    merge([seg], turns)
    assert seg.words[0].speaker is None
    assert seg.speaker is None


def test_no_words_uses_segment_range():
    seg = Segment(1.0, 4.0, "untimed", words=[])
    turns = [SpeakerTurn(0.0, 5.0, "SPEAKER_00")]
    merge([seg], turns)
    assert seg.speaker == "SPEAKER_00"
