"""Silence-gap speaker heuristic and dialogue rendering."""

from __future__ import annotations

from src.config import Segment
from src.diarize import assign_speakers, to_dialogue


def test_assign_speakers_switches_on_gap():
    segments = [
        Segment(start=0.0, end=2.0, text="Hi how are you"),
        Segment(start=2.1, end=4.0, text="I am fine thanks"),   # small gap -> same speaker
        Segment(start=6.0, end=8.0, text="Glad to hear it"),    # big gap -> new speaker
    ]
    labelled = assign_speakers(segments, gap_threshold=1.2)
    assert labelled[0].speaker == "Speaker 1"
    assert labelled[1].speaker == "Speaker 1"
    assert labelled[2].speaker == "Speaker 2"


def test_assign_speakers_cycles_between_two():
    # Every segment separated by a big gap -> alternate 1,2,1,2.
    segments = [Segment(start=i * 10.0, end=i * 10.0 + 2.0, text=f"turn {i}") for i in range(4)]
    labelled = assign_speakers(segments, gap_threshold=1.0, max_speakers=2)
    assert [s.speaker for s in labelled] == ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2"]


def test_to_dialogue_merges_consecutive_same_speaker():
    segments = [
        Segment(start=0.0, end=2.0, text="Hello", speaker="Speaker 1"),
        Segment(start=2.0, end=4.0, text="again", speaker="Speaker 1"),
        Segment(start=4.5, end=6.0, text="Hi there", speaker="Speaker 2"),
    ]
    dialogue = to_dialogue(segments)
    lines = dialogue.split("\n\n")
    assert lines[0] == "Speaker 1: Hello again"
    assert lines[1] == "Speaker 2: Hi there"


def test_to_dialogue_defaults_missing_speaker():
    segments = [Segment(start=0.0, end=1.0, text="unlabelled")]
    assert to_dialogue(segments) == "Speaker 1: unlabelled"
