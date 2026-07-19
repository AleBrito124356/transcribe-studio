"""SRT/VTT formatting, wrapping, and reading-speed timing math."""

from __future__ import annotations

import re

import pytest

from src.config import Segment, format_srt_timestamp, format_vtt_timestamp
from src.subtitles import (
    build_cues,
    to_srt,
    to_vtt,
    translate_segments,
    wrap_text,
)


def test_srt_timestamp_format():
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_srt_timestamp(1.5) == "00:00:01,500"
    assert format_srt_timestamp(61.25) == "00:01:01,250"
    assert format_srt_timestamp(3661.007) == "01:01:01,007"


def test_vtt_timestamp_uses_dot():
    assert format_vtt_timestamp(61.25) == "00:01:01.250"
    assert "." in format_vtt_timestamp(1.0)
    assert "," not in format_vtt_timestamp(1.0)


def test_timestamp_rounds_millis_without_overflow():
    # 0.9999 s rounds to 1000 ms -> must roll into the seconds field, not show 1000.
    assert format_srt_timestamp(0.9999) == "00:00:01,000"


def test_wrap_text_respects_max_chars():
    lines = wrap_text("the quick brown fox jumps over the lazy dog", max_chars=15)
    assert all(len(line) <= 15 for line in lines)
    assert " ".join(lines) == "the quick brown fox jumps over the lazy dog"


def test_wrap_text_keeps_long_word_intact():
    lines = wrap_text("supercalifragilistic short", max_chars=10)
    assert "supercalifragilistic" in lines


def test_build_cues_no_overlap_and_ordered():
    segments = [
        Segment(start=0.0, end=2.0, text="Hello there friend"),
        Segment(start=2.1, end=4.0, text="How are you doing today"),
        Segment(start=4.0, end=6.5, text="I am doing quite well thanks"),
    ]
    cues = build_cues(segments, max_chars=42)
    # Monotonic, non-overlapping.
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start + 1e-9
        assert a.start <= a.end
    # Indices are 1-based and sequential.
    assert [c.index for c in cues] == list(range(1, len(cues) + 1))


def test_reading_speed_extends_short_cue():
    # A very brief segment with real text should be stretched to the min duration.
    segments = [Segment(start=0.0, end=0.2, text="This is several words of text")]
    cues = build_cues(segments, min_duration=0.7)
    assert cues[0].end - cues[0].start >= 0.7 - 1e-9


def test_long_segment_splits_into_multiple_cues():
    long_text = "word " * 40  # ~200 chars, far more than 2 lines of 42
    segments = [Segment(start=0.0, end=10.0, text=long_text.strip())]
    cues = build_cues(segments, max_chars=42, max_lines=2)
    assert len(cues) > 1
    # Sub-cues stay within the original segment window (allowing the reading-speed
    # floor to nudge only the final cue's end).
    assert cues[0].start == pytest.approx(0.0)


def test_to_srt_structure():
    segments = [
        Segment(start=0.0, end=2.0, text="First line"),
        Segment(start=2.5, end=4.0, text="Second line"),
    ]
    srt = to_srt(build_cues(segments))
    blocks = srt.strip().split("\n\n")
    assert len(blocks) == 2
    first = blocks[0].splitlines()
    assert first[0] == "1"
    assert "-->" in first[1]
    assert re.match(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", first[1])
    assert first[2] == "First line"


def test_to_vtt_header_and_dots():
    segments = [Segment(start=0.0, end=2.0, text="Hi")]
    vtt = to_vtt(build_cues(segments))
    assert vtt.startswith("WEBVTT\n")
    assert "-->" in vtt
    assert re.search(r"\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}", vtt)


def test_empty_segments_produce_no_cues():
    assert build_cues([Segment(start=0, end=1, text="   ")]) == []


def test_translate_segments_preserves_timestamps(fake_nim):
    import json

    segments = [
        Segment(start=0.0, end=1.0, text="Hello"),
        Segment(start=1.0, end=2.0, text="World"),
    ]

    def responder(messages):
        # Echo a JSON array of the same length in "Spanish".
        return json.dumps(["Hola", "Mundo"])

    client = fake_nim(responder)
    translated = translate_segments(segments, "Spanish", client)
    assert [s.text for s in translated] == ["Hola", "Mundo"]
    assert [(s.start, s.end) for s in translated] == [(0.0, 1.0), (1.0, 2.0)]


def test_translate_falls_back_when_count_mismatches(fake_nim):
    segments = [Segment(start=0.0, end=1.0, text="Hello"), Segment(start=1.0, end=2.0, text="World")]
    state = {"n": 0}

    def responder(messages):
        state["n"] += 1
        # First call (batch) returns a bad-shape array; per-line calls return text.
        if state["n"] == 1:
            return "[\"only-one\"]"
        return "traducido"

    client = fake_nim(responder)
    translated = translate_segments(segments, "Spanish", client)
    assert len(translated) == 2
    assert all(s.text == "traducido" for s in translated)
