"""Chunk-boundary math and transcript paragraph assembly (no whisper needed)."""

from __future__ import annotations

import pytest

from transcribe_studio.config import Segment
from transcribe_studio.transcribe import chunk_boundaries, is_video, segments_to_paragraphs


def test_no_chunking_when_disabled():
    assert chunk_boundaries(100.0, 0.0) == [(0.0, 100.0)]


def test_single_chunk_when_shorter_than_chunk():
    assert chunk_boundaries(30.0, 60.0) == [(0.0, 30.0)]


def test_chunk_boundaries_cover_full_duration():
    bounds = chunk_boundaries(100.0, 30.0)
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] == 100.0
    # Every window advances and none exceeds the duration.
    for start, end in bounds:
        assert 0.0 <= start < end <= 100.0


def test_chunk_boundaries_contiguous_without_overlap():
    bounds = chunk_boundaries(90.0, 30.0)
    assert bounds == [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]


def test_chunk_boundaries_with_overlap_steps_correctly():
    bounds = chunk_boundaries(100.0, 30.0, overlap=5.0)
    # Step is chunk_length - overlap = 25.
    starts = [round(b[0], 3) for b in bounds]
    assert starts[:3] == [0.0, 25.0, 50.0]
    assert bounds[-1][1] == 100.0


def test_overlap_larger_than_chunk_raises():
    with pytest.raises(ValueError):
        chunk_boundaries(100.0, 10.0, overlap=10.0)


def test_zero_duration_returns_zero_window():
    assert chunk_boundaries(0.0, 30.0) == [(0.0, 0.0)]


def test_is_video_detection():
    assert is_video("clip.mp4")
    assert is_video("movie.MKV")
    assert not is_video("audio.wav")
    assert not is_video("podcast.mp3")


def test_segments_to_paragraphs_breaks_on_gap():
    segments = [
        Segment(start=0.0, end=2.0, text="First sentence."),
        Segment(start=2.2, end=4.0, text="Still same paragraph."),
        Segment(start=10.0, end=12.0, text="New paragraph after a long pause."),
    ]
    text = segments_to_paragraphs(segments, gap=2.0)
    paragraphs = text.split("\n\n")
    assert len(paragraphs) == 2
    assert "First sentence." in paragraphs[0]
    assert "New paragraph" in paragraphs[1]


def test_segments_to_paragraphs_ignores_blank():
    segments = [Segment(start=0.0, end=1.0, text="  "), Segment(start=1.0, end=2.0, text="real")]
    assert segments_to_paragraphs(segments) == "real"
