"""Chapter timestamp formatting, parsing, and topic-shift detection."""

from __future__ import annotations

from src.chapters import (
    chapters_to_markdown,
    detect_chapters,
    detect_chapters_local,
    parse_chapter_lines,
)
from src.config import Segment, format_chapter_timestamp, parse_timestamp


def test_chapter_timestamp_youtube_style():
    assert format_chapter_timestamp(0) == "00:00"
    assert format_chapter_timestamp(65) == "01:05"
    assert format_chapter_timestamp(600) == "10:00"
    assert format_chapter_timestamp(3725) == "1:02:05"


def test_parse_timestamp_roundtrip():
    for seconds in (0, 65, 600, 3725):
        assert int(parse_timestamp(format_chapter_timestamp(seconds))) == seconds


def test_parse_timestamp_accepts_srt_millis():
    assert parse_timestamp("00:01:01,250") == 61.25


def test_parse_chapter_lines_various_formats():
    text = """
    00:00 Intro
    - 01:30 The main idea
    [02:45] Deep dive
    1:00:00 Wrap up
    garbage line without timestamp
    """
    chapters = parse_chapter_lines(text)
    assert [c.start for c in chapters] == [0.0, 90.0, 165.0, 3600.0]
    assert chapters[0].title == "Intro"
    assert chapters[1].title == "The main idea"


def test_detect_chapters_local_first_is_zero(make_segments):
    # Two clearly different topics separated in time.
    topic_a = [(t, t + 5, "database migration schema postgres index") for t in range(0, 90, 5)]
    topic_b = [(t, t + 5, "cooking recipe tomato onion garlic kitchen") for t in range(90, 180, 5)]
    segments = make_segments(topic_a + topic_b)
    chapters = detect_chapters_local(segments, window_seconds=20, min_chapter_seconds=40)
    assert chapters[0].start == 0.0
    assert len(chapters) >= 2
    # A boundary should land near the topic change at 90s.
    assert any(80 <= c.start <= 100 for c in chapters[1:])


def test_detect_chapters_local_single_topic_stays_one(make_segments):
    same = [(t, t + 5, "weather sunny warm forecast temperature") for t in range(0, 120, 5)]
    segments = make_segments(same)
    chapters = detect_chapters_local(segments, window_seconds=20, min_chapter_seconds=40, sim_threshold=0.1)
    assert chapters[0].start == 0.0


def test_detect_chapters_uses_nim_when_client_present(make_segments, fake_nim):
    segments = make_segments([(t, t + 5, "some talking here about things") for t in range(0, 60, 5)])

    def responder(messages):
        return "00:00 Opening\n00:30 Main topic"

    client = fake_nim(responder)
    chapters = detect_chapters(segments, nim_client=client)
    assert chapters[0].start == 0.0
    assert chapters[0].title == "Opening"
    assert client.calls  # NIM was actually consulted


def test_detect_chapters_falls_back_on_nim_error(make_segments, fake_nim):
    segments = make_segments([(t, t + 5, "topic content words here") for t in range(0, 80, 5)])

    def responder(messages):
        raise RuntimeError("nim down")

    client = fake_nim(responder)
    chapters = detect_chapters(segments, nim_client=client)
    assert chapters  # degraded to local heuristic, still returns chapters
    assert chapters[0].start == 0.0


def test_chapters_to_markdown():
    from src.chapters import Chapter

    md = chapters_to_markdown([Chapter(0.0, "Intro"), Chapter(90.0, "Details")])
    assert "# Chapters" in md
    assert "00:00 Intro" in md
    assert "01:30 Details" in md
