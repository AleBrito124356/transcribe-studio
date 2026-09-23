"""TF-IDF chapter detection, distinctive titles and YouTube normalisation."""

from __future__ import annotations

import random
import time

from transcribe_studio.chapters import (
    Chapter,
    chapters_to_markdown,
    detect_chapters,
    detect_chapters_local,
    fold,
    normalize_chapters,
)
from transcribe_studio.config import Segment


def _segs(spec):
    return [Segment(start=s, end=e, text=t) for s, e, t in spec]


def _two_topics(noise="podcast podcast podcast", a="database schema index", b="cooking tomato garlic",
                length=180, step=5):
    half = length // 2
    return _segs([(t, t + step, f"{noise} {a}") for t in range(0, half, step)]
                 + [(t, t + step, f"{noise} {b}") for t in range(half, length, step)])


def test_recurring_word_no_longer_hides_a_topic_change():
    # Audit case: raw term frequency returned ONE chapter titled "Podcast Database Schema Index".
    chapters = detect_chapters_local(_two_topics(), window_seconds=20, min_chapter_seconds=40)
    assert len(chapters) >= 2
    assert any(80 <= c.start <= 100 for c in chapters[1:])
    assert all("podcast" not in c.title.lower() for c in chapters)
    boundary = next(c for c in chapters if 80 <= c.start <= 100)
    assert boundary.title == "Cooking Tomato Garlic"


def test_accented_spanish_function_words_never_become_titles():
    segments = _segs([(t, t + 5, "también está aquí más qué así o sea bueno entonces") for t in range(0, 60, 5)])
    for chapter in detect_chapters_local(segments):
        for word in chapter.title.split():
            assert fold(word) not in {"tambien", "esta", "aqui", "mas", "que", "asi", "bueno", "entonces"}


def test_spanish_topics_keep_their_accents_and_casing():
    a = "hoy hablamos del jardín y la cosecha de tomates en el balcón"
    b = "ahora pasamos al café de especialidad y la molienda con la prensa francesa"
    segments = _two_topics(noise="bueno pues", a=a, b=b, length=240)
    chapters = detect_chapters_local(segments, window_seconds=30, min_chapter_seconds=60)
    titles = " ".join(c.title for c in chapters)
    assert "Jardín" in titles or "Balcón" in titles or "Cosecha" in titles
    assert "Café" in titles or "Molienda" in titles or "Prensa" in titles
    assert "Bueno" not in titles and "Pues" not in titles
    assert any(100 <= c.start <= 140 for c in chapters)


def test_original_casing_of_names_is_kept():
    a = "we migrated PostgreSQL replicas and tuned PostgreSQL vacuum settings"
    b = "then we benchmarked NVIDIA GPUs with CUDA kernels and NVIDIA drivers"
    chapters = detect_chapters_local(_two_topics(noise="", a=a, b=b), window_seconds=20, min_chapter_seconds=40)
    titles = [c.title for c in chapters]
    assert any("PostgreSQL" in t for t in titles)
    assert any("NVIDIA" in t for t in titles)


def test_titles_are_never_reused():
    # A-B-A structure: both "A" chapters would get the same keywords.
    a = "backup restore snapshot offsite"
    b = "tomato garden watering soil"
    spec = ([(t, t + 5, a) for t in range(0, 100, 5)] + [(t, t + 5, b) for t in range(100, 200, 5)]
            + [(t, t + 5, a) for t in range(200, 300, 5)])
    chapters = detect_chapters_local(_segs(spec), window_seconds=20, min_chapter_seconds=40)
    titles = [c.title.lower() for c in chapters]
    assert len(chapters) >= 3
    assert len(titles) == len(set(titles))


def test_single_topic_short_media_stays_one_chapter():
    segments = _segs([(t, t + 5, "weather sunny warm forecast temperature") for t in range(0, 120, 5)])
    chapters = detect_chapters_local(segments, window_seconds=20, min_chapter_seconds=60)
    assert [c.start for c in chapters] == [0.0]


def test_long_media_gets_youtubes_three_chapter_minimum():
    segments = _segs([(t, t + 5, "weather sunny warm forecast temperature") for t in range(0, 600, 5)])
    chapters = detect_chapters_local(segments)
    assert len(chapters) >= 3
    assert chapters[0].start == 0.0
    starts = [c.start for c in chapters]
    assert all(b - a >= 10 for a, b in zip(starts, starts[1:]))


def test_silence_still_marks_a_boundary():
    spec = [(t, t + 5, "same words every time here") for t in range(0, 100, 5)]
    spec += [(t, t + 5, "same words every time here") for t in range(110, 200, 5)]  # 10 s silence at 100
    chapters = detect_chapters_local(_segs(spec), window_seconds=20, min_chapter_seconds=40)
    assert any(105 <= c.start <= 115 for c in chapters)


def test_max_chapters_caps_the_count_keeping_the_strongest_shifts():
    topics = ["alpha bravo charlie", "delta echo foxtrot", "golf hotel india", "juliet kilo lima",
              "mike november oscar", "papa quebec romeo"]
    spec = [(t, t + 5, topics[t // 100]) for t in range(0, 600, 5)]
    chapters = detect_chapters_local(_segs(spec), window_seconds=20, min_chapter_seconds=40, max_chapters=4)
    assert len(chapters) == 4
    assert all(c.start % 100 == 0 for c in chapters)


def test_normalize_fixes_the_audit_nim_reply():
    raw = [Chapter(0.0, "Intro"), Chapter(3.0, "Too close"), Chapter(2700.0, "Past the end")]
    out = normalize_chapters(raw, duration=60.0)
    assert out[0].start == 0.0
    assert all(c.start <= 50.0 for c in out)  # nothing past the end or too close to it
    starts = [c.start for c in out]
    assert all(b - a >= 10 for a, b in zip(starts, starts[1:]))
    assert out == [Chapter(0.0, "Too close")]


def test_normalize_youtube_rules():
    # first chapter slightly late -> moved to 00:00
    assert normalize_chapters([Chapter(6, "Hello"), Chapter(60, "Next")], 300)[0] == Chapter(0.0, "Hello")
    # first chapter much later -> an Intro is inserted
    out = normalize_chapters([Chapter(45, "Main"), Chapter(120, "Later")], 300)
    assert [c.title for c in out] == ["Intro", "Main", "Later"]
    # consecutive duplicates merge; far-apart duplicates are numbered
    out = normalize_chapters([Chapter(0, "Setup"), Chapter(40, "setup"), Chapter(80, "Deploy"),
                              Chapter(120, "Setup")], 300)
    assert [(c.start, c.title) for c in out] == [(0.0, "Setup"), (80.0, "Deploy"), (120.0, "Setup (part 2)")]
    # titles are cleaned, empties dropped
    out = normalize_chapters([Chapter(0, "  - Intro:  "), Chapter(30, "   "), Chapter(60, '"Wrap up"')], 300)
    assert [c.title for c in out] == ["Intro", "Wrap up"]


def test_normalize_fills_to_three_chapters_on_long_media_from_segments():
    a = [(t, t + 5, "backup restore snapshot offsite drive") for t in range(0, 300, 5)]
    b = [(t, t + 5, "tomato garden watering soil compost") for t in range(300, 600, 5)]
    segments = _segs(a + b)
    out = normalize_chapters([Chapter(0, "Backups"), Chapter(300, "Gardening")], 600, segments=segments)
    assert len(out) == 3
    assert [c.title for c in out if c.start in (0.0, 300.0)] == ["Backups", "Gardening"]


def test_nim_chapters_are_normalised(fake_nim, make_segments):
    segments = make_segments([(t, t + 5, "talking about the thing") for t in range(0, 60, 5)])
    client = fake_nim(lambda m: "00:00 Intro\n00:03 Too close\n45:00 Past the end")
    chapters = detect_chapters(segments, nim_client=client)
    assert [c.start for c in chapters] == [0.0]


def test_chapters_are_deterministic():
    segments = _two_topics(length=600)
    first = chapters_to_markdown(detect_chapters_local(segments))
    second = chapters_to_markdown(detect_chapters_local(list(reversed(segments))))
    assert first == second


def test_five_thousand_segments_in_under_two_seconds():
    rng = random.Random(3)
    vocab = [f"term{i}" for i in range(3000)]
    segments = []
    t = 0.0
    for i in range(5000):
        topic = (i // 400) * 50  # a new vocabulary slice every 400 segments
        words = " ".join(rng.choice(vocab[topic: topic + 200]) for _ in range(12))
        segments.append(Segment(start=t, end=t + 3.5, text=words))
        t += 4.0
    began = time.perf_counter()
    chapters = detect_chapters_local(segments)
    elapsed = time.perf_counter() - began
    assert elapsed < 2.0, f"took {elapsed:.2f}s"
    assert 3 <= len(chapters) <= 15
