"""Word-timed cues, strict SRT/VTT validity, escaping, speakers and lint."""

from __future__ import annotations

import html
import random
import re

import pytest

from transcribe_studio.config import Segment, Word
from transcribe_studio.subtitles import (
    Cue,
    balance_lines,
    build_cues,
    to_srt,
    to_vtt,
    validate_cues,
    write_captions,
)

SRT_TIME = r"(\d{2}):([0-5]\d):([0-5]\d),(\d{3})"
VTT_TIME = r"(\d{2}):([0-5]\d):([0-5]\d)\.(\d{3})"


def _secs(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt_strict(text):
    """Parse SRT, failing loudly on anything a strict player would reject."""
    assert text.endswith("\n") and not text.startswith("\n")
    cues = []
    for n, block in enumerate(text.strip("\n").split("\n\n"), start=1):
        lines = block.split("\n")
        assert lines[0] == str(n), f"bad index in block {n}: {lines[0]!r}"
        m = re.fullmatch(SRT_TIME + " --> " + SRT_TIME, lines[1])
        assert m, f"bad timing line {lines[1]!r}"
        start, end = _secs(*m.groups()[:4]), _secs(*m.groups()[4:])
        body = lines[2:]
        assert body and all(line.strip() for line in body), f"empty text in cue {n}"
        assert not any("-->" in line for line in body)
        cues.append((start, end, body))
    return cues


def parse_vtt_strict(text):
    assert text.startswith("WEBVTT\n\n")
    cues = []
    for block in text[len("WEBVTT\n\n"):].strip("\n").split("\n\n"):
        lines = block.split("\n")
        m = re.fullmatch(VTT_TIME + " --> " + VTT_TIME, lines[0])
        assert m, f"bad timing line {lines[0]!r}"
        body = lines[1:]
        assert body and all(line.strip() for line in body)
        voice = None
        for i, line in enumerate(body):
            assert "-->" not in line
            vm = re.match(r"<v ([^>]+)>", line)
            if vm:
                assert i == 0
                voice = html.unescape(vm.group(1))
                line = line[vm.end():]
            # after removing the voice tag, no raw markup may remain
            assert "<" not in line and ">" not in line, line
            assert not re.search(r"&(?!amp;|lt;|gt;)", line), line
            body[i] = html.unescape(line)
        cues.append((_secs(*m.groups()[:4]), _secs(*m.groups()[4:]), body, voice))
    return cues


def _words(spec):
    """spec: list of (start, end, text) -> Words with Whisper-style leading spaces."""
    return [Word(start=s, end=e, word=" " + t) for s, e, t in spec]


def _seg(spec, speaker=None):
    words = _words(spec)
    return Segment(start=spec[0][0], end=spec[-1][1], text=" ".join(t for _, _, t in spec),
                   words=words, speaker=speaker)


# ---------------------------------------------------------------------------
# Timing follows the words
# ---------------------------------------------------------------------------
def test_cues_follow_word_timestamps_not_character_share():
    # The audit's drift case: "Short intro." then a 6.8 s pause, then fast words.
    spec = [(0.0, 0.5, "Short"), (0.5, 1.2, "intro."), (8.0, 8.25, "Then")]
    spec += [(8.3 + i * 0.1, 8.3 + i * 0.1 + 0.09, "word") for i in range(15)]
    seg = Segment(0, 10, "Short intro. Then " + " ".join(["word"] * 15), words=_words(spec))
    cues = build_cues([seg], max_chars=20, max_lines=1)

    assert cues[0].lines == ["Short intro."]
    assert cues[0].start == 0.0 and cues[0].end < 8.0  # gone before "Then" is spoken
    then_cue = next(c for c in cues if "Then" in c.text)
    assert then_cue.start >= 8.0
    for cue in cues[1:]:
        assert 8.0 <= cue.start <= 10.0
        if cue is not then_cue:
            assert cue.start >= 8.3  # a "word" cue never appears before the words
    # Old proportional timing put "Then" at ~1.9 s; make sure that is gone.
    assert all(not (1.0 < c.start < 8.0) for c in cues)


def test_each_cue_starts_on_its_first_word_and_covers_its_last():
    spec = [(i * 0.4, i * 0.4 + 0.3, f"w{i}") for i in range(40)]
    seg = _seg(spec)
    cues = build_cues([seg], max_chars=20, max_lines=2, min_duration=0.1, max_cps=100)
    by_text = {w.word.strip(): w for w in seg.words}
    for cue in cues:
        tokens = cue.text.split()
        first, last = by_text[tokens[0]], by_text[tokens[-1]]
        assert cue.start == pytest.approx(first.start)
        assert cue.end >= last.end - 0.041  # only trimmed by the min gap before the next cue


def test_breaks_prefer_sentence_ends_and_pauses():
    spec = [(0.0, 0.3, "We"), (0.3, 0.6, "fixed"), (0.6, 0.9, "the"), (0.9, 1.3, "bug."),
            (1.4, 1.7, "Now"), (1.7, 2.0, "we"), (2.0, 2.4, "ship"), (2.4, 2.8, "the"),
            (2.8, 3.3, "release"), (3.3, 3.8, "tonight")]
    cues = build_cues([_seg(spec)], max_chars=30, max_lines=1)
    assert cues[0].text == "We fixed the bug."
    assert cues[1].text.startswith("Now")


def test_long_silence_always_splits_and_long_cues_are_capped():
    spec = [(0.0, 0.4, "Hello"), (0.4, 0.8, "there"), (2.5, 2.9, "friend")]  # 1.7 s gap
    cues = build_cues([_seg(spec)])
    assert [c.text for c in cues] == ["Hello there", "friend"]

    slow = [(i * 1.0, i * 1.0 + 0.9, f"x{i}") for i in range(12)]  # 12 s of slow speech
    cues = build_cues([_seg(slow)], max_chars=80, max_lines=2)
    assert len(cues) >= 2
    assert all(c.duration <= 7.0 + 1e-9 for c in cues)


def test_edited_text_keeps_word_timing_when_token_count_matches():
    seg = _seg([(0.0, 0.4, "helo"), (0.5, 0.9, "wrold"), (3.0, 3.4, "again")])
    seg.text = "Hello world again"  # corrected by hand in transcript.json
    cues = build_cues([seg])
    assert [c.text for c in cues] == ["Hello world", "again"]
    assert cues[1].start == 3.0


def test_edited_text_with_different_word_count_falls_back_to_proportional():
    seg = _seg([(0.0, 0.4, "helo"), (0.5, 0.9, "wrold")])
    seg.text = "Hello, wide world"
    cues = build_cues([seg])
    assert [c.text for c in cues] == ["Hello, wide world"]  # the edit is never lost


def test_languages_without_spaces_join_correctly():
    words = [Word(0.0, 0.3, "今日"), Word(0.3, 0.6, "は"), Word(0.6, 1.0, "晴れ")]
    seg = Segment(0.0, 1.0, "今日は晴れ", words=words)
    assert [c.text for c in build_cues([seg])] == ["今日は晴れ"]


def test_overlapping_whisper_segments_become_sequential_cues():
    a = _seg([(0.0, 0.5, "first"), (0.5, 3.0, "segment")])
    b = _seg([(2.0, 2.5, "second"), (2.5, 4.0, "segment")])  # starts inside a
    c = Segment(start=1.0, end=5.0, text="third without words")  # starts even earlier
    cues = build_cues([a, b, c])
    for x, y in zip(cues, cues[1:]):
        assert x.start < x.end <= y.start


# ---------------------------------------------------------------------------
# Property-style: random transcripts always give strictly valid files
# ---------------------------------------------------------------------------
def _random_transcript(rng):
    vocab = ("the backup restore snapshot drive plan tomato garlic sauce pan heat "
             "salt quickly slowly really, maybe; yes. no? antidisestablishmentarianism "
             "Q&A <3 --> R&D a b").split()
    segments, t = [], rng.uniform(0, 2)
    for _ in range(rng.randint(1, 25)):
        n = rng.randint(1, 30)
        spec = []
        for _ in range(n):
            dur = rng.uniform(0.05, 0.8)
            spec.append((round(t, 3), round(t + dur, 3), rng.choice(vocab)))
            t += dur + rng.choice([0, 0, 0.05, 0.2, 0.6, 1.5])
        seg = _seg(spec, speaker=rng.choice([None, "Speaker 1", "Speaker 2"]))
        if rng.random() < 0.25:
            seg.words = []  # some segments lack word timings
        segments.append(seg)
        t -= rng.choice([0, 0, 0, 0.3, 1.0])  # Whisper segments sometimes overlap
    return segments


@pytest.mark.parametrize("seed", range(40))
def test_random_transcripts_produce_strictly_valid_captions(seed):
    rng = random.Random(seed)
    segments = _random_transcript(rng)
    max_chars = rng.choice([20, 32, 42])
    max_lines = rng.choice([1, 2])
    labels = rng.random() < 0.5
    cues = build_cues(segments, max_chars=max_chars, max_lines=max_lines, speaker_labels=labels)

    srt = parse_srt_strict(to_srt(cues))
    vtt = parse_vtt_strict(to_vtt(cues))
    assert len(srt) == len(vtt) == len(cues)
    for (s1, e1, _), (s2, e2, _, _) in zip(srt, vtt):
        assert (s1, e1) == (s2, e2)

    for (start, end, lines), nxt in zip(srt, srt[1:] + [None]):
        assert start < end
        assert len(lines) <= max_lines
        for line in lines:
            assert len(line) <= max_chars or " " not in line.strip()
        if nxt is not None:
            assert end <= nxt[0], "cues overlap"

    report = validate_cues(cues, max_chars=max_chars, max_lines=max_lines)
    assert report.overlaps == 0 and report.out_of_order == 0
    assert report.tall_cues == 0 and report.wide_lines == 0 and report.empty_cues == 0

    # Nothing lost or duplicated: every spoken word is shown exactly once.
    spoken = [w for s in sorted(segments, key=lambda s: s.start) for w in s.text.split()]
    shown = []
    for cue in cues:
        text = cue.text
        if cue.label:
            text = text.replace(cue.label, "", 1)
        shown.extend(text.split())
    assert sorted(shown) == sorted(spoken)
    assert len(shown) == len(spoken)


# ---------------------------------------------------------------------------
# Escaping, speakers, layout, fallback compatibility, lint
# ---------------------------------------------------------------------------
def test_vtt_escapes_markup_and_arrows_and_srt_neutralises_arrows():
    cues = build_cues([Segment(0, 2, "Q&A with Tom <3 --> next")])
    vtt = to_vtt(cues)
    assert "Q&amp;A with Tom &lt;3 --&gt; next" in vtt
    assert vtt.count("-->") == 1  # only the timing line
    assert parse_vtt_strict(vtt)[0][2] == ["Q&A with Tom <3 --> next"]
    srt = to_srt(cues)
    assert srt.count("-->") == 1
    assert "Q&A with Tom <3 -> next" in srt


def test_speaker_labels_srt_prefix_on_turns_and_vtt_voice_tags():
    segs = [
        _seg([(0.0, 0.5, "Hi"), (0.5, 1.0, "there.")], speaker="Speaker 1"),
        _seg([(1.5, 2.0, "Still"), (2.0, 2.5, "me.")], speaker="Speaker 1"),
        _seg([(3.0, 3.5, "Hello"), (3.5, 4.0, "back.")], speaker="Speaker 2"),
    ]
    cues = build_cues(segs, speaker_labels=True)
    srt_blocks = [c[2] for c in parse_srt_strict(to_srt(cues))]
    assert srt_blocks[0][0].startswith("[Speaker 1] Hi")
    assert not srt_blocks[1][0].startswith("[")  # same speaker: no repeated label
    assert srt_blocks[2][0].startswith("[Speaker 2] Hello")
    vtt = parse_vtt_strict(to_vtt(cues))
    assert [v for *_, v in vtt] == ["Speaker 1", "Speaker 1", "Speaker 2"]
    assert vtt[0][2] == ["Hi there."]  # the voice tag replaces the visible prefix
    assert "<v Speaker 2>Hello back." in to_vtt(cues)
    # Without the option nothing changes.
    plain = build_cues(segs)
    assert all(c.speaker is None and c.label is None for c in plain)
    assert "<v " not in to_vtt(plain) and "[Speaker" not in to_srt(plain)


def test_speaker_label_is_never_split_across_lines():
    seg = _seg([(i * 0.3, i * 0.3 + 0.25, w) for i, w in
                enumerate("alpha beta gamma delta epsilon zeta eta theta".split())], speaker="Speaker 12")
    for width in (12, 14, 16, 20):
        for cue in build_cues([seg], max_chars=width, max_lines=2, speaker_labels=True):
            joined = "\n".join(cue.lines)
            assert "[Speaker\n" not in joined and "\n12]" not in joined


def test_balance_lines_prefers_even_lines_and_punctuation():
    assert balance_lines("the quick brown fox jumps over the lazy dog", 42) == [
        "the quick brown fox", "jumps over the lazy dog"]
    assert balance_lines("Yes, we tested it and the restore worked fine", 42) == [
        "Yes, we tested it", "and the restore worked fine"]
    assert balance_lines("short", 42) == ["short"]


def test_wordless_fallback_keeps_the_old_proportional_timing():
    # Recorded from the previous release's build_cues() on the same input.
    segs = [
        Segment(0.0, 2.5, "Welcome back to the show."),
        Segment(2.7, 9.9, "Today we are talking about backups, why they fail silently, "
                          "and how to test a restore before you actually need one."),
        Segment(10.4, 10.6, "Right."),
        Segment(11.0, 19.0, "The rule of thumb is three copies, on two different kinds of "
                            "storage, with one of them kept somewhere else entirely."),
        Segment(19.0, 21.0, "Makes sense to me"),
        Segment(21.1, 30.0, "word " * 30),
    ]
    old = [
        (0.0, 2.5, "Welcome back to the show."),
        (2.7, 7.502478, "Today we are talking about backups, why they fail silently, and how to test a"),
        (7.542478, 9.9, "restore before you actually need one."),
        (10.4, 10.96, "Right."),
        (11.0, 16.574035, "The rule of thumb is three copies, on two different kinds of storage, with one of"),
        (16.614035, 18.96, "them kept somewhere else entirely."),
        (19.0, 21.0, "Makes sense to me"),
        (21.1, 25.814795, " ".join(["word"] * 16)),
        (25.854795, 30.0, " ".join(["word"] * 14)),
    ]
    new = [(round(c.start, 6), round(c.end, 6), " ".join(c.lines)) for c in build_cues(segs)]
    assert new == old


def test_validate_cues_flags_problems():
    cues = [
        Cue(1, 0.0, 2.0, ["fine"]),
        Cue(2, 1.5, 2.5, ["overlaps the previous cue"]),
        Cue(3, 3.0, 3.2, ["way too much text for two hundred ms"]),
        Cue(4, 4.0, 5.0, ["a line that is definitely wider than twenty chars"]),
        Cue(5, 6.0, 7.0, ["one", "two", "three"]),
    ]
    report = validate_cues(cues, max_chars=20, max_lines=2, max_cps=17)
    assert report.overlaps == 1
    assert report.fast_cues >= 1 and report.max_cps > 17
    assert report.wide_lines == 3  # cues 2, 3 and 4 have lines wider than 20 chars
    assert report.tall_cues == 1
    assert not report.ok
    assert "1 overlaps" in report.summary()


def test_write_captions_writes_both_formats_and_reports(tmp_path):
    segs = [_seg([(0.0, 0.4, "Hello"), (0.4, 0.9, "world.")]),
            _seg([(1.5, 1.9, "Second"), (1.9, 2.4, "line.")])]
    report = write_captions(segs, tmp_path / "c.srt", tmp_path / "c.vtt")
    assert report.cues == 2 and report.ok
    assert len(parse_srt_strict((tmp_path / "c.srt").read_text(encoding="utf-8"))) == 2
    assert len(parse_vtt_strict((tmp_path / "c.vtt").read_text(encoding="utf-8"))) == 2
