"""transcribe() end to end with real ffmpeg and a fake Whisper model.

ffmpeg generates real media (a 70 s WAV tone, a small MP4) inside tmp_path, so
probing, chunk extraction, timestamp offsetting and the overlap stitching all
run for real; only the speech model is faked. Skipped when ffmpeg/ffprobe are
not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from types import SimpleNamespace

import pytest

from transcribe_studio.config import Segment, Word
from transcribe_studio.transcribe import (
    MediaDecodeError,
    chunk_cut_points,
    keep_between,
    transcribe,
)

_ffmpeg_missing = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def needs_ffmpeg(test):
    """Real ffmpeg work: skipped without ffmpeg, and marked slow."""
    return pytest.mark.slow(_ffmpeg_missing(test))


def _ffmpeg(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


class FakeWhisper:
    """Emits one 1.5 s segment (two words) every 2 s of whatever audio it gets.

    It reads the real WAV that transcribe() extracted, so the chunk durations it
    sees are the ones ffmpeg actually produced.
    """

    def __init__(self, detected="es"):
        self.detected = detected
        self.calls = []

    def transcribe(self, audio_path, language=None, **kwargs):
        try:
            with wave.open(audio_path) as w:
                duration = w.getnframes() / w.getframerate()
                fmt = (w.getnchannels(), w.getframerate())
        except wave.Error:
            # Non-WAV input handed straight to the model (as faster-whisper allows).
            from transcribe_studio.transcribe import probe_duration

            duration, fmt = probe_duration(audio_path), ("original", "original")
        self.calls.append({"duration": round(duration, 2), "language": language, "format": fmt})
        segments = []
        t = 0.0
        while t + 1.5 <= duration + 1e-6:
            words = [
                SimpleNamespace(start=t, end=t + 0.7, word=" hola"),
                SimpleNamespace(start=t + 0.75, end=t + 1.5, word=" mundo"),
            ]
            segments.append(SimpleNamespace(start=t, end=t + 1.5, text=" hola mundo", words=words))
            t += 2.0
        info = SimpleNamespace(language=language or self.detected)
        return iter(segments), info


@pytest.fixture(scope="module")
def tone_70s(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("media") / "tone.wav"
    _ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=70", str(path))
    return path


@needs_ffmpeg
def test_overlapping_chunks_lose_and_duplicate_nothing(tone_70s):
    model = FakeWhisper()
    result = transcribe(tone_70s, chunk_length=30, chunk_overlap=10, model=model)

    # Windows really overlap: [0,30] [20,50] [40,70].
    assert [c["duration"] for c in model.calls] == [30.0, 30.0, 30.0]
    starts = [round(s.start, 3) for s in result.segments]
    expected = [float(t) for t in range(0, 70, 2)]  # one segment per 2 s slot, 0..68
    assert starts == expected, f"missing={sorted(set(expected) - set(starts))}"
    assert len(starts) == len(set(starts))  # nothing doubled
    assert result.duration == pytest.approx(70.0, abs=0.05)
    # Every chunk fed Whisper 16 kHz mono audio.
    assert {c["format"] for c in model.calls} == {(1, 16000)}


@needs_ffmpeg
def test_auto_detected_language_is_locked_after_first_chunk(tone_70s):
    model = FakeWhisper(detected="es")
    result = transcribe(tone_70s, chunk_length=30, chunk_overlap=10, model=model)
    assert [c["language"] for c in model.calls] == [None, "es", "es"]
    assert result.language == "es"


@needs_ffmpeg
def test_forced_language_is_passed_to_every_chunk(tone_70s):
    model = FakeWhisper(detected="es")
    transcribe(tone_70s, language="en", chunk_length=30, chunk_overlap=2, model=model)
    assert [c["language"] for c in model.calls] == ["en", "en", "en"]


@needs_ffmpeg
def test_contiguous_chunks_without_overlap_still_cover_everything(tone_70s):
    model = FakeWhisper()
    result = transcribe(tone_70s, chunk_length=30, chunk_overlap=0, model=model)
    assert [c["duration"] for c in model.calls] == [30.0, 30.0, 10.0]
    assert [round(s.start, 3) for s in result.segments] == [float(t) for t in range(0, 70, 2)]


@needs_ffmpeg
def test_progress_reports_monotonic_time(tone_70s):
    seen = []
    transcribe(tone_70s, chunk_length=30, chunk_overlap=10, model=FakeWhisper(),
               progress=lambda done, total: seen.append((done, total)))
    assert seen, "progress was never reported"
    assert all(total == pytest.approx(70.0, abs=0.05) for _, total in seen)
    assert seen[-1][0] == pytest.approx(69.5)


@needs_ffmpeg
def test_video_audio_is_extracted_with_ffmpeg(tmp_path):
    mp4 = tmp_path / "clip.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=300:duration=4",
        "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(mp4),
    )
    model = FakeWhisper(detected="en")
    result = transcribe(mp4, model=model)
    assert len(model.calls) == 1
    assert model.calls[0]["format"] == (1, 16000)
    assert model.calls[0]["duration"] == pytest.approx(4.0, abs=0.1)
    assert [s.start for s in result.segments] == [0.0, 2.0]
    assert result.language == "en"
    assert result.text == "hola mundo hola mundo"


@needs_ffmpeg
def test_undecodable_video_raises_media_decode_error(tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"this is not a video file")
    with pytest.raises(MediaDecodeError, match="broken.mp4"):
        transcribe(bad, model=FakeWhisper())


def test_overlap_must_be_smaller_than_chunk(tmp_path):
    media = tmp_path / "x.wav"
    media.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="chunk_overlap"):
        transcribe(media, chunk_length=10, chunk_overlap=10, model=FakeWhisper())


def test_cut_points_are_overlap_midpoints():
    assert chunk_cut_points([(0, 30), (20, 50), (40, 70)]) == [25.0, 45.0]
    assert chunk_cut_points([(0, 30), (30, 60)]) == [30.0]
    assert chunk_cut_points([(0, 10)]) == []


def test_keep_between_splits_a_straddling_segment_at_word_level():
    words = [Word(24.0, 24.4, " one"), Word(24.6, 24.9, " two"),
             Word(25.1, 25.5, " three"), Word(25.6, 26.0, " four")]
    seg = Segment(start=23.9, end=26.1, text="one two three four", words=words)
    left = keep_between([seg], float("-inf"), 25.0)
    right = keep_between([seg], 25.0, float("inf"))
    assert [s.text for s in left] == ["one two"]
    assert [s.text for s in right] == ["three four"]
    assert left[0].start == 23.9 and left[0].end == 24.9
    assert right[0].start == 25.1 and right[0].end == 26.1
    # Together they reproduce the original words exactly once.
    assert [w.word for s in left + right for w in s.words] == [w.word for w in words]


def test_keep_between_without_words_uses_segment_start():
    segs = [Segment(24.0, 26.0, "a"), Segment(25.0, 27.0, "b")]
    assert [s.text for s in keep_between(segs, float("-inf"), 25.0)] == ["a"]
    assert [s.text for s in keep_between(segs, 25.0, float("inf"))] == ["b"]
