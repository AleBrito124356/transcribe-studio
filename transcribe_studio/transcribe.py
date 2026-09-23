"""Local speech-to-text with faster-whisper, plus ffmpeg audio extraction.

Design goals:
- Import cleanly without faster-whisper installed (it is imported lazily inside
  the functions that need it), so the pure helpers and the test-suite run anywhere.
- Handle both audio and video: video files have their audio extracted with ffmpeg
  first, with a friendly message if ffmpeg is missing.
- Support long files via optional chunking with timestamp offsetting.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from .config import (
    VIDEO_EXTENSIONS,
    Segment,
    TranscriptResult,
    Word,
)

_FFMPEG_HINT = (
    "ffmpeg was not found on your PATH. transcribe-studio needs it to decode\n"
    "audio and to pull the audio track out of video files.\n"
    "  - macOS:    brew install ffmpeg\n"
    "  - Ubuntu:   sudo apt install ffmpeg\n"
    "  - Windows:  winget install Gyan.FFmpeg   (or  choco install ffmpeg)\n"
    "Then re-run. Verify with:  ffmpeg -version"
)


class FfmpegNotFound(RuntimeError):
    """Raised when ffmpeg is required but not installed."""


class MediaDecodeError(RuntimeError):
    """Raised when ffmpeg cannot decode or extract audio from the input."""


class WhisperUnavailable(RuntimeError):
    """Raised when faster-whisper is missing or the model cannot be loaded."""


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def ensure_ffmpeg() -> None:
    if not ffmpeg_available():
        raise FfmpegNotFound(_FFMPEG_HINT)


def is_video(path: str | os.PathLike) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def probe_duration(path: str | os.PathLike) -> Optional[float]:
    """Return media duration in seconds via ffprobe, or ``None`` if unavailable."""
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    try:
        return float(value)
    except ValueError:
        return None


def extract_audio(
    media_path: str | os.PathLike,
    out_wav: str | os.PathLike,
    sample_rate: int = 16000,
    start: Optional[float] = None,
    duration: Optional[float] = None,
) -> str:
    """Extract a mono 16 kHz WAV from any media file using ffmpeg.

    ``start`` / ``duration`` let callers pull a single chunk out of a long file.
    """
    ensure_ffmpeg()
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(media_path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "wav", str(out_wav)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        raise MediaDecodeError(
            f"ffmpeg could not read audio from {Path(media_path).name}: "
            + (detail[-1] if detail else f"exit code {proc.returncode}")
        )
    return str(out_wav)


def chunk_boundaries(
    duration: float, chunk_length: float, overlap: float = 0.0
) -> List[Tuple[float, float]]:
    """Split ``[0, duration]`` into ``(start, end)`` windows.

    A single window is returned when chunking is disabled (``chunk_length <= 0``)
    or the file is shorter than one chunk. Consecutive windows advance by
    ``chunk_length - overlap`` and the final window always ends exactly at
    ``duration``.
    """
    if duration <= 0:
        return [(0.0, 0.0)]
    if chunk_length <= 0 or duration <= chunk_length:
        return [(0.0, duration)]
    step = chunk_length - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_length")
    bounds: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + chunk_length, duration)
        bounds.append((start, end))
        if end >= duration:
            break
        start += step
    return bounds


def _load_model(model_size: str, device: str, compute_type: str):
    """Instantiate a faster-whisper model (imported lazily).

    Raises :class:`WhisperUnavailable` with an actionable message when
    faster-whisper is not installed or the model cannot be loaded (the first
    use of a size downloads it from Hugging Face, which needs a connection).
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise WhisperUnavailable(
            "faster-whisper is not installed, so media cannot be transcribed. Install it "
            "with `pip install faster-whisper` (or `pip install -r requirements.txt`). "
            "Commands given a transcript.json do not need it."
        ) from exc
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:
        first_line = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        raise WhisperUnavailable(
            f"could not load the Whisper model '{model_size}' ({type(exc).__name__}: "
            f"{first_line}). The first use of a model size downloads it from Hugging Face, "
            "so check your connection, or pass a local model directory with --model. "
            "On GPU errors, retry with --device cpu --compute-type int8."
        ) from exc


def _run_model(
    model,
    audio_path: str,
    language: Optional[str],
    beam_size: int,
    vad_filter: bool,
    word_timestamps: bool,
    time_offset: float = 0.0,
) -> Tuple[List[Segment], str]:
    """Run one decode pass and return offset segments plus detected language."""
    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
    )
    out: List[Segment] = []
    for seg in segments_iter:
        words: List[Word] = []
        for w in getattr(seg, "words", None) or []:
            words.append(
                Word(
                    start=float(w.start) + time_offset,
                    end=float(w.end) + time_offset,
                    word=w.word,
                )
            )
        out.append(
            Segment(
                start=float(seg.start) + time_offset,
                end=float(seg.end) + time_offset,
                text=seg.text.strip(),
                words=words,
            )
        )
    return out, getattr(info, "language", language or "")


def transcribe(
    media_path: str | os.PathLike,
    model_size: str = "base",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "int8",
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    chunk_length: float = 0.0,
    chunk_overlap: float = 0.0,
    model=None,
) -> TranscriptResult:
    """Transcribe an audio or video file into a :class:`TranscriptResult`.

    Parameters
    ----------
    model_size:
        faster-whisper size: ``tiny``, ``base``, ``small``, ``medium``,
        ``large-v3`` (or the distil-* variants).
    language:
        Force a language (e.g. ``"en"``/``"es"``) or leave ``None`` to auto-detect.
    device / compute_type:
        ``"auto"``/``"cpu"``/``"cuda"`` and ``int8``/``int8_float16``/``float16``.
    chunk_length:
        When > 0 and the file is longer than one chunk, decode in ffmpeg-cut
        windows and stitch timestamps back together. Requires ffprobe.
    model:
        A preloaded ``WhisperModel`` to reuse across a batch. Loaded on demand
        otherwise.
    """
    media_path = str(media_path)
    if not Path(media_path).exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    owns_model = model is None
    if owns_model:
        model = _load_model(model_size, device, compute_type)

    tmp_dir = tempfile.mkdtemp(prefix="transcribe_studio_")
    created_files: List[str] = []
    try:
        duration = probe_duration(media_path)
        bounds = chunk_boundaries(duration, chunk_length) if (chunk_length and duration) else [(0.0, duration or 0.0)]

        all_segments: List[Segment] = []
        detected_language = language or ""

        if len(bounds) == 1 and not is_video(media_path):
            # Simplest path: hand the file straight to faster-whisper.
            segs, detected_language = _run_model(
                model, media_path, language, beam_size, vad_filter, word_timestamps
            )
            all_segments.extend(segs)
        else:
            # Video, or a chunked long file: cut audio windows with ffmpeg.
            ensure_ffmpeg()
            for index, (start, end) in enumerate(bounds):
                wav = os.path.join(tmp_dir, f"chunk_{index:04d}.wav")
                created_files.append(wav)
                seg_duration = None if (len(bounds) == 1) else max(0.0, end - start)
                extract_audio(
                    media_path,
                    wav,
                    start=(start if len(bounds) > 1 else None),
                    duration=seg_duration,
                )
                segs, lang = _run_model(
                    model,
                    wav,
                    language,
                    beam_size,
                    vad_filter,
                    word_timestamps,
                    time_offset=(start if len(bounds) > 1 else 0.0),
                )
                if index == 0:
                    detected_language = lang
                # Drop segments that fall entirely inside the overlap of the
                # previous chunk to avoid duplicated lines.
                if index > 0 and chunk_overlap > 0:
                    cutoff = start + chunk_overlap * 0.5
                    segs = [s for s in segs if s.start >= cutoff]
                all_segments.extend(segs)

        all_segments.sort(key=lambda s: s.start)
        full_text = " ".join(s.text for s in all_segments if s.text).strip()
        total_duration = duration if duration else (all_segments[-1].end if all_segments else 0.0)

        return TranscriptResult(
            segments=all_segments,
            language=detected_language or (language or "unknown"),
            duration=float(total_duration),
            text=full_text,
            model_size=model_size,
        )
    finally:
        for f in created_files:
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def segments_to_paragraphs(segments: List[Segment], gap: float = 2.0) -> str:
    """Join segments into readable paragraphs, breaking on pauses longer than ``gap``."""
    paragraphs: List[str] = []
    current: List[str] = []
    prev_end: Optional[float] = None
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if prev_end is not None and (seg.start - prev_end) > gap and current:
            paragraphs.append(" ".join(current))
            current = []
        current.append(text)
        prev_end = seg.end
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)
