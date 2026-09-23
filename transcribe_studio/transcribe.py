"""Local speech-to-text with faster-whisper, plus ffmpeg audio extraction.

Design goals:
- Import cleanly without faster-whisper installed (it is imported lazily inside
  the functions that need it), so the pure helpers and the test-suite run anywhere.
- Handle both audio and video: video files have their audio extracted with ffmpeg
  first, with a friendly message if ffmpeg is missing.
- Support long files via optional chunking: overlapping windows, timestamp
  offsetting, word-level de-duplication at the overlap midpoints, and the
  auto-detected language locked in after the first window.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

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


# Default seconds shared by consecutive windows when --chunk-length is used.
DEFAULT_CHUNK_OVERLAP = 2.0


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
    on_segment: Optional[Callable[[float], None]] = None,
) -> Tuple[List[Segment], str]:
    """Run one decode pass and return offset segments plus detected language."""
    try:
        segments_iter, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )
    except Exception as exc:
        # faster-whisper decodes plain audio itself with PyAV; report an
        # unreadable file the same way as an ffmpeg extraction failure.
        if type(exc).__module__.split(".")[0] == "av":
            first_line = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
            raise MediaDecodeError(
                f"could not decode audio from {Path(audio_path).name}: {first_line[:200]}"
            ) from exc
        raise
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
        if on_segment is not None:
            on_segment(out[-1].end)
    return out, getattr(info, "language", None) or (language or "")


def chunk_cut_points(bounds: List[Tuple[float, float]]) -> List[float]:
    """Where ownership passes from one chunk to the next: the overlap midpoints.

    Chunk ``i`` owns speech that starts in ``[cut[i-1], cut[i])``. Cutting in the
    middle of the overlap means both neighbours heard that audio with context on
    either side, so the words kept on each side were decoded from full audio,
    not from a window edge that sliced a word in half.
    """
    return [(bounds[i + 1][0] + bounds[i][1]) / 2.0 for i in range(len(bounds) - 1)]


def keep_between(segments: List[Segment], lo: float, hi: float) -> List[Segment]:
    """Keep what starts inside ``[lo, hi)``, trimming segments at word level.

    A segment that straddles a cut point is split using its word timestamps, so
    the words before the cut come from one chunk and the words after it from the
    next — nothing is lost and nothing is transcribed twice. Segments without
    word timestamps are kept or dropped whole by their start time.
    """
    kept: List[Segment] = []
    for seg in segments:
        if not seg.words:
            if lo <= seg.start < hi:
                kept.append(seg)
            continue
        inside = [w for w in seg.words if lo <= w.start < hi]
        if not inside:
            continue
        if len(inside) == len(seg.words):
            kept.append(seg)
            continue
        first_is_original = inside[0] is seg.words[0]
        last_is_original = inside[-1] is seg.words[-1]
        kept.append(
            Segment(
                start=seg.start if first_is_original else inside[0].start,
                end=seg.end if last_is_original else inside[-1].end,
                text="".join(w.word for w in inside).strip(),
                words=inside,
                speaker=seg.speaker,
            )
        )
    return kept


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
    chunk_overlap: float = DEFAULT_CHUNK_OVERLAP,
    model=None,
    progress: Optional[Callable[[float, Optional[float]], None]] = None,
) -> TranscriptResult:
    """Transcribe an audio or video file into a :class:`TranscriptResult`.

    Parameters
    ----------
    model_size:
        faster-whisper size: ``tiny``, ``base``, ``small``, ``medium``,
        ``large-v3`` (or the distil-* variants), or a local model directory.
    language:
        Force a language (e.g. ``"en"``/``"es"``) or leave ``None`` to auto-detect.
        When auto-detecting a chunked file, the language found in the first chunk
        is locked in for the rest, so one quiet or noisy window cannot switch
        the transcript to another language halfway through.
    device / compute_type:
        ``"auto"``/``"cpu"``/``"cuda"`` and ``int8``/``int8_float16``/``float16``.
    chunk_length:
        When > 0 and the file is longer than one chunk, decode in ffmpeg-cut
        windows and stitch timestamps back together. Requires ffprobe.
    chunk_overlap:
        Seconds shared by consecutive windows. Words are de-duplicated at the
        middle of each overlap (see :func:`keep_between`). Must be smaller than
        ``chunk_length``.
    model:
        A preloaded ``WhisperModel`` to reuse across a batch. Loaded on demand
        otherwise.
    progress:
        Optional callback ``progress(seconds_done, total_seconds_or_None)``,
        called as segments are decoded.
    """
    media_path = str(media_path)
    if not Path(media_path).exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")
    overlap = max(0.0, float(chunk_overlap or 0.0)) if chunk_length and chunk_length > 0 else 0.0
    if chunk_length and chunk_length > 0 and overlap >= chunk_length:
        raise ValueError(
            f"chunk_overlap ({overlap:g}s) must be smaller than chunk_length ({chunk_length:g}s)"
        )

    if model is None:
        model = _load_model(model_size, device, compute_type)

    tmp_dir = tempfile.mkdtemp(prefix="transcribe_studio_")
    created_files: List[str] = []
    try:
        duration = probe_duration(media_path)
        if chunk_length and duration:
            bounds = chunk_boundaries(duration, chunk_length, overlap)
        else:
            bounds = [(0.0, duration or 0.0)]

        def report(done: float) -> None:
            if progress is not None:
                progress(min(done, duration) if duration else done, duration)

        all_segments: List[Segment] = []
        detected_language = language or ""

        if len(bounds) == 1 and not is_video(media_path):
            # Simplest path: hand the file straight to faster-whisper.
            segs, detected_language = _run_model(
                model, media_path, language, beam_size, vad_filter, word_timestamps,
                on_segment=report,
            )
            all_segments.extend(segs)
        else:
            # Video, or a chunked long file: cut audio windows with ffmpeg.
            ensure_ffmpeg()
            chunked = len(bounds) > 1
            cuts = chunk_cut_points(bounds)
            chunk_language = language
            for index, (start, end) in enumerate(bounds):
                wav = os.path.join(tmp_dir, f"chunk_{index:04d}.wav")
                created_files.append(wav)
                extract_audio(
                    media_path,
                    wav,
                    start=(start if chunked else None),
                    duration=(max(0.0, end - start) if chunked else None),
                )
                segs, lang = _run_model(
                    model,
                    wav,
                    chunk_language,
                    beam_size,
                    vad_filter,
                    word_timestamps,
                    time_offset=(start if chunked else 0.0),
                    on_segment=report,
                )
                if index == 0:
                    detected_language = lang
                    if chunk_language is None and lang:
                        chunk_language = lang  # lock auto-detected language for later chunks
                if chunked:
                    lo = cuts[index - 1] if index > 0 else float("-inf")
                    hi = cuts[index] if index < len(cuts) else float("inf")
                    segs = keep_between(segs, lo, hi)
                all_segments.extend(segs)
                try:
                    os.remove(wav)  # free disk space as we go on long files
                except OSError:
                    pass

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
