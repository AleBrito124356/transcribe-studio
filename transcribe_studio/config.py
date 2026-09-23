"""Shared data structures, environment loading, and time-formatting helpers.

Everything in this module is dependency-free (standard library only) so it can be
imported without faster-whisper, requests, or any network access. The heavier
modules (transcribe/summarize) import their optional dependencies lazily.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# NVIDIA NIM convention
# ---------------------------------------------------------------------------
# These are the *defaults*. The effective values are resolved from the
# environment each time a client is built (see nim_base_url()/nim_model()), so
# a .env loaded after import — which is exactly what the CLI does — still wins.
DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_MODEL = "meta/llama-3.3-70b-instruct"
NIM_BASE_URL = DEFAULT_NIM_BASE_URL  # backwards-compatible alias (a default, not the live value)
API_KEY_ENV = "NVIDIA_API_KEY"


def nim_base_url() -> str:
    """The OpenAI-compatible base URL, read from ``NIM_BASE_URL`` right now."""
    return (os.environ.get("NIM_BASE_URL") or DEFAULT_NIM_BASE_URL).strip().rstrip("/")


def nim_model() -> str:
    """The chat model, read from ``NIM_MODEL`` right now."""
    return (os.environ.get("NIM_MODEL") or DEFAULT_NIM_MODEL).strip()


def env_number(name: str, default: float) -> float:
    """Read a number from the environment, ignoring missing or garbage values."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

# Media extensions we recognise. Anything with a video container gets its audio
# extracted with ffmpeg before transcription.
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".wmv", ".mpg", ".mpeg"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Word:
    """A single word with its own start/end timestamp (from word_timestamps)."""

    start: float
    end: float
    word: str


@dataclass
class Segment:
    """A contiguous span of transcribed speech."""

    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class TranscriptResult:
    """The full output of a transcription run."""

    segments: List[Segment]
    language: str
    duration: float
    text: str
    model_size: str = ""
    # How speaker labels were produced ("pause-heuristic", "pyannote", ...).
    # Empty when the segments carry no speakers.
    diarization: str = ""

    def to_dict(self) -> dict:
        data = {
            "language": self.language,
            "duration": self.duration,
            "model_size": self.model_size,
            "text": self.text,
        }
        if self.diarization:
            data["diarization"] = self.diarization
        data["segments"] = [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "speaker": s.speaker,
                "words": [asdict(w) for w in s.words],
            }
            for s in self.segments
        ]
        return data

    @property
    def has_speakers(self) -> bool:
        return any(s.speaker for s in self.segments)

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptResult":
        if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
            raise InvalidTranscript(
                "not a transcript: expected a JSON object with a 'segments' list "
                "(the transcript.json that transcribe-studio writes)"
            )
        try:
            segments = [
                Segment(
                    start=float(s["start"]),
                    end=float(s["end"]),
                    text=str(s.get("text", "")),
                    speaker=s.get("speaker"),
                    # Tolerate extra keys (e.g. faster-whisper's "probability").
                    words=[
                        Word(start=float(w["start"]), end=float(w["end"]), word=str(w.get("word", "")))
                        for w in s.get("words") or []
                    ],
                )
                for s in data["segments"]
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidTranscript(f"malformed segment in transcript: {exc!r}") from None
        text = data.get("text")
        if not text:
            text = " ".join(s.text.strip() for s in segments if s.text.strip())
        duration = float(data.get("duration") or 0.0)
        if not duration and segments:
            duration = max(s.end for s in segments)
        return cls(
            segments=segments,
            language=data.get("language", "") or "",
            duration=duration,
            text=text,
            model_size=data.get("model_size", "") or "",
            diarization=data.get("diarization", "") or "",
        )


class InvalidTranscript(ValueError):
    """Raised when a JSON file is not a transcript this tool can read."""


def save_json(result: TranscriptResult, path: str | os.PathLike) -> None:
    Path(path).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | os.PathLike) -> TranscriptResult:
    """Load a transcript.json, raising :class:`InvalidTranscript` on bad input."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise InvalidTranscript(f"{Path(path).name} is not valid JSON ({exc.msg} at line {exc.lineno})") from None
    try:
        return TranscriptResult.from_dict(data)
    except InvalidTranscript as exc:
        raise InvalidTranscript(f"{Path(path).name}: {exc}") from None


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float, decimal_sep: str = ",") -> str:
    """Format ``seconds`` as ``HH:MM:SS<sep>mmm`` (SRT uses ``,``, VTT uses ``.``)."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    total_millis = int(round(seconds * 1000))
    hours, total_millis = divmod(total_millis, 3_600_000)
    minutes, total_millis = divmod(total_millis, 60_000)
    secs, millis = divmod(total_millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_sep}{millis:03d}"


def format_srt_timestamp(seconds: float) -> str:
    return format_timestamp(seconds, ",")


def format_vtt_timestamp(seconds: float) -> str:
    return format_timestamp(seconds, ".")


def format_chapter_timestamp(seconds: float) -> str:
    """YouTube-style timestamp. ``M:SS`` under an hour, ``H:MM:SS`` above."""
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def parse_timestamp(text: str) -> float:
    """Parse ``MM:SS`` or ``HH:MM:SS`` (optionally ``,mmm``/``.mmm``) into seconds."""
    text = text.strip().replace(",", ".")
    if not text:
        raise ValueError("empty timestamp")
    parts = text.split(":")
    if len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError(f"unrecognised timestamp: {text!r}")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ---------------------------------------------------------------------------
# Minimal .env loader (no python-dotenv dependency)
# ---------------------------------------------------------------------------
def load_env(path: str | os.PathLike = ".env", override: bool = False) -> None:
    """Load ``KEY=VALUE`` pairs from a .env file into ``os.environ``.

    Existing environment variables win unless ``override`` is True. Silently does
    nothing if the file is absent, so it is safe to call unconditionally.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
