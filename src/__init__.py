"""transcribe-studio: media -> transcript, subtitles, summary, chapters.

Local transcription with faster-whisper; summaries, chapters and translation via
free NVIDIA NIM. See the README for the full workflow.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (  # noqa: F401
    Segment,
    TranscriptResult,
    Word,
    format_chapter_timestamp,
    format_srt_timestamp,
    format_vtt_timestamp,
    load_env,
    parse_timestamp,
)

__all__ = [
    "__version__",
    "Segment",
    "Word",
    "TranscriptResult",
    "format_srt_timestamp",
    "format_vtt_timestamp",
    "format_chapter_timestamp",
    "parse_timestamp",
    "load_env",
]
