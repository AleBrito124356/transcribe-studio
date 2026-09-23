"""Auto chapters: detect topic shifts and emit YouTube-style timestamps.

Two strategies:

- ``detect_chapters_local`` — no API needed. Groups segments into fixed-time
  windows, measures lexical similarity (bag-of-words cosine) between adjacent
  windows, and starts a new chapter where similarity drops or a long silence
  occurs. Titles come from the chapter's most distinctive keywords.
- ``detect_chapters_nim`` — sends a condensed, timestamped transcript to NIM and
  asks for well-phrased chapter titles, then parses them back to seconds.

The local strategy is deterministic and dependency-free, which makes it easy to
test; NIM produces nicer titles when a key is available.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from .config import Segment, format_chapter_timestamp, parse_timestamp

# Small bilingual (EN/ES) stopword list — enough to keep keyword titles clean.
_STOPWORDS = set(
    """
    the a an and or but if then else of to in on at by for with without from into over under
    is are was were be been being do does did have has had will would can could should may might
    this that these those it its as so not no yes we you they he she i me my your our their them us
    about after again all also any because before between both during each few more most other some
    such than too very just now here there when where which who whom why how what
    el la los las un una unos unas y o pero si de del al a en con sin por para que se lo le les su sus
    es son era eran ser estar este esta estos estas eso esa aqui alli cuando donde como porque muy mas
    ya no si tambien todo todos toda todas entre cada algun alguna otro otra sobre desde hasta
    """.split()
)

_WORD_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)


@dataclass
class Chapter:
    start: float
    title: str


def _tokenize(text: str) -> List[str]:
    return [
        w.lower()
        for w in _WORD_RE.findall(text)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    ]


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokens_in_range(segments: List[Segment], start: float, end: float) -> List[str]:
    """All content tokens from segments that begin within ``[start, end)``."""
    tokens: List[str] = []
    for seg in segments:
        if start <= seg.start < end:
            tokens.extend(_tokenize(seg.text))
    return tokens


def _title_from_tokens(tokens: List[str], max_words: int = 4) -> str:
    if not tokens:
        return "Chapter"
    counts = Counter(tokens)
    top = [word for word, _ in counts.most_common(max_words)]
    return " ".join(w.capitalize() for w in top)


def detect_chapters_local(
    segments: List[Segment],
    window_seconds: float = 40.0,
    sim_threshold: float = 0.30,
    min_chapter_seconds: float = 60.0,
    silence_gap: float = 3.0,
) -> List[Chapter]:
    """Detect chapters from topic drift and long silences, no API required.

    Uses a TextTiling-style comparison: at each candidate boundary (a segment
    start), the bag-of-words of the preceding ``window_seconds`` is compared with
    the following ``window_seconds``. A low cosine similarity, or a silence longer
    than ``silence_gap``, marks a topic change — subject to a ``min_chapter_seconds``
    spacing so chapters never come too close together.
    """
    segments = [s for s in segments if s.text.strip()]
    if not segments:
        return []

    boundaries: List[float] = [0.0]
    last_boundary = 0.0
    for i in range(1, len(segments)):
        cand = segments[i].start
        if cand - last_boundary < min_chapter_seconds:
            continue
        before = Counter(_tokens_in_range(segments, cand - window_seconds, cand))
        after = Counter(_tokens_in_range(segments, cand, cand + window_seconds))
        if not before or not after:
            continue
        sim = _cosine(before, after)
        gap = segments[i].start - segments[i - 1].end
        if sim < sim_threshold or gap >= silence_gap:
            boundaries.append(cand)
            last_boundary = cand

    chapters: List[Chapter] = []
    end_of_media = segments[-1].end + 1.0
    for idx, start in enumerate(boundaries):
        next_start = boundaries[idx + 1] if idx + 1 < len(boundaries) else end_of_media
        tokens = _tokens_in_range(segments, start, next_start)
        chapters.append(
            Chapter(start=(0.0 if idx == 0 else start), title=_title_from_tokens(tokens))
        )
    return chapters


def _condense_transcript(segments: List[Segment], step_seconds: float = 30.0) -> str:
    """One timestamped line every ~step_seconds, to fit long media into a prompt."""
    lines: List[str] = []
    bucket_start: Optional[float] = None
    buffer: List[str] = []
    for seg in segments:
        if bucket_start is None:
            bucket_start = seg.start
        buffer.append(seg.text.strip())
        if seg.end - bucket_start >= step_seconds:
            lines.append(f"[{format_chapter_timestamp(bucket_start)}] {' '.join(buffer)}")
            bucket_start = None
            buffer = []
    if buffer and bucket_start is not None:
        lines.append(f"[{format_chapter_timestamp(bucket_start)}] {' '.join(buffer)}")
    return "\n".join(lines)


_CHAPTER_LINE = re.compile(r"^\s*\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?\s*[-–—]?\s*(.+?)\s*$")

_NIM_SYSTEM = (
    "You create YouTube-style chapters from a timestamped transcript. "
    "Return between 3 and 12 chapters that mark real topic changes. "
    "The first chapter MUST start at 00:00. "
    "Output one chapter per line in the exact format 'MM:SS Title' (or 'H:MM:SS Title' "
    "past an hour). Titles are 2-6 words, no numbering, no extra commentary."
)


def detect_chapters_nim(
    segments: List[Segment], nim_client, step_seconds: float = 30.0
) -> List[Chapter]:
    """Ask NIM for chapter titles, then parse them back into :class:`Chapter`."""
    condensed = _condense_transcript(segments, step_seconds)
    if not condensed:
        return []
    reply = nim_client.chat(
        [
            {"role": "system", "content": _NIM_SYSTEM},
            {"role": "user", "content": condensed},
        ],
        temperature=0.4,
        max_tokens=800,
    )
    chapters = parse_chapter_lines(reply)
    if not chapters:
        return detect_chapters_local(segments)
    if chapters[0].start > 0:
        chapters.insert(0, Chapter(start=0.0, title="Intro"))
    return chapters


def parse_chapter_lines(text: str) -> List[Chapter]:
    """Parse ``MM:SS Title`` / ``H:MM:SS Title`` lines into chapters."""
    chapters: List[Chapter] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if not line:
            continue
        match = _CHAPTER_LINE.match(line)
        if not match:
            continue
        ts, title = match.group(1), match.group(2).strip()
        try:
            start = parse_timestamp(ts)
        except ValueError:
            continue
        if title:
            chapters.append(Chapter(start=start, title=title))
    chapters.sort(key=lambda c: c.start)
    return chapters


def detect_chapters(
    segments: List[Segment],
    nim_client=None,
    warnings: Optional[List[str]] = None,
    **kwargs,
) -> List[Chapter]:
    """Use NIM when a client is supplied, otherwise the local heuristic.

    Any NIM failure degrades to the offline heuristic; when ``warnings`` is a
    list, the reason is appended to it so callers can report it.
    """
    if nim_client is not None:
        try:
            return detect_chapters_nim(segments, nim_client)
        except Exception as exc:
            if warnings is not None:
                warnings.append(
                    f"NIM chapters failed ({exc}); used the offline chapter heuristic instead."
                )
            return detect_chapters_local(segments, **kwargs)
    return detect_chapters_local(segments, **kwargs)


def chapters_to_markdown(chapters: List[Chapter], title: str = "Chapters") -> str:
    lines = [f"# {title}", ""]
    for ch in chapters:
        lines.append(f"{format_chapter_timestamp(ch.start)} {ch.title}")
    return "\n".join(lines).rstrip("\n") + "\n"


def chapters_to_text(chapters: List[Chapter]) -> str:
    """Plain ``00:00 Title`` lines, ready to paste into a video description."""
    return "\n".join(f"{format_chapter_timestamp(ch.start)} {ch.title}" for ch in chapters)
