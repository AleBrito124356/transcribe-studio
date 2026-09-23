"""Build valid SRT and VTT subtitles from transcript segments.

Handles the fiddly parts that separate usable subtitles from raw dumps:
- line wrapping to a max character width, capped at a small number of lines;
- splitting an over-long segment into several cues, with time allocated in
  proportion to characters;
- reading-speed-aware minimum durations so cues never flash by faster than a
  human can read, while never overlapping the next cue;
- optional translation through NIM that keeps every timestamp intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .config import Segment, format_srt_timestamp, format_vtt_timestamp

DEFAULT_MAX_CHARS = 42
DEFAULT_MAX_LINES = 2
DEFAULT_MIN_DURATION = 0.7
DEFAULT_MAX_CPS = 17.0  # characters per second a viewer can comfortably read
DEFAULT_MIN_GAP = 0.04  # keep a small gap so players don't merge adjacent cues


@dataclass
class Cue:
    """A single subtitle event: an index, a time window, and 1-N text lines."""

    index: int
    start: float
    end: float
    lines: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def char_count(self) -> int:
        return sum(len(line) for line in self.lines)


def wrap_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """Greedy word-wrap into lines no wider than ``max_chars``.

    Words longer than ``max_chars`` are placed on their own line rather than
    hard-split, which keeps real words intact at the cost of one wide line.
    """
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    current = ""
    for word in words:
        if not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _group_lines(lines: List[str], max_lines: int) -> List[List[str]]:
    """Slice wrapped lines into groups of at most ``max_lines`` lines."""
    if not lines:
        return []
    return [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]


def build_cues(
    segments: List[Segment],
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_cps: float = DEFAULT_MAX_CPS,
    min_gap: float = DEFAULT_MIN_GAP,
) -> List[Cue]:
    """Turn segments into display-ready cues.

    Long segments are split into several cues, each covering a slice of the
    segment's duration proportional to its character count. Timing is then
    adjusted so no cue is shorter than the reading-speed floor, and no cue
    overlaps the next one.
    """
    raw: List[dict] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        lines = wrap_text(text, max_chars)
        groups = _group_lines(lines, max_lines)
        if not groups:
            continue
        char_counts = [sum(len(line) for line in g) for g in groups]
        total_chars = sum(char_counts) or 1
        seg_duration = max(0.0, seg.end - seg.start)
        cursor = seg.start
        for group, chars in zip(groups, char_counts):
            portion = seg_duration * (chars / total_chars)
            start = cursor
            end = cursor + portion
            cursor = end
            raw.append({"start": start, "end": end, "lines": group, "chars": chars})

    return _finalize_cues(raw, min_duration, max_cps, min_gap)


def _finalize_cues(
    raw: List[dict], min_duration: float, max_cps: float, min_gap: float
) -> List[Cue]:
    cues: List[Cue] = []
    for i, item in enumerate(raw):
        start = item["start"]
        end = item["end"]
        chars = item["chars"]
        # Reading-speed-aware floor: at least min_duration, and at least the time
        # needed to read the text at max_cps characters per second.
        needed = max(min_duration, chars / max_cps) if chars else min_duration
        if end - start < needed:
            end = start + needed
        # Never overlap the next cue.
        if i + 1 < len(raw):
            next_start = raw[i + 1]["start"]
            if end > next_start - min_gap:
                end = max(start + 0.1, next_start - min_gap)
        cues.append(Cue(index=len(cues) + 1, start=start, end=end, lines=item["lines"]))
    return cues


def to_srt(cues: List[Cue]) -> str:
    """Serialize cues as SubRip (.srt)."""
    out: List[str] = []
    for cue in cues:
        out.append(str(cue.index))
        out.append(f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(cue.end)}")
        out.append(cue.text)
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def to_vtt(cues: List[Cue]) -> str:
    """Serialize cues as WebVTT (.vtt)."""
    out: List[str] = ["WEBVTT", ""]
    for cue in cues:
        out.append(f"{format_vtt_timestamp(cue.start)} --> {format_vtt_timestamp(cue.end)}")
        out.append(cue.text)
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def write_srt(segments: List[Segment], path: str | Path, **kwargs) -> str:
    cues = build_cues(segments, **kwargs)
    content = to_srt(cues)
    Path(path).write_text(content, encoding="utf-8")
    return str(path)


def write_vtt(segments: List[Segment], path: str | Path, **kwargs) -> str:
    cues = build_cues(segments, **kwargs)
    content = to_vtt(cues)
    Path(path).write_text(content, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Translation (NIM) — timestamps are preserved exactly.
# ---------------------------------------------------------------------------
_TRANSLATE_SYSTEM = (
    "You are a professional subtitle translator. Translate each numbered line "
    "into {target}. Keep the meaning natural and concise for on-screen reading. "
    "Return ONLY a JSON array of strings, one per input line, in the same order "
    "and with the same count. Do not add, merge, or drop lines."
)


def _extract_json_array(text: str) -> Optional[list]:
    import json

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, list) else None


def _translate_batch(texts: List[str], target_lang: str, nim_client) -> List[str]:
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    system = _TRANSLATE_SYSTEM.format(target=target_lang)
    reply = nim_client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": numbered},
        ],
        temperature=0.2,
        max_tokens=min(4096, 200 + sum(len(t) for t in texts)),
    )
    parsed = _extract_json_array(reply)
    if parsed is not None and len(parsed) == len(texts):
        return [str(x) for x in parsed]
    # Fall back to translating each line on its own if the batch shape is wrong.
    result: List[str] = []
    for t in texts:
        single = nim_client.chat(
            [
                {
                    "role": "system",
                    "content": f"Translate the user's text into {target_lang}. "
                    "Return only the translation, no quotes or notes.",
                },
                {"role": "user", "content": t},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        result.append(single.strip())
    return result


def translate_segments(
    segments: List[Segment],
    target_lang: str,
    nim_client,
    batch_size: int = 40,
) -> List[Segment]:
    """Translate segment text into ``target_lang`` while keeping timestamps.

    Word-level timestamps are dropped because they no longer align after
    translation; segment start/end and speaker labels are preserved.
    """
    translated: List[Segment] = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        texts = [s.text for s in batch]
        outputs = _translate_batch(texts, target_lang, nim_client)
        for seg, new_text in zip(batch, outputs):
            translated.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    text=new_text.strip(),
                    words=[],
                    speaker=seg.speaker,
                )
            )
    return translated
