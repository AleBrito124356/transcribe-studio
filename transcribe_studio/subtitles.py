"""Build valid, well-timed SRT and VTT subtitles from transcript segments.

Handles the fiddly parts that separate usable subtitles from raw dumps:

- **word-accurate timing**: when Whisper word timestamps are present (they are
  by default), cues are built by walking the words, so every cue starts when
  its first word is spoken and ends when its last word is. Breaks prefer
  sentence ends, pauses and clause punctuation over arbitrary positions, and a
  cue never spans a long silence or runs longer than a few seconds;
- a proportional fallback for text without word timings (translations, edited
  or older transcripts): long segments are split with time allocated in
  proportion to characters;
- balanced two-line layout within a max width;
- reading-speed-aware minimum durations, strictly monotonic and
  non-overlapping cues even when Whisper's segments overlap;
- WebVTT-safe text (``&``, ``<``, ``>`` escaped; ``-->`` neutralised in SRT);
- optional speaker labels: ``[Speaker 1]`` at each turn in SRT and standard
  ``<v Speaker 1>`` voice tags in VTT;
- a lint report (overlaps, reading speed, line width) for the finished cues;
- optional translation through NIM that keeps every timestamp intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .config import Segment, Word, format_srt_timestamp, format_vtt_timestamp

DEFAULT_MAX_CHARS = 42
DEFAULT_MAX_LINES = 2
DEFAULT_MIN_DURATION = 0.7
DEFAULT_MAX_CPS = 17.0  # characters per second a viewer can comfortably read
DEFAULT_MIN_GAP = 0.04  # keep a small gap so players don't merge adjacent cues
DEFAULT_MAX_DURATION = 7.0  # no cue stays up longer than this while words are spoken
DEFAULT_PAUSE = 0.5  # a pause this long after a sentence ends starts a new cue
DEFAULT_LONG_PAUSE = 1.0  # never keep one cue on screen across a silence this long

_SENTENCE_END = re.compile(r"[.!?…。！？][\"'»”’)\]]*$")
_CLAUSE_END = re.compile(r"[,;:，、；：—–][\"'»”’)\]]*$")

# Line-break etiquette (EN/ES): break *before* these words...
_BREAK_BEFORE = frozenset(
    "and but or so because that which who when while if to of in on at for with from "
    "y e pero o u que porque cuando si de en con para por sin".split()
)
# ...and never leave these dangling at the end of a line.
_NO_BREAK_AFTER = frozenset(
    "a an the and or but of to in on at for with from my your our their his her its this "
    "el la los las un una unos unas y o de del al en con para por mi tu su sus".split()
)


def _bare(word: str) -> str:
    return word.strip(".,;:!?¡¿\"'()[]«»“”‘’").lower()


@dataclass
class Cue:
    """A single subtitle event: an index, a time window, and 1-N text lines.

    ``speaker`` is only set when speaker labels were requested. ``label`` is the
    visible ``[Speaker N]`` prefix already laid out at the start of ``lines[0]``
    on the first cue of each speaker turn (SRT shows it; VTT uses a voice tag).
    """

    index: int
    start: float
    end: float
    lines: List[str] = field(default_factory=list)
    speaker: Optional[str] = None
    label: Optional[str] = None

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def char_count(self) -> int:
        return sum(len(line) for line in self.lines)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ---------------------------------------------------------------------------
# Line layout
# ---------------------------------------------------------------------------
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


def balance_lines(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> List[str]:
    """Lay out text that fits in two lines as two *balanced* lines.

    Greedy wrapping gives "the quick brown fox jumps over the lazy" / "dog".
    Subtitle style guides ask for lines of similar length, broken after
    punctuation where possible, and a bottom-heavy shape when uneven. Text that
    needs one line, or more than two, is returned greedily wrapped.
    """
    greedy = wrap_text(text, max_chars)
    if len(greedy) != 2:
        return greedy
    words = text.split()
    best: Optional[Tuple[float, int, List[str]]] = None
    for i in range(1, len(words)):
        top, bottom = " ".join(words[:i]), " ".join(words[i:])
        if len(top) > max_chars or len(bottom) > max_chars:
            continue
        cost = abs(len(top) - len(bottom))
        if _SENTENCE_END.search(words[i - 1]) or _CLAUSE_END.search(words[i - 1]):
            cost -= 8  # a break after punctuation reads more naturally
        if _bare(words[i]) in _BREAK_BEFORE:
            cost -= 5  # "...tested it / and the restore..."
        if _bare(words[i - 1]) in _NO_BREAK_AFTER:
            cost += 5  # never leave "the", "and", "de"... dangling at a line end
        if len(top) > len(bottom):
            cost += 1  # prefer the bottom-heavy pyramid on ties
        candidate = (cost, -i, [top, bottom])
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return best[2] if best else greedy


def _group_lines(lines: List[str], max_lines: int) -> List[List[str]]:
    """Slice wrapped lines into groups of at most ``max_lines`` lines."""
    if not lines:
        return []
    return [lines[i : i + max_lines] for i in range(0, len(lines), max_lines)]


def _protect(label: Optional[str]) -> str:
    """A same-width, space-free stand-in so layout never splits ``[Speaker 1]``."""
    return "\x01" * len(label) if label else ""


def _layout(text: str, max_chars: int, max_lines: int, label: Optional[str] = None) -> List[str]:
    if label:
        text = text.replace(label, _protect(label), 1)
    lines = wrap_text(text, max_chars)
    if max_lines >= 2 and len(lines) == 2:
        lines = balance_lines(text, max_chars)
    if label:
        lines = [line.replace(_protect(label), label, 1) for line in lines]
    return lines


def _fits(text: str, n_tokens: int, max_chars: int, max_lines: int) -> bool:
    if n_tokens <= 1:
        return True  # a single word always gets its own cue, however long
    lines = wrap_text(text, max_chars)
    return len(lines) <= max_lines and all(len(line) <= max_chars for line in lines)


# ---------------------------------------------------------------------------
# Cue building
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return "".join(text.split()).lower()


def _timed_tokens(seg: Segment) -> Optional[List[Tuple[float, float, str]]]:
    """(start, end, raw_text) per word, or None when the words can't be trusted.

    Raw texts keep Whisper's own leading spaces so languages written without
    spaces join correctly. If the segment text was edited after transcription
    (a fixed typo in transcript.json), the edited tokens are mapped onto the
    word timings when the token count still matches; otherwise the segment
    falls back to proportional timing so the edit is never lost.
    """
    words = [w for w in seg.words if w.word and w.word.strip()]
    if not words:
        return None
    timed: List[Tuple[float, float, str]] = []
    last_start = float("-inf")
    for w in words:
        start = max(float(w.start), last_start)  # keep word starts monotonic
        end = max(float(w.end), start)
        timed.append((start, end, w.word))
        last_start = start
    if _norm("".join(w.word for w in words)) == _norm(seg.text):
        return timed
    tokens = seg.text.split()
    if len(tokens) == len(timed):
        return [(s, e, (" " if i else "") + tok) for i, ((s, e, _), tok) in enumerate(zip(timed, tokens))]
    return None


def _join(tokens: List[Tuple[float, float, str]]) -> str:
    return "".join(t[2] for t in tokens).strip()


def _best_break(cur: List[Tuple[float, float, str]], pause: float) -> int:
    """How many leading tokens of ``cur`` to emit as a cue when it must split.

    Prefers, in order: the latest sentence end, the latest pause of at least
    ``pause`` seconds, the latest clause punctuation — each only if at least
    30% of the text goes before it, so no tiny orphan cue is left behind.
    Otherwise everything accumulated so far is emitted.
    """
    total = len(_join(cur)) or 1
    best_class, best_i = 0, len(cur)
    for i in range(len(cur) - 1, 0, -1):
        before = _join(cur[:i])
        if len(before) < 0.3 * total:
            break
        word = cur[i - 1][2].strip()
        gap = cur[i][0] - cur[i - 1][1]
        if _SENTENCE_END.search(word):
            klass = 3
        elif gap >= pause:
            klass = 2
        elif _CLAUSE_END.search(word):
            klass = 1
        else:
            continue
        if klass > best_class:
            best_class, best_i = klass, i
    return best_i


def _word_cues(
    tokens: List[Tuple[float, float, str]],
    label: Optional[str],
    max_chars: int,
    max_lines: int,
    max_duration: float,
    pause: float,
    long_pause: float,
) -> List[Tuple[float, float, str, bool]]:
    """Group timed tokens into (start, end, text, carries_label) cue drafts."""
    drafts: List[Tuple[float, float, str, bool]] = []
    cur: List[Tuple[float, float, str]] = []

    def text_of(chunk, with_label):
        text = _join(chunk)
        return f"{label} {text}" if (with_label and label) else text

    def fits(chunk):
        with_label = not drafts and bool(label)
        text = text_of(chunk, with_label)
        if with_label:
            text = text.replace(label, _protect(label), 1)
        return _fits(text, len(chunk) + (1 if with_label else 0), max_chars, max_lines) and (
            chunk[-1][1] - chunk[0][0] <= max_duration or len(chunk) == 1
        )

    def flush(chunk):
        if chunk:
            with_label = not drafts and bool(label)
            drafts.append((chunk[0][0], chunk[-1][1], text_of(chunk, with_label), with_label))

    for tok in tokens:
        if cur:
            prev = cur[-1]
            gap = tok[0] - prev[1]
            if gap >= long_pause or (gap >= pause and _SENTENCE_END.search(prev[2].strip())):
                flush(cur)
                cur = []
            elif not fits(cur + [tok]):
                k = _best_break(cur, pause)
                flush(cur[:k])
                cur = cur[k:]
                if cur and not fits(cur + [tok]):
                    flush(cur)
                    cur = []
        cur.append(tok)
    flush(cur)
    return drafts


def build_cues(
    segments: List[Segment],
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_cps: float = DEFAULT_MAX_CPS,
    min_gap: float = DEFAULT_MIN_GAP,
    speaker_labels: bool = False,
    use_words: bool = True,
    max_duration: float = DEFAULT_MAX_DURATION,
    pause: float = DEFAULT_PAUSE,
    long_pause: float = DEFAULT_LONG_PAUSE,
) -> List[Cue]:
    """Turn segments into display-ready cues.

    Segments with word timestamps are cut on word boundaries and timed from
    their words (``use_words=False`` forces the old behaviour). Segments without
    them are split into several cues, each covering a slice of the segment's
    duration proportional to its character count. Timing is then adjusted so no
    cue is shorter than the reading-speed floor and cues never overlap.

    With ``speaker_labels``, cues carry their segment's speaker; the first cue
    of each turn gets a visible ``[Speaker N]`` prefix (width is reserved for it).
    """
    raw: List[dict] = []
    previous_speaker: Optional[str] = None
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        speaker = seg.speaker if speaker_labels else None
        label = f"[{speaker}]" if (speaker and speaker != previous_speaker) else None
        if speaker:
            previous_speaker = speaker

        tokens = _timed_tokens(seg) if use_words else None
        if tokens:
            for start, end, draft, has_label in _word_cues(
                tokens, label, max_chars, max_lines, max_duration, pause, long_pause
            ):
                lines = _layout(draft, max_chars, max_lines, label if has_label else None)
                raw.append({
                    "start": start, "end": end, "lines": lines,
                    "chars": sum(len(line) for line in lines),
                    "speaker": speaker, "label": label if has_label else None,
                })
            continue

        # Proportional fallback (no usable word timings).
        full = f"{_protect(label)} {text}" if label else text
        groups = _group_lines(wrap_text(full, max_chars), max_lines)
        if not groups:
            continue
        char_counts = [sum(len(line) for line in g) for g in groups]
        total_chars = sum(char_counts) or 1
        seg_duration = max(0.0, seg.end - seg.start)
        cursor = seg.start
        for n, (group, chars) in enumerate(zip(groups, char_counts)):
            portion = seg_duration * (chars / total_chars)
            start = cursor
            end = cursor + portion
            cursor = end
            lines = _layout(" ".join(group), max_chars, max_lines)
            if label and n == 0:
                lines = [line.replace(_protect(label), label, 1) for line in lines]
            raw.append({
                "start": start, "end": end, "lines": lines, "chars": chars,
                "speaker": speaker, "label": label if n == 0 else None,
            })

    return _finalize_cues(raw, min_duration, max_cps, min_gap)


def _finalize_cues(
    raw: List[dict], min_duration: float, max_cps: float, min_gap: float
) -> List[Cue]:
    """Apply the reading-speed floor and make cues strictly sequential.

    Each cue lasts at least ``min_duration`` and long enough to read at
    ``max_cps``, but is cut ``min_gap`` before the next cue starts. When the
    source segments overlap (Whisper does this), a cue's start is pushed past
    the previous cue's end, so the output is always monotonic and overlap-free.
    """
    raw = sorted(raw, key=lambda r: r["start"])  # stable: keeps source order on ties
    min_show = 0.1
    cues: List[Cue] = []
    prev_end: Optional[float] = None
    for i, item in enumerate(raw):
        start = max(0.0, item["start"])
        if prev_end is not None and start < prev_end + min_gap:
            start = prev_end + min_gap
        end = max(item["end"], start)
        chars = item["chars"]
        # Reading-speed-aware floor: at least min_duration, and at least the time
        # needed to read the text at max_cps characters per second.
        needed = max(min_duration, chars / max_cps) if chars else min_duration
        if end - start < needed:
            end = start + needed
        # Never overlap the next cue (but always stay visible a little while).
        if i + 1 < len(raw):
            next_start = raw[i + 1]["start"]
            if end > next_start - min_gap:
                end = max(start + min_show, next_start - min_gap)
        cues.append(Cue(index=len(cues) + 1, start=start, end=end, lines=item["lines"],
                        speaker=item.get("speaker"), label=item.get("label")))
        prev_end = end
    return cues


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------
def _srt_safe(line: str) -> str:
    # A literal "-->" inside cue text is read as a timing line by many parsers.
    return line.replace("-->", "->")


def _vtt_escape(text: str) -> str:
    """Escape cue text for WebVTT: ``&``, ``<`` and ``>`` become entities.

    This also neutralises ``-->`` (it becomes ``--&gt;``), which the WebVTT spec
    forbids inside cue payloads.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_srt(cues: List[Cue]) -> str:
    """Serialize cues as SubRip (.srt)."""
    out: List[str] = []
    for cue in cues:
        out.append(str(cue.index))
        out.append(f"{format_srt_timestamp(cue.start)} --> {format_srt_timestamp(cue.end)}")
        out.extend(_srt_safe(line) for line in cue.lines)
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def _vtt_lines(cue: Cue) -> List[str]:
    lines = list(cue.lines)
    if cue.label and lines and lines[0].startswith(cue.label):
        first = lines[0][len(cue.label):].strip()
        lines = ([first] if first else []) + lines[1:]
    return lines


def to_vtt(cues: List[Cue]) -> str:
    """Serialize cues as WebVTT (.vtt), escaping text and tagging voices."""
    out: List[str] = ["WEBVTT", ""]
    for cue in cues:
        out.append(f"{format_vtt_timestamp(cue.start)} --> {format_vtt_timestamp(cue.end)}")
        lines = [_vtt_escape(line) for line in _vtt_lines(cue)]
        if cue.speaker and lines:
            lines[0] = f"<v {_vtt_escape(cue.speaker)}>{lines[0]}"
        out.extend(lines)
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


def write_captions(
    segments: List[Segment],
    srt_path: str | Path,
    vtt_path: str | Path,
    **kwargs,
) -> "CaptionReport":
    """Build cues once, write both formats, and return their lint report."""
    cues = build_cues(segments, **kwargs)
    Path(srt_path).write_text(to_srt(cues), encoding="utf-8")
    Path(vtt_path).write_text(to_vtt(cues), encoding="utf-8")
    return validate_cues(
        cues,
        max_chars=kwargs.get("max_chars", DEFAULT_MAX_CHARS),
        max_lines=kwargs.get("max_lines", DEFAULT_MAX_LINES),
        max_cps=kwargs.get("max_cps", DEFAULT_MAX_CPS),
    )


# ---------------------------------------------------------------------------
# Lint
# ---------------------------------------------------------------------------
@dataclass
class CaptionReport:
    """Quality checks over a finished set of cues."""

    cues: int = 0
    overlaps: int = 0
    out_of_order: int = 0
    fast_cues: int = 0
    max_cps: float = 0.0
    wide_lines: int = 0
    tall_cues: int = 0
    empty_cues: int = 0
    limits: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when nothing would break a player or a style checker.

        Reading speed is reported but not fatal: very fast speech cannot always
        be slowed down without drifting away from the audio.
        """
        return not (self.overlaps or self.out_of_order or self.wide_lines
                    or self.tall_cues or self.empty_cues)

    def summary(self) -> str:
        cps_limit = self.limits.get("max_cps", DEFAULT_MAX_CPS)
        width = self.limits.get("max_chars", DEFAULT_MAX_CHARS)
        parts = [
            f"{self.cues} cues",
            f"{self.overlaps} overlaps",
            f"{self.fast_cues} above {cps_limit:g} chars/s (max {self.max_cps:.1f})",
            f"{self.wide_lines} lines over {width} chars",
        ]
        if self.out_of_order:
            parts.append(f"{self.out_of_order} out of order")
        if self.tall_cues:
            parts.append(f"{self.tall_cues} cues over {self.limits.get('max_lines')} lines")
        return " | ".join(parts)


def validate_cues(
    cues: List[Cue],
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
    max_cps: float = DEFAULT_MAX_CPS,
) -> CaptionReport:
    """Check cues for overlaps, ordering, reading speed and layout.

    A single word longer than ``max_chars`` on its own line is not counted as
    a wide line, since it cannot be wrapped.
    """
    report = CaptionReport(cues=len(cues),
                           limits={"max_chars": max_chars, "max_lines": max_lines, "max_cps": max_cps})
    for i, cue in enumerate(cues):
        if not any(line.strip() for line in cue.lines) or cue.end <= cue.start:
            report.empty_cues += 1
        if i:
            prev = cues[i - 1]
            if cue.start < prev.start:
                report.out_of_order += 1
            if cue.start < prev.end:
                report.overlaps += 1
        if len(cue.lines) > max_lines:
            report.tall_cues += 1
        report.wide_lines += sum(1 for line in cue.lines if len(line) > max_chars and " " in line)
        if cue.duration > 0:
            cps = cue.char_count / cue.duration
            report.max_cps = max(report.max_cps, cps)
            if cps > max_cps + 1e-9:
                report.fast_cues += 1
    return report


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
