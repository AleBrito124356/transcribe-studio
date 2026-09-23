"""Auto chapters: detect topic shifts and emit YouTube-style timestamps.

Two strategies:

- ``detect_chapters_local`` — no API needed. Every segment start is a
  candidate boundary. The words spoken in the ``window_seconds`` before and
  after it are compared as TF-IDF vectors, where the IDF is computed over
  fixed-length blocks of the whole recording, so a word that is said all the
  time (the show's name, "podcast", a verbal tic) carries no weight and cannot
  hide a real change of topic. Low similarity, or a long silence, marks a
  boundary; the strongest boundaries win, spaced at least
  ``min_chapter_seconds`` apart. Titles are the terms most *distinctive* of
  each chapter compared with the rest of the recording, in their original
  casing, and never repeat.
- ``detect_chapters_nim`` — sends a condensed, timestamped transcript to NIM and
  asks for well-phrased chapter titles, then parses them back to seconds.

Both go through :func:`normalize_chapters`, which enforces what YouTube needs
for chapters to show up: the first at 00:00, every chapter at least 10 s long,
nothing past the end of the media, unique titles, and at least three chapters
when the media is long enough to hold them.

Everything is deterministic and uses only the standard library. Stopwords and
filler words are bilingual (English/Spanish) and accent-insensitive, so
"también", "está" or "o sea" never become titles.
"""

from __future__ import annotations

import math
import re
import unicodedata
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .config import Segment, format_chapter_timestamp, parse_timestamp

YOUTUBE_MIN_CHAPTER_SECONDS = 10.0
YOUTUBE_MIN_CHAPTERS = 3


@dataclass
class Chapter:
    start: float
    title: str


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------
def fold(text: str) -> str:
    """Lowercase and strip accents: "También" -> "tambien", "Ñandú" -> "nandu"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


# Written unaccented; everything is compared after fold().
_STOPWORDS_EN = """
a about above after again against all almost also although always am among an and another any anybody
anyone anything anyway are around as at back be became because become been before being below between
both but by can cannot could did do does doing done down during each either else enough even ever every
everybody everyone everything few for from further get gets getting go goes going gone got gotten had
has have having he her here hers herself him himself his how however i if in into is it its itself
just least less let lets like made make makes making many may maybe me might mine more most much must
my myself neither never no nobody none nor not nothing now of off often on once one only onto or other
others otherwise our ours ourselves out over own per perhaps quite rather really right said same say
says see seem seemed seems several shall she should since so some somebody someone something sometimes
somewhat still such sure than that the their theirs them themselves then there therefore these they
thing things think this those though through thus to together too toward towards under until up upon
us use used using very via want wanted wants was way ways we well went were what whatever when whenever
where whether which while who whoever whole whom whose why will with within without would yes yet you
your yours yourself yourselves
don didn doesn isn wasn aren weren couldn wouldn shouldn won hasn haven hadn ain ll ve re
"""
_FILLERS_EN = """
yeah yep yup okay ok oh uh um umm hmm huh ah er erm mm gonna gotta wanna kinda sorta basically
actually literally honestly totally absolutely obviously definitely probably pretty stuff kind sort
lot lots bit mean means know knew guess guys guy thanks thank hey hello hi welcome today anyway
cool awesome great good nice fine alright bye look listen talk talking said saying tell told stuff
first second third next last new old big small little
two three four five six seven eight nine ten eleven twelve twenty hundred thousand
show episode everybody everyone folks
"""
_STOPWORDS_ES = """
a al algo alguien algun alguna algunas alguno algunos ante antes aqui asi aun aunque bajo bien cada casi
como con contra cual cuales cualquier cuando cuanto de del desde donde dos el ella ellas ello ellos en
entonces entre era eramos eran eres es esa esas ese eso esos esta estaba estado estamos estan estar
estas este esto estos estoy fue fueron fui ha hace hacer hacia han has hasta hay he hemos la las le les
lo los mas me mi mia mias mio mios mis mismo mucha muchas mucho muchos muy nada ni ningun ninguna no
nos nosotras nosotros nuestra nuestras nuestro nuestros nunca o os otra otras otro otros para pero poco
por porque pues que quien quienes se sea segun ser si sido siempre sin sino sobre solo somos son soy su
sus tal tambien tan tanto te tenemos tener tengo ti tiene tienen toda todas todo todos tu tus un una
unas uno unos usted ustedes va vamos van vez y ya yo
"""
_FILLERS_ES = """
bueno pues vale oye mira claro osea digamos tipo verdad eh este ahi aca hola gracias bienvenidos
bienvenidas hoy cosa cosas manera forma parte hablar hablamos dicho decir dice digo creo vez veces
primero segundo tercero siguiente ultimo nuevo nueva gran grande pequeno bien mal igual
uno dos tres cuatro cinco seis siete ocho nueve diez once doce veinte cien mil
programa episodio seccion
"""
_STOPWORDS = frozenset(fold(w) for w in (_STOPWORDS_EN + _FILLERS_EN + _STOPWORDS_ES + _FILLERS_ES).split())

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _content_tokens(text: str) -> List[Tuple[str, str]]:
    """(folded, surface) pairs for the words that can carry a topic."""
    out: List[Tuple[str, str]] = []
    for match in _TOKEN_RE.finditer(text):
        surface = match.group()
        folded = fold(surface)
        if len(folded) <= 2 or folded.isdigit() or folded in _STOPWORDS:
            continue
        out.append((folded, surface))
    return out


def _tokenize(text: str) -> List[str]:
    """Folded content words (kept for backwards compatibility)."""
    return [f for f, _ in _content_tokens(text)]


# ---------------------------------------------------------------------------
# Index: segments, tokens, block IDF and window similarities — built once
# ---------------------------------------------------------------------------
def _remove(counter: Counter, tokens: List[str]) -> None:
    """Decrement counts and drop keys that reach zero (keeps windows small)."""
    for token in tokens:
        left = counter[token] - 1
        if left > 0:
            counter[token] = left
        else:
            del counter[token]


class _Index:
    def __init__(self, segments: List[Segment], window_seconds: float) -> None:
        self.segments = sorted((s for s in segments if s.text.strip()), key=lambda s: s.start)
        self.starts = [s.start for s in self.segments]
        self.window = float(window_seconds)
        self.tokens: List[List[str]] = []
        self.surfaces: Dict[str, Counter] = {}
        self.first_seen: Dict[str, int] = {}
        for seg in self.segments:
            pairs = _content_tokens(seg.text)
            self.tokens.append([f for f, _ in pairs])
            for folded, surface in pairs:
                self.surfaces.setdefault(folded, Counter())[surface] += 1
                self.first_seen.setdefault(folded, len(self.first_seen))
        self.total_tf: Counter = Counter()
        for toks in self.tokens:
            self.total_tf.update(toks)
        self.idf = self._block_idf()
        self.similarity: List[Optional[float]] = self._window_similarities()

    def _block_idf(self) -> Dict[str, float]:
        """IDF over fixed blocks of ``window`` seconds: log(N / df)."""
        if not self.segments:
            return {}
        t0 = self.starts[0]
        blocks: Dict[int, set] = {}
        for seg, toks in zip(self.segments, self.tokens):
            if toks:
                blocks.setdefault(int((seg.start - t0) // self.window), set()).update(toks)
        n = len(blocks)
        df: Counter = Counter()
        for vocab in blocks.values():
            df.update(vocab)
        return {term: math.log(n / count) for term, count in df.items()} if n else {}

    def _cosine(self, a: Counter, b: Counter) -> Optional[float]:
        idf = self.idf
        norm_a = math.sqrt(sum((c * idf.get(t, 0.0)) ** 2 for t, c in a.items()))
        norm_b = math.sqrt(sum((c * idf.get(t, 0.0)) ** 2 for t, c in b.items()))
        if norm_a == 0 or norm_b == 0:
            return None  # no topical evidence on one side (silence or only common words)
        small, large = (a, b) if len(a) <= len(b) else (b, a)
        dot = sum(c * large.get(t, 0) * idf.get(t, 0.0) ** 2 for t, c in small.items())
        return dot / (norm_a * norm_b)

    def _window_similarities(self) -> List[Optional[float]]:
        """TF-IDF cosine between the windows before/after each segment start.

        Two pointers slide over the time-sorted segments, so every segment
        enters and leaves each window once: O(n) window updates overall.
        """
        n = len(self.segments)
        sims: List[Optional[float]] = [None] * n
        before: Counter = Counter()
        after: Counter = Counter()
        lo = 0  # first segment inside the "before" window
        hi = 0  # first segment past the "after" window
        for i in range(n):
            t = self.starts[i]
            while hi < n and self.starts[hi] < t + self.window:
                after.update(self.tokens[hi])
                hi += 1
            if i > 0:
                before.update(self.tokens[i - 1])
                _remove(after, self.tokens[i - 1])
            while lo < i and self.starts[lo] < t - self.window:
                _remove(before, self.tokens[lo])
                lo += 1
            if i > 0:
                sims[i] = self._cosine(before, after)
        return sims

    def segment_index_near(self, t: float) -> int:
        return min(max(bisect_left(self.starts, t), 0), len(self.starts) - 1)

    def tokens_between(self, start: float, end: float) -> Iterable[str]:
        for i in range(bisect_left(self.starts, start), bisect_left(self.starts, end)):
            yield from self.tokens[i]


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------
def _display(index: _Index, folded: str) -> str:
    surface = index.surfaces[folded].most_common(1)[0][0]
    return surface[:1].upper() + surface[1:] if surface.islower() else surface


def _ranked_terms(index: _Index, start: float, end: float) -> List[str]:
    """Terms most distinctive of ``[start, end)`` versus the whole recording.

    score = tf_chapter * idf_blocks * (tf_chapter / tf_total): frequent here,
    rare elsewhere. When the recording is one block (idf all zero), the IDF
    factor is dropped so short media still get sensible titles.
    """
    tf = Counter(index.tokens_between(start, end))
    if not tf:
        return []
    use_idf = any(index.idf.get(t, 0.0) > 0 for t in tf)

    def score(term: str) -> float:
        concentration = tf[term] / index.total_tf[term]
        return tf[term] * concentration * (index.idf.get(term, 0.0) if use_idf else 1.0)

    ranked = sorted(tf, key=lambda t: (-score(t), index.first_seen[t]))
    picked: List[str] = []
    for term in ranked:
        if use_idf and score(term) <= 0:
            break
        # skip simple plural/singular twins ("backup" / "backups")
        if any(term.rstrip("s") == p.rstrip("s") for p in picked):
            continue
        picked.append(term)
    return picked


def _unique_title(index: _Index, terms: List[str], used: set, max_words: int, fallback: str) -> str:
    if not terms:
        title = fallback
    else:
        title = " ".join(_display(index, t) for t in terms[:max_words])
        # try swapping the last word for the next-best terms before numbering
        k = max_words
        while title.lower() in used and k < len(terms):
            head = terms[: max_words - 1]
            title = " ".join(_display(index, t) for t in head + [terms[k]])
            k += 1
    base, n = title, 2
    while title.lower() in used:
        title = f"{base} (part {n})"
        n += 1
    used.add(title.lower())
    return title


def _title_chapters(index: _Index, starts: List[float], end_of_media: float,
                    max_words: int = 3) -> List[Chapter]:
    used: set = set()
    chapters: List[Chapter] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else end_of_media + 1.0
        terms = _ranked_terms(index, start, end)
        chapters.append(Chapter(start, _unique_title(index, terms, used, max_words, f"Chapter {i + 1}")))
    return chapters


def _title_from_tokens(tokens: List[str], max_words: int = 4) -> str:
    """Most frequent tokens as a title (kept for backwards compatibility)."""
    if not tokens:
        return "Chapter"
    return " ".join(w.capitalize() for w, _ in Counter(tokens).most_common(max_words))


# ---------------------------------------------------------------------------
# Boundary selection
# ---------------------------------------------------------------------------
def _default_max_chapters(duration: float) -> int:
    """At most one chapter per ~2 minutes, between 2 and 15."""
    return int(min(15, max(2, duration // 120 + 1)))


def _split_longest(index: _Index, starts: List[float], end_of_media: float, spacing: float) -> bool:
    """Add one boundary inside the longest chapter, at its weakest-cohesion point.

    Used only to reach YouTube's three-chapter minimum. Among segment starts at
    least ``spacing`` away from both ends of the longest chapter, pick the one
    with the lowest window similarity (ties: closest to the middle). Returns
    False when no chapter can be split.
    """
    spans = sorted(
        ((nxt - cur, cur, nxt) for cur, nxt in zip(starts, starts[1:] + [end_of_media])),
        key=lambda s: (-s[0], s[1]),
    )
    for _, cur, nxt in spans:
        middle = (cur + nxt) / 2
        best = None
        for i in range(bisect_left(index.starts, cur + spacing), len(index.starts)):
            t = index.starts[i]
            if t > nxt - spacing:
                break
            sim = index.similarity[i]
            key = (1.0 if sim is None else sim, abs(t - middle), t)
            if best is None or key < best[0]:
                best = (key, t)
        if best is not None:
            starts.append(best[1])
            starts.sort()
            return True
    return False


def detect_chapters_local(
    segments: List[Segment],
    window_seconds: float = 40.0,
    sim_threshold: float = 0.30,
    min_chapter_seconds: float = 60.0,
    silence_gap: float = 3.0,
    max_chapters: Optional[int] = None,
    duration: Optional[float] = None,
    min_chapters: int = YOUTUBE_MIN_CHAPTERS,
) -> List[Chapter]:
    """Detect chapters from topic drift and long silences, no API required.

    A boundary is proposed at every segment start whose surrounding windows
    have a TF-IDF cosine below ``sim_threshold`` or that follows a silence of
    ``silence_gap`` seconds (silences rank above lexical shifts). Boundaries are
    then accepted strongest-first, at least ``min_chapter_seconds`` from each
    other and from both ends, up to ``max_chapters`` (default: about one per two
    minutes). If that yields fewer than ``min_chapters`` and the media can hold
    that many chapters of ``min_chapter_seconds``, the longest chapters are split
    at their weakest-cohesion point.
    """
    index = _Index(segments, window_seconds)
    if not index.segments:
        return []
    end_of_media = max(duration or 0.0, max(s.end for s in index.segments))
    limit = max_chapters if max_chapters is not None else _default_max_chapters(end_of_media)

    candidates: List[Tuple[float, float]] = []  # (strength, time)
    for i in range(1, len(index.segments)):
        t = index.starts[i]
        gap = t - index.segments[i - 1].end
        sim = index.similarity[i]
        if gap >= silence_gap:
            candidates.append((2.0 + gap, t))
        elif sim is not None and sim < sim_threshold:
            candidates.append((1.0 - sim, t))
    candidates.sort(key=lambda c: (-c[0], c[1]))

    starts: List[float] = [0.0]
    for _, t in candidates:
        if len(starts) >= limit:
            break
        if t < min_chapter_seconds or end_of_media - t < min_chapter_seconds:
            continue
        if any(abs(t - s) < min_chapter_seconds for s in starts):
            continue
        starts.append(t)
    starts.sort()

    if end_of_media >= min_chapters * min_chapter_seconds:
        spacing = max(YOUTUBE_MIN_CHAPTER_SECONDS, min_chapter_seconds / 2)
        while len(starts) < min_chapters and _split_longest(index, starts, end_of_media, spacing):
            pass

    chapters = _title_chapters(index, starts, end_of_media)
    return normalize_chapters(chapters, end_of_media, min_count=0)


# ---------------------------------------------------------------------------
# Normalisation (YouTube rules) — used for both local and NIM chapters
# ---------------------------------------------------------------------------
def _clean_title(title: str) -> str:
    title = " ".join(title.split()).strip(" -–—:|•*#\"'")
    return title[:100].rstrip()


def normalize_chapters(
    chapters: List[Chapter],
    duration: Optional[float],
    min_seconds: float = YOUTUBE_MIN_CHAPTER_SECONDS,
    min_count: int = YOUTUBE_MIN_CHAPTERS,
    segments: Optional[List[Segment]] = None,
    min_count_duration: float = 180.0,
) -> List[Chapter]:
    """Make a chapter list paste-ready for YouTube.

    - titles are trimmed; empty ones are dropped;
    - chapters past the end of the media (or too close to it to last
      ``min_seconds``) are dropped; the first chapter starts at 00:00;
    - a chapter shorter than ``min_seconds`` is merged into the next one (the
      earlier start is kept, with the later title, which covers most of it);
    - consecutive duplicates merge, other repeated titles get "(part N)";
    - with ``segments``, media of at least ``min_count_duration`` seconds get at
      least ``min_count`` chapters by splitting the longest ones at their
      weakest-cohesion point (titled from their own distinctive words).
    """
    items = sorted(
        (Chapter(max(0.0, float(c.start)), _clean_title(c.title)) for c in chapters if _clean_title(c.title)),
        key=lambda c: c.start,
    )
    if duration and duration > 0:
        items = [c for c in items if c.start == 0.0 or c.start <= duration - min_seconds]
    if not items:
        return []

    if items[0].start > 0:
        if items[0].start < min_seconds:
            items[0] = Chapter(0.0, items[0].title)
        else:
            items.insert(0, Chapter(0.0, "Intro"))

    merged: List[Chapter] = []
    for ch in items:
        if merged and ch.start - merged[-1].start < min_seconds:
            merged[-1] = Chapter(merged[-1].start, ch.title)
            continue
        if merged and ch.title.lower() == merged[-1].title.lower():
            continue  # same topic continues
        merged.append(ch)

    if segments and duration and duration >= min_count_duration and len(merged) < min_count:
        index = _Index(segments, 40.0)
        if index.segments:
            starts = [c.start for c in merged]
            known = {c.start: c.title for c in merged}
            spacing = max(min_seconds, duration / (min_count * 3))
            while len(starts) < min_count and _split_longest(index, starts, duration, spacing):
                pass
            used = {t.lower() for t in known.values()}
            filled = []
            for i, s in enumerate(starts):
                if s in known:
                    filled.append(Chapter(s, known[s]))
                    continue
                end = starts[i + 1] if i + 1 < len(starts) else duration + 1.0
                terms = _ranked_terms(index, s, end)
                filled.append(Chapter(s, _unique_title(index, terms, used, 3, f"Part {i + 1}")))
            merged = filled

    seen: Dict[str, int] = {}
    out: List[Chapter] = []
    for ch in merged:
        key = ch.title.lower()
        seen[key] = seen.get(key, 0) + 1
        title = ch.title if seen[key] == 1 else f"{ch.title} (part {seen[key]})"
        out.append(Chapter(ch.start, title))
    return out


# ---------------------------------------------------------------------------
# NIM chapters
# ---------------------------------------------------------------------------
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
    segments: List[Segment], nim_client, step_seconds: float = 30.0,
    duration: Optional[float] = None,
) -> List[Chapter]:
    """Ask NIM for chapter titles, parse them back, and enforce YouTube's rules."""
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
    end_of_media = max(duration or 0.0, max((s.end for s in segments), default=0.0))
    chapters = normalize_chapters(parse_chapter_lines(reply), end_of_media, segments=segments)
    if not chapters:
        return detect_chapters_local(segments, duration=duration)
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
    duration: Optional[float] = None,
    **kwargs,
) -> List[Chapter]:
    """Use NIM when a client is supplied, otherwise the local heuristic.

    Any NIM failure degrades to the offline heuristic; when ``warnings`` is a
    list, the reason is appended to it so callers can report it.
    """
    if nim_client is not None:
        try:
            return detect_chapters_nim(segments, nim_client, duration=duration)
        except Exception as exc:
            if warnings is not None:
                warnings.append(
                    f"NIM chapters failed ({exc}); used the offline chapter heuristic instead."
                )
            return detect_chapters_local(segments, duration=duration, **kwargs)
    return detect_chapters_local(segments, duration=duration, **kwargs)


def chapters_to_markdown(chapters: List[Chapter], title: str = "Chapters") -> str:
    lines = [f"# {title}", ""]
    for ch in chapters:
        lines.append(f"{format_chapter_timestamp(ch.start)} {ch.title}")
    return "\n".join(lines).rstrip("\n") + "\n"


def chapters_to_text(chapters: List[Chapter]) -> str:
    """Plain ``00:00 Title`` lines, ready to paste into a video description."""
    return "\n".join(f"{format_chapter_timestamp(ch.start)} {ch.title}" for ch in chapters)
