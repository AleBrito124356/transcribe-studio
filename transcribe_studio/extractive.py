"""Offline extractive summary: no API key, no network, standard library only.

Used when there is no NVIDIA NIM key, with ``--no-nim`` / ``--local``, or after
a NIM failure, so ``summary.md`` is always produced. Nothing is paraphrased:
every line is a sentence quoted from the transcript, chosen by how central it
is to the whole conversation.

How sentences are chosen:

1. The transcript is split into sentences (``. ! ? …``, Spanish ``¿ ¡``
   aware); run-on text without punctuation is cut into ~30-word pieces.
2. Each sentence becomes a TF-IDF vector over its content words (the same
   accent-folded bilingual stopword list as the chapter detector).
3. Sentences are ranked with a small TextRank: a PageRank over the cosine
   similarity graph, so a sentence scores high when many others talk about
   the same things. Very long transcripts use centroid similarity instead,
   which is linear. Very short and very long sentences are down-weighted.
4. Picks skip near-duplicates of sentences already picked.

The result has the same shape as the NIM summary (``tldr``, ``summary``,
``key_points``, ``action_items``) plus ``method: "extractive"``, and
:func:`transcribe_studio.summarize.to_markdown` labels it as such.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from .chapters import _content_tokens, fold

EXTRACTIVE_NOTE = (
    "_Offline extractive summary: the sentences below are quoted from the transcript and "
    "ranked by TF-IDF centrality (no LLM, no network). Set NVIDIA_API_KEY for an "
    "abstractive summary written by NVIDIA NIM._"
)

# Split after . ! ? … (optionally followed by one closing quote/bracket, which
# stays with its sentence) when the next sentence starts with a capital,
# a digit, or an opening quote / ¿ / ¡.
_SENTENCE_END = re.compile(
    r"(?:(?<=[.!?…])|(?<=[.!?…][\"'»”’)\]]))\s+(?=[\"'«“‘(\[¿¡]?[A-ZÁÉÍÓÚÑÜ0-9])"
)
_MAX_WORDS = 45
_RUN_ON_WORDS = 60
_PIECE_WORDS = 30
_TEXTRANK_LIMIT = 700  # above this many sentences, use linear centroid scoring

# Commitments and next steps. Matched on accent-folded, lowercased text.
_ACTION_PATTERNS = [
    r"\b(?:we|i|you|they|he|she|someone)\s+(?:really\s+)?(?:need|needs|have|has|got)\s+to\b",
    r"\b(?:we|i|you|they)\s+(?:should|must|ought to)\b",
    r"\blet'?s\b",
    r"\b(?:i'll|i will|we'll|we will|i'm going to|we're going to|i am going to|we are going to)\b",
    # "to-do" needs its hyphen: a bare "todo" is Spanish for "everything".
    r"\b(?:make sure|don'?t forget|remember to|follow up|action items?|to-do|next steps?)\b",
    r"\bhay que\b",
    r"\b(?:tenemos|tengo|tienes|tienen|hay) que\b",
    r"\bvamos a\b",
    r"\b(?:debemos|deberiamos|necesitamos|necesito|necesitas)\b",
    r"\b(?:no (?:te )?olvides|no nos olvidemos|recuerda|recuerden|acuerdate)\b",
    r"\b(?:quedamos en|queda pendiente|proximo paso|siguiente paso|me encargo|te encargas|se encarga)\b",
]
# Conversational framing that looks like an action but is not one.
_NOT_ACTIONS = [
    r"\blet'?s (?:talk|get started|start|begin|dive|move on|go back|see|look|recap|jump)\b",
    r"\bvamos a (?:hablar|ver|empezar|comenzar|pasar|seguir|platicar|charlar|conversar|dar paso|repasar)\b",
    r"\b(?:we're|we are|i'm|i am) going to (?:talk|start|begin|cover|look|discuss)\b",
    r"\b(?:we|i) will (?:talk|start|begin|cover|look|discuss)\b",
]
_ACTION_RE = re.compile("|".join(_ACTION_PATTERNS))
_NOT_ACTION_RE = re.compile("|".join(_NOT_ACTIONS))


# ---------------------------------------------------------------------------
# Sentences
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> List[str]:
    """Split transcript text into sentences, cutting run-ons into pieces."""
    sentences: List[str] = []
    for block in re.split(r"\n+", text or ""):
        block = " ".join(block.split())
        if not block:
            continue
        for sentence in _SENTENCE_END.split(block):
            words = sentence.split()
            if not words:
                continue
            if len(words) > _RUN_ON_WORDS:
                for i in range(0, len(words), _PIECE_WORDS):
                    sentences.append(" ".join(words[i:i + _PIECE_WORDS]))
            else:
                sentences.append(" ".join(words))
    return sentences


def is_action_item(sentence: str) -> bool:
    """True for commitments and next steps ("we need to...", "hay que...")."""
    folded = fold(sentence).replace("’", "'")
    if sentence.rstrip().endswith("?"):
        return False
    if not _ACTION_RE.search(folded):
        return False
    stripped = _NOT_ACTION_RE.sub(" ", folded)
    return bool(_ACTION_RE.search(stripped))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _vectors(sentences: Sequence[str]) -> List[Dict[str, float]]:
    tfs = [Counter(f for f, _ in _content_tokens(s)) for s in sentences]
    df: Counter = Counter()
    for tf in tfs:
        df.update(tf.keys())
    n = len(sentences)
    idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}
    vectors = []
    for tf in tfs:
        vec = {t: c * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        vectors.append({t: v / norm for t, v in vec.items()} if norm else {})
    return vectors


def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


def _textrank(vectors: List[Dict[str, float]], damping: float = 0.85,
              iterations: int = 60, tol: float = 1e-9) -> List[float]:
    n = len(vectors)
    # Sparse similarity graph through an inverted index: only sentence pairs
    # that share a term are ever compared.
    postings: Dict[str, List[int]] = {}
    for i, vec in enumerate(vectors):
        for term in vec:
            postings.setdefault(term, []).append(i)
    edges: List[Dict[int, float]] = [dict() for _ in range(n)]
    for i, vec in enumerate(vectors):
        neighbours = set()
        for term in vec:
            neighbours.update(postings[term])
        for j in neighbours:
            if j > i:
                w = _dot(vec, vectors[j])
                if w > 0:
                    edges[i][j] = w
                    edges[j][i] = w
    out_weight = [sum(e.values()) for e in edges]
    scores = [1.0 / n] * n
    for _ in range(iterations):
        new = [(1 - damping) / n] * n
        for i in range(n):
            if out_weight[i] == 0:
                continue
            share = damping * scores[i] / out_weight[i]
            for j, w in edges[i].items():
                new[j] += share * w
        # sentences with no edges keep only the teleport mass
        delta = sum(abs(a - b) for a, b in zip(new, scores))
        scores = new
        if delta < tol:
            break
    return scores


def _centroid(vectors: List[Dict[str, float]]) -> List[float]:
    centroid: Counter = Counter()
    for vec in vectors:
        centroid.update(vec)
    norm = math.sqrt(sum(v * v for v in centroid.values())) or 1.0
    unit = {t: v / norm for t, v in centroid.items()}
    return [_dot(vec, unit) for vec in vectors]


def rank_sentences(sentences: Sequence[str]) -> List[float]:
    """Centrality score per sentence (higher = more representative)."""
    if not sentences:
        return []
    vectors = _vectors(sentences)
    base = _textrank(vectors) if len(sentences) <= _TEXTRANK_LIMIT else _centroid(vectors)
    scores = []
    for sentence, vec, score in zip(sentences, vectors, base):
        words = len(sentence.split())
        if not vec:
            factor = 0.0  # nothing but filler ("Yeah. Okay.")
        elif words < 6:
            factor = 0.5
        elif words > _MAX_WORDS:
            factor = 0.8
        else:
            factor = 1.0
        scores.append(score * factor)
    return scores


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _pick(order: List[int], vectors: List[Dict[str, float]], count: int,
          taken: List[int], max_similarity: float = 0.6) -> List[int]:
    picked: List[int] = []
    for i in order:
        if len(picked) >= count:
            break
        if i in taken or i in picked:
            continue
        if any(_dot(vectors[i], vectors[j]) > max_similarity for j in taken + picked):
            continue
        picked.append(i)
    return picked


def summarize_extractive(
    text: str,
    max_overview: Optional[int] = None,
    max_key_points: int = 6,
    max_actions: int = 8,
) -> dict:
    """Summarise ``text`` by quoting its most central sentences.

    Returns ``{"tldr", "summary", "key_points", "action_items", "method"}``.
    Deterministic: the same text always yields the same summary.
    """
    sentences = split_sentences(text)
    empty = {"tldr": "", "summary": "", "key_points": [], "action_items": [], "method": "extractive"}
    if not sentences:
        return empty
    scores = rank_sentences(sentences)
    vectors = _vectors(sentences)
    order = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    order = [i for i in order if scores[i] > 0] or list(range(len(sentences)))

    n_overview = max_overview or int(min(5, max(2, round(len(sentences) / 12))))
    tldr = order[0]
    overview = sorted(_pick(order, vectors, n_overview, []))
    n_points = int(min(max_key_points, max(3, round(len(sentences) / 10))))
    points = sorted(_pick(order, vectors, n_points, list(overview)))

    actions: List[str] = []
    seen = set()
    for sentence in sentences:
        key = fold(sentence)
        if key not in seen and is_action_item(sentence):
            seen.add(key)
            actions.append(sentence)
        if len(actions) >= max_actions:
            break

    return {
        "tldr": sentences[tldr],
        "summary": " ".join(sentences[i] for i in overview),
        "key_points": [sentences[i] for i in points],
        "action_items": actions,
        "method": "extractive",
    }
