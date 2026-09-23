"""Transcript summaries via NVIDIA NIM: TL;DR, detailed summary, key points, actions.

Long transcripts are handled with a map-reduce (chunk-and-reduce) strategy: each
chunk is summarised on its own, then the partial summaries are summarised
together into a single result. All model calls go through the NIM client so the
whole module is trivially mockable in tests.

Without a key (or after a NIM failure) the pipeline and the ``summarize``
command use :mod:`transcribe_studio.extractive` instead, which returns the same
dict shape and is labelled as an offline extractive summary in ``summary.md``.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

# A transcript chunk of roughly this many characters keeps each request well
# within a 70B model's context while minimising the number of round-trips.
DEFAULT_MAX_CHUNK_CHARS = 8000

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

_SINGLE_SYSTEM = (
    "You are an expert meeting and media analyst. Read the transcript and produce "
    "a structured summary. Respond in {language}. Return ONLY a JSON object with "
    "exactly these keys:\n"
    '  "tldr": a single sentence, under 40 words.\n'
    '  "summary": 2-4 short paragraphs of prose.\n'
    '  "key_points": an array of 3-8 concise bullet strings.\n'
    '  "action_items": an array of concrete next steps as strings (empty array if none).\n'
    "Do not wrap the JSON in markdown fences or add commentary."
)

_REDUCE_SYSTEM = (
    "You are combining several partial summaries of one long transcript into a "
    "single coherent summary. Respond in {language}. Merge overlapping points and "
    "remove duplication. Return ONLY the same JSON object with keys tldr, summary, "
    "key_points, action_items."
)


def chunk_text(
    text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS, overlap: int = 200
) -> List[str]:
    """Split ``text`` into chunks no larger than ``max_chars``, breaking on sentences.

    Chunks carry a small character ``overlap`` of trailing context so a thought
    split across a boundary is still summarised with its lead-in. A single
    sentence longer than ``max_chars`` becomes its own oversized chunk rather
    than being cut mid-word.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip() if tail else sentence
        else:
            # Single sentence longer than the limit: emit it whole.
            chunks.append(sentence)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _parse_summary(reply: str) -> dict:
    """Parse the model's JSON object, tolerating fences and surrounding prose."""
    fallback = {"tldr": "", "summary": reply.strip(), "key_points": [], "action_items": []}
    start = reply.find("{")
    end = reply.rfind("}")
    if start == -1 or end == -1 or end < start:
        return fallback
    try:
        data = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return fallback
    if not isinstance(data, dict):
        return fallback
    return {
        "tldr": str(data.get("tldr", "")).strip(),
        "summary": str(data.get("summary", "")).strip(),
        "key_points": [str(x).strip() for x in data.get("key_points", []) if str(x).strip()],
        "action_items": [str(x).strip() for x in data.get("action_items", []) if str(x).strip()],
    }


def _summarize_single(text: str, nim_client, language: str, reduce: bool = False) -> dict:
    system = (_REDUCE_SYSTEM if reduce else _SINGLE_SYSTEM).format(language=language)
    reply = nim_client.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    return _parse_summary(reply)


def _partial_to_text(partial: dict) -> str:
    lines = [partial.get("summary", "")]
    for kp in partial.get("key_points", []):
        lines.append(f"- {kp}")
    for ai in partial.get("action_items", []):
        lines.append(f"ACTION: {ai}")
    return "\n".join(l for l in lines if l).strip()


def summarize_transcript(
    text: str,
    nim_client,
    language: str = "the same language as the transcript",
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> dict:
    """Summarise a transcript, mapping-and-reducing when it is long.

    Returns a dict with keys ``tldr``, ``summary``, ``key_points`` (list) and
    ``action_items`` (list).
    """
    text = (text or "").strip()
    if not text:
        return {"tldr": "", "summary": "", "key_points": [], "action_items": []}

    chunks = chunk_text(text, max_chunk_chars)
    if len(chunks) == 1:
        return _summarize_single(chunks[0], nim_client, language)

    partials = [_summarize_single(c, nim_client, language) for c in chunks]
    combined = "\n\n".join(_partial_to_text(p) for p in partials)
    return _summarize_single(combined, nim_client, language, reduce=True)


def to_markdown(summary: dict, title: str = "Summary") -> str:
    """Render a summary dict as Markdown.

    Offline extractive summaries (``method == "extractive"``) are labelled as
    such right under the title, so nobody mistakes quoted sentences for an
    LLM-written summary.
    """
    lines = [f"# {title}", ""]
    if summary.get("method") == "extractive":
        from .extractive import EXTRACTIVE_NOTE

        lines += [EXTRACTIVE_NOTE, ""]
    if summary.get("tldr"):
        lines += ["## TL;DR", "", summary["tldr"], ""]
    if summary.get("summary"):
        lines += ["## Overview", "", summary["summary"], ""]
    if summary.get("key_points"):
        lines += ["## Key points", ""]
        lines += [f"- {kp}" for kp in summary["key_points"]]
        lines.append("")
    if summary.get("action_items"):
        lines += ["## Action items", ""]
        lines += [f"- [ ] {ai}" for ai in summary["action_items"]]
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def resolve_language(name: Optional[str], detected: str) -> str:
    """Map a CLI language flag onto an instruction phrase for the model."""
    if not name or name == "auto":
        pretty = {"en": "English", "es": "Spanish"}.get(detected, None)
        if pretty:
            return pretty
        return "the same language as the transcript"
    return {"en": "English", "es": "Spanish"}.get(name, name)
