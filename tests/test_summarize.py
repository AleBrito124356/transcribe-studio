"""Chunking, JSON parsing, map-reduce, and markdown rendering for summaries."""

from __future__ import annotations

import json

from src.summarize import (
    DEFAULT_MAX_CHUNK_CHARS,
    chunk_text,
    resolve_language,
    summarize_transcript,
    to_markdown,
)


def test_chunk_text_short_stays_single():
    assert chunk_text("A short sentence.") == ["A short sentence."]


def test_chunk_text_splits_long_input():
    sentence = "This is a sentence with several words in it. "
    text = sentence * 500  # ~22k chars
    chunks = chunk_text(text, max_chars=2000, overlap=0)
    assert len(chunks) > 1
    # Each chunk stays within the limit (sentences are short here).
    assert all(len(c) <= 2000 for c in chunks)


def test_chunk_text_preserves_content_words():
    text = " ".join(f"sentence{i}." for i in range(1000))
    chunks = chunk_text(text, max_chars=1000, overlap=0)
    joined = " ".join(chunks)
    assert "sentence0." in joined
    assert "sentence999." in joined


def test_chunk_text_oversized_single_sentence():
    huge = "word" * 5000  # one 20k-char "sentence" with no boundary
    chunks = chunk_text(huge, max_chars=1000)
    assert len(chunks) == 1  # emitted whole rather than cut mid-word


def test_summarize_single_chunk(fake_nim):
    payload = {
        "tldr": "A short talk about testing.",
        "summary": "The speaker explains why tests matter.",
        "key_points": ["Tests catch regressions", "Mock the network"],
        "action_items": ["Write more tests"],
    }

    def responder(messages):
        return "Here you go:\n" + json.dumps(payload)

    client = fake_nim(responder)
    result = summarize_transcript("A brief transcript.", client)
    assert result["tldr"] == payload["tldr"]
    assert result["key_points"] == payload["key_points"]
    assert result["action_items"] == ["Write more tests"]
    # Only one chunk -> exactly one NIM call.
    assert len(client.calls) == 1


def test_summarize_map_reduce_calls_multiple_times(fake_nim):
    payload = {"tldr": "t", "summary": "s", "key_points": ["k"], "action_items": []}

    def responder(messages):
        return json.dumps(payload)

    client = fake_nim(responder)
    long_text = " ".join(f"Sentence number {i} here." for i in range(4000))
    assert len(long_text) > DEFAULT_MAX_CHUNK_CHARS
    summarize_transcript(long_text, client)
    # N chunk summaries + 1 reduce call.
    assert len(client.calls) >= 3


def test_summarize_tolerates_non_json(fake_nim):
    def responder(messages):
        return "The model forgot to return JSON but here is prose."

    client = fake_nim(responder)
    result = summarize_transcript("text", client)
    assert result["summary"]  # falls back to putting the reply in summary
    assert result["key_points"] == []


def test_to_markdown_renders_sections():
    md = to_markdown(
        {
            "tldr": "One line.",
            "summary": "Paragraph.",
            "key_points": ["p1", "p2"],
            "action_items": ["do a", "do b"],
        }
    )
    assert "## TL;DR" in md
    assert "## Key points" in md
    assert "- p1" in md
    assert "- [ ] do a" in md


def test_resolve_language():
    assert resolve_language("es", "en") == "Spanish"
    assert resolve_language("en", "es") == "English"
    assert resolve_language("auto", "es") == "Spanish"
    assert resolve_language("auto", "de") == "the same language as the transcript"
