"""Offline extractive summary: sentence ranking, action items, determinism."""

from __future__ import annotations

import random
import time

from transcribe_studio.extractive import (
    is_action_item,
    rank_sentences,
    split_sentences,
    summarize_extractive,
)
from transcribe_studio.summarize import to_markdown

BILINGUAL = " ".join([
    # English half: backups
    "Welcome back to the show, everybody.",
    "Today we want to talk about backups for a small home server.",
    "Let's talk about why most home backups fail silently.",
    "A backup only counts once you restore a snapshot and check the restored files.",
    "Most people copy files to a second drive and never test a restore.",
    "Snapshots on the same drive are not a backup when the drive dies.",
    "An offsite copy protects the backup from fire and theft.",
    "Encryption matters for the offsite backup because the cloud provider holds the files.",
    "We need to test a restore from the offsite snapshot every month.",
    "Yeah.",
    "I'll write down the retention policy for the snapshots tonight.",
    "Okay, okay.",
    # Spanish half: tomatoes on the balcony
    "Ahora pasamos a la parte en español del programa.",
    "Vamos a hablar de tomates en el balcón.",
    "Los tomates en maceta necesitan sol directo y un riego constante.",
    "El riego por goteo evita que la maceta se seque en verano.",
    "Una maceta pequeña limita las raíces y los tomates salen pequeños.",
    "Hay que regar temprano, antes de que el sol caliente la maceta.",
    "Tenemos que comprar abono orgánico para los tomates esta semana.",
    "¿Tú riegas por la noche?",
    "No te olvides de poner un tutor a cada planta de tomate.",
])

ACTIONS = {
    "We need to test a restore from the offsite snapshot every month.",
    "I'll write down the retention policy for the snapshots tonight.",
    "Hay que regar temprano, antes de que el sol caliente la maceta.",
    "Tenemos que comprar abono orgánico para los tomates esta semana.",
    "No te olvides de poner un tutor a cada planta de tomate.",
}
FRAMING = {
    "Let's talk about why most home backups fail silently.",
    "Vamos a hablar de tomates en el balcón.",
    "¿Tú riegas por la noche?",
}


def test_split_sentences_handles_spanish_marks_quotes_and_run_ons():
    parts = split_sentences('He said "stop." Then we left. ¿Qué pasó? ¡Nada! Fin.')
    assert parts == ['He said "stop."', "Then we left.", "¿Qué pasó?", "¡Nada!", "Fin."]
    run_on = " ".join(f"word{i}" for i in range(130))  # no punctuation at all
    pieces = split_sentences(run_on)
    assert len(pieces) == 5 and all(len(p.split()) <= 30 for p in pieces)
    assert " ".join(pieces) == run_on


def test_action_items_are_found_in_both_languages_and_framing_is_ignored():
    summary = summarize_extractive(BILINGUAL)
    found = set(summary["action_items"])
    assert ACTIONS <= found
    assert not (FRAMING & found)
    # Kept in the order they were said.
    order = [s for s in split_sentences(BILINGUAL) if s in found]
    assert summary["action_items"] == order


def test_is_action_item_edge_cases():
    assert is_action_item("Vamos a hablar de tomates y vamos a comprar abono.")
    assert not is_action_item("We will talk about that later.")
    assert is_action_item("Let’s ship it on Friday.")  # curly apostrophe
    assert not is_action_item("Do we need to buy soil?")


def test_overview_keeps_original_order_and_key_points_do_not_repeat_it():
    sentences = split_sentences(BILINGUAL)
    summary = summarize_extractive(BILINGUAL)
    overview_idx = [sentences.index(s) for s in split_sentences(summary["summary"])]
    assert overview_idx == sorted(overview_idx) and len(overview_idx) >= 2
    points_idx = [sentences.index(p) for p in summary["key_points"]]
    assert points_idx == sorted(points_idx)
    assert not set(summary["key_points"]) & set(split_sentences(summary["summary"]))
    assert summary["tldr"] in sentences
    assert summary["method"] == "extractive"


def test_filler_never_makes_the_summary():
    summary = summarize_extractive(BILINGUAL)
    chosen = [summary["tldr"], *split_sentences(summary["summary"]), *summary["key_points"]]
    assert "Yeah." not in chosen and "Okay, okay." not in chosen


def test_the_most_central_sentence_is_the_tldr():
    text = " ".join([
        "A backup only counts once you restore a snapshot and verify the restored files.",
        "Restore tests catch broken backups early.",
        "A snapshot is cheap to take.",
        "Verify the files after every restore.",
        "Backups without a restore test are hope, not a plan.",
        "My cat likes the sunny window in the afternoon.",
        "Every snapshot should be verified with a real restore.",
    ])
    assert summarize_extractive(text)["tldr"].startswith("A backup only counts")


def test_summary_is_deterministic_and_markdown_is_labelled():
    first = summarize_extractive(BILINGUAL)
    second = summarize_extractive(BILINGUAL)
    assert first == second
    md = to_markdown(first)
    assert md == to_markdown(second)
    assert md.startswith("# Summary\n\n_Offline extractive summary:")
    assert "## Action items" in md and "- [ ] Hay que regar temprano" in md


def test_empty_and_tiny_inputs():
    assert summarize_extractive("")["tldr"] == ""
    tiny = summarize_extractive("Just one sentence here about backups.")
    assert tiny["tldr"] == "Just one sentence here about backups."
    assert rank_sentences([]) == []


def test_long_transcripts_use_the_linear_scorer_quickly():
    rng = random.Random(5)
    vocab = [f"topic{i}" for i in range(400)]
    text = " ".join(
        "Sentence " + " ".join(rng.choice(vocab) for _ in range(14)) + "." for _ in range(3000)
    )
    began = time.perf_counter()
    summary = summarize_extractive(text)
    assert time.perf_counter() - began < 3.0
    assert summary["tldr"] and 2 <= len(split_sentences(summary["summary"])) <= 5
