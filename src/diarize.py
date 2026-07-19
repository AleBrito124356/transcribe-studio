"""Lightweight, approximate speaker turns from silence gaps.

This is intentionally simple: whenever the gap between two segments exceeds a
threshold, we treat it as a possible turn change and alternate the speaker
label. It needs no extra dependencies and no model, and it is genuinely useful
for two-person interviews and podcasts.

It is NOT real diarization: it cannot tell voices apart, count speakers, or
handle overlaps. For accurate results, install pyannote.audio and use the hook
described in ``PYANNOTE_NOTE`` / ``diarize_pyannote``.
"""

from __future__ import annotations

from typing import List

from .config import Segment

PYANNOTE_NOTE = (
    "Speaker labels below are approximate. They are inferred from pauses between "
    "segments, not from voice characteristics. For accurate speaker diarization, "
    "install pyannote.audio (pip install pyannote.audio), accept the model terms "
    "on Hugging Face, set HUGGINGFACE_TOKEN, and call diarize_pyannote() from "
    "src/diarize.py."
)


def assign_speakers(
    segments: List[Segment],
    gap_threshold: float = 1.2,
    max_speakers: int = 2,
) -> List[Segment]:
    """Return copies of ``segments`` with alternating ``Speaker N`` labels.

    A new turn is declared when the silence before a segment exceeds
    ``gap_threshold`` seconds; the label then advances round-robin across
    ``max_speakers`` labels.
    """
    if max_speakers < 1:
        raise ValueError("max_speakers must be >= 1")
    labelled: List[Segment] = []
    current = 1
    prev_end = None
    for seg in segments:
        if prev_end is not None and (seg.start - prev_end) > gap_threshold:
            current = current % max_speakers + 1
        labelled.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=seg.words,
                speaker=f"Speaker {current}",
            )
        )
        prev_end = seg.end
    return labelled


def to_dialogue(segments: List[Segment]) -> str:
    """Render speaker-labelled segments as a merged dialogue transcript."""
    lines: List[str] = []
    current_speaker = None
    buffer: List[str] = []
    for seg in segments:
        speaker = seg.speaker or "Speaker 1"
        text = seg.text.strip()
        if not text:
            continue
        if speaker != current_speaker:
            if buffer:
                lines.append(f"{current_speaker}: {' '.join(buffer)}")
            current_speaker = speaker
            buffer = [text]
        else:
            buffer.append(text)
    if buffer:
        lines.append(f"{current_speaker}: {' '.join(buffer)}")
    return "\n\n".join(lines)


def diarize_pyannote(audio_path: str, segments: List[Segment], hf_token: str) -> List[Segment]:
    """Real diarization via pyannote.audio (optional, heavier dependency).

    Runs a pretrained pipeline to get speaker turns, then assigns each transcript
    segment the speaker whose turn overlaps it most. Requires
    ``pip install pyannote.audio`` and a Hugging Face token with access to the
    ``pyannote/speaker-diarization-3.1`` model.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pyannote.audio is not installed. Install it with:\n"
            "    pip install pyannote.audio\n"
            "and accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        ) from exc

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    diarization = pipeline(audio_path)

    # Build a list of (start, end, speaker) turns.
    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    labelled: List[Segment] = []
    for seg in segments:
        best_speaker = None
        best_overlap = 0.0
        for start, end, speaker in turns:
            overlap = max(0.0, min(seg.end, end) - max(seg.start, start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        labelled.append(
            Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=seg.words,
                speaker=best_speaker or "Speaker 1",
            )
        )
    return labelled
