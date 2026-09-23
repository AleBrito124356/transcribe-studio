"""End-to-end pipeline wiring with whisper and NIM mocked out."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcribe_studio import pipeline as pipeline_mod
from transcribe_studio import transcribe as transcribe_mod
from transcribe_studio.config import Segment, TranscriptResult
from transcribe_studio.pipeline import PipelineOptions, run, run_batch


def _fake_transcript():
    segments = [
        Segment(start=0.0, end=3.0, text="Welcome to the show today."),
        Segment(start=3.2, end=7.0, text="We will talk about testing pipelines."),
        Segment(start=7.5, end=12.0, text="Then we move on to deployment topics."),
    ]
    text = " ".join(s.text for s in segments)
    return TranscriptResult(segments=segments, language="en", duration=12.0, text=text, model_size="base")


class RoutingNim:
    """Fake NIM that answers summary vs chapter requests appropriately."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, **kwargs):
        self.calls += 1
        system = messages[0]["content"].lower()
        if "chapter" in system:
            return "00:00 Intro\n00:07 Deployment"
        # Summary request -> valid JSON object.
        return json.dumps(
            {
                "tldr": "A short test episode.",
                "summary": "The hosts discuss testing and deployment.",
                "key_points": ["Testing matters", "Deploy carefully"],
                "action_items": ["Add CI"],
            }
        )

    def complete(self, system, user, **kwargs):
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], **kwargs)


@pytest.fixture(autouse=True)
def _patch_transcribe(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "transcribe", lambda *a, **k: _fake_transcript())


def test_run_produces_all_artifacts(tmp_path):
    out = tmp_path / "out"
    result = run(
        "fake_media.mp3",
        str(out),
        PipelineOptions(make_summary=True, make_chapters=True),
        nim_client=RoutingNim(),
    )
    assert result.ok
    for name in ("transcript.txt", "transcript.json", "captions.srt", "captions.vtt", "summary.md", "chapters.md"):
        assert (out / name).exists(), f"missing {name}"

    # Subtitles are valid-ish.
    assert (out / "captions.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
    assert "-->" in (out / "captions.srt").read_text(encoding="utf-8")

    # Summary markdown carries the TL;DR.
    assert "A short test episode." in (out / "summary.md").read_text(encoding="utf-8")

    # Chapters start at 00:00.
    assert "00:00 Intro" in (out / "chapters.md").read_text(encoding="utf-8")

    # transcript.json round-trips.
    data = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    assert data["language"] == "en"
    assert len(data["segments"]) == 3


def test_run_without_nim_skips_summary_but_keeps_local(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    out = tmp_path / "out"
    result = run(
        "fake_media.mp3",
        str(out),
        PipelineOptions(make_summary=True, make_chapters=True, use_nim=True),
    )
    # No key -> no summary, but transcript/subs/chapters still land.
    assert not (out / "summary.md").exists()
    assert (out / "transcript.txt").exists()
    assert (out / "captions.srt").exists()
    assert (out / "chapters.md").exists()  # offline heuristic
    assert any("NVIDIA_API_KEY" in w for w in result.warnings)


def test_run_with_translation(tmp_path):
    out = tmp_path / "out"

    class TransNim(RoutingNim):
        def chat(self, messages, **kwargs):
            system = messages[0]["content"].lower()
            if "translat" in system:
                return json.dumps(["Hola", "Mundo", "Adios"])
            return super().chat(messages, **kwargs)

    run(
        "fake_media.mp3",
        str(out),
        PipelineOptions(translate="es", make_summary=False, make_chapters=False),
        nim_client=TransNim(),
    )
    assert (out / "captions.es.srt").exists()
    assert (out / "captions.es.vtt").exists()
    assert "Hola" in (out / "captions.es.srt").read_text(encoding="utf-8")


def test_diarized_transcript_has_speaker_labels(tmp_path):
    out = tmp_path / "out"
    run(
        "fake_media.mp3",
        str(out),
        PipelineOptions(diarize=True, make_summary=False, make_chapters=False),
        nim_client=RoutingNim(),
    )
    body = (out / "transcript.txt").read_text(encoding="utf-8")
    assert "Speaker 1" in body


def test_batch_isolates_failures(tmp_path, monkeypatch):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "good.mp3").write_bytes(b"x")
    (media_dir / "bad.mp3").write_bytes(b"x")
    (media_dir / "notes.txt").write_text("ignore me", encoding="utf-8")  # not media

    # Avoid loading a real whisper model in batch mode.
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda *a, **k: object())

    def flaky(media_path, *a, **k):
        if "bad" in str(media_path):
            raise RuntimeError("boom")
        return _fake_transcript()

    monkeypatch.setattr(pipeline_mod, "transcribe", flaky)

    results = run_batch(str(media_dir), str(tmp_path / "out"), PipelineOptions(make_summary=False, use_nim=False))
    assert len(results) == 2  # the .txt file is ignored
    by_name = {Path(r.source).name: r for r in results}
    assert by_name["good.mp3"].ok
    assert not by_name["bad.mp3"].ok
    assert "boom" in by_name["bad.mp3"].error
    # A report is written.
    assert (tmp_path / "out" / "batch_report.md").exists()
