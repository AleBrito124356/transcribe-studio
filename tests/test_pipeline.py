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


class FailingNim:
    """A NIM client whose every call fails the way the real one does."""

    def __init__(self, fatal=False):
        self.calls = 0
        self.fatal = fatal

    def chat(self, messages, **kwargs):
        from transcribe_studio.nim import NimError

        self.calls += 1
        if self.fatal:
            raise NimError("NIM rejected the API key (HTTP 401).", status=401, fatal=True)
        raise NimError("Could not reach NIM at http://127.0.0.1:1/v1: connection refused (after 4 attempts).")


def test_nim_failure_does_not_abort_the_run(tmp_path):
    out = tmp_path / "out"
    nim = FailingNim()
    result = run(
        "fake_media.mp3",
        str(out),
        PipelineOptions(translate="es", make_summary=True, make_chapters=True),
        nim_client=nim,
    )
    assert result.ok
    for name in ("transcript.txt", "transcript.json", "captions.srt", "captions.vtt", "chapters.md"):
        assert (out / name).exists(), f"missing {name}"
    assert not (out / "captions.es.srt").exists()
    assert nim.calls == 1  # after the first failure the run stops calling NIM
    assert result.nim_error and "connection refused" in result.nim_error
    assert not result.nim_fatal
    assert any("translation to 'es' failed" in w for w in result.warnings)
    assert (out / "chapters.md").read_text(encoding="utf-8").startswith("# Chapters\n\n00:00 ")


def test_nim_chapter_failure_falls_back_with_a_warning(tmp_path):
    out = tmp_path / "out"
    result = run(
        "fake_media.mp3",
        str(out),
        PipelineOptions(make_summary=False, make_chapters=True),
        nim_client=FailingNim(),
    )
    assert (out / "chapters.md").exists()
    assert any("offline chapter heuristic" in w for w in result.warnings)


def test_batch_stops_using_nim_after_a_fatal_error(tmp_path, monkeypatch):
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    for name in ("a.mp3", "b.mp3", "c.mp3"):
        (media_dir / name).write_bytes(b"x")
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda *a, **k: object())
    nim = FailingNim(fatal=True)
    results = run_batch(str(media_dir), str(tmp_path / "out"),
                        PipelineOptions(make_chapters=False), nim_client=nim)
    assert [r.ok for r in results] == [True, True, True]
    assert nim.calls == 1  # the bad key is not retried on b.mp3 and c.mp3
    assert results[0].nim_fatal
    assert any("NIM disabled for the rest of the batch" in w for w in results[1].warnings)
    assert any("NIM disabled for the rest of the batch" in w for w in results[2].warnings)


def test_same_stem_inputs_get_separate_folders(tmp_path, monkeypatch):
    media_dir = tmp_path / "d"
    media_dir.mkdir()
    (media_dir / "ep01.mp3").write_bytes(b"x")
    (media_dir / "ep01.wav").write_bytes(b"xy")
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda *a, **k: object())
    results = run_batch(str(media_dir), str(tmp_path / "out"), PipelineOptions(use_nim=False))
    folders = sorted(Path(r.output_dir).name for r in results)
    assert folders == ["ep01", "ep01-wav"]
    for folder in folders:
        assert (tmp_path / "out" / folder / "transcript.txt").exists()
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["batch_report.md", "ep01", "ep01-wav"]


def test_batch_output_names_are_unique_and_deterministic():
    from transcribe_studio.pipeline import batch_output_names

    files = [Path(n) for n in sorted(["ep01-wav.mp3", "ep01.mp3", "ep01.wav", "EP01.MP4", "talk.m4a"])]
    names = batch_output_names(files)
    assert len(set(n.lower() for n in names.values())) == len(files)
    assert names[Path("talk.m4a")] == "talk"
    # The first file (in sorted order) with a stem keeps the bare stem.
    assert names[Path("EP01.MP4")] == "EP01"
    assert names[Path("ep01.mp3")] == "ep01-mp3"
    assert names[Path("ep01-wav.mp3")] == "ep01-wav"
    assert names[Path("ep01.wav")] == "ep01-wav-2"  # collides with the real stem above
    assert batch_output_names(files) == names


def test_skip_existing_resumes_a_batch(tmp_path, monkeypatch):
    media_dir = tmp_path / "d"
    media_dir.mkdir()
    for name in ("a.mp3", "b.mp3"):
        (media_dir / name).write_bytes(b"audio")
    loads, calls = [], []
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda *a, **k: loads.append(1) or object())

    def counting(media_path, *a, **k):
        calls.append(Path(media_path).name)
        return _fake_transcript()

    monkeypatch.setattr(pipeline_mod, "transcribe", counting)
    out = tmp_path / "out"
    opts = PipelineOptions(use_nim=False)
    first = run_batch(str(media_dir), str(out), opts)
    assert calls == ["a.mp3", "b.mp3"] and not any(r.skipped for r in first)

    # Re-run: everything is up to date, so neither Whisper nor transcribe run.
    loads.clear(), calls.clear()
    second = run_batch(str(media_dir), str(out), opts, skip_existing=True)
    assert [r.skipped for r in second] == [True, True]
    assert calls == [] and loads == []
    assert "2 already up to date" in (out / "batch_report.md").read_text(encoding="utf-8")

    # A changed input and a half-written folder are both redone.
    (media_dir / "a.mp3").write_bytes(b"a different recording")
    (out / "b" / "captions.srt").unlink()
    third = run_batch(str(media_dir), str(out), opts, skip_existing=True)
    assert calls == ["a.mp3", "b.mp3"]
    assert [r.skipped for r in third] == [False, False]


def test_manifest_records_the_run(tmp_path):
    out = tmp_path / "out"
    media = tmp_path / "show.mp3"
    media.write_bytes(b"12345")
    result = run(str(media), str(out), PipelineOptions(use_nim=False, make_summary=False))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"] == {"name": "show.mp3", "size": 5}
    assert manifest["outputs"] == [Path(o).name for o in result.outputs]
    assert manifest["options"]["use_nim"] is False
    assert manifest["language"] == "en"


def test_diarized_speakers_are_saved_in_transcript_json(tmp_path):
    out = tmp_path / "out"
    run("fake_media.mp3", str(out),
        PipelineOptions(diarize=True, diarize_gap=0.3, use_nim=False, make_summary=False))
    data = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    speakers = [s["speaker"] for s in data["segments"]]
    assert all(speakers), speakers
    assert speakers == ["Speaker 1", "Speaker 1", "Speaker 2"]  # 0.5s gap before seg 3 > 0.3
    assert data["diarization"] == "pause-heuristic"
    assert "Speaker 2:" in (out / "transcript.txt").read_text(encoding="utf-8")


def test_chunk_options_reach_the_transcriber(tmp_path, monkeypatch):
    seen = {}

    def spy(media_path, **kwargs):
        seen.update(kwargs)
        return _fake_transcript()

    monkeypatch.setattr(pipeline_mod, "transcribe", spy)
    run("fake_media.mp3", str(tmp_path / "o"),
        PipelineOptions(chunk_length=600, chunk_overlap=4, use_nim=False, make_summary=False))
    assert seen["chunk_length"] == 600 and seen["chunk_overlap"] == 4
