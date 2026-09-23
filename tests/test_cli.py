"""The command-line interface: exit codes, friendly errors, real subprocesses."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from transcribe_studio import cli

ROOT = Path(__file__).resolve().parents[1]


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run_cli(args, cwd, **env_overrides):
    env = {k: v for k, v in os.environ.items()
           if k not in ("NVIDIA_API_KEY", "NIM_BASE_URL", "NIM_MODEL", "NIM_MAX_RETRIES")}
    env.update({"NO_PROXY": "127.0.0.1,localhost", "PYTHONIOENCODING": "utf-8"})
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, str(ROOT / "cli.py"), *args],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )


@pytest.fixture
def no_nim_env(monkeypatch):
    for k in ("NVIDIA_API_KEY", "NIM_BASE_URL", "NIM_MODEL", "NIM_MAX_RETRIES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(ROOT / "tests")  # no .env here, so load_env() is a no-op


@pytest.mark.slow
def test_unreachable_nim_prints_one_friendly_line(tmp_path):
    (tmp_path / "talk.txt").write_text("We talked about backups. Then we talked about tomatoes.",
                                       encoding="utf-8")
    port = _closed_port()
    proc = _run_cli(
        ["summarize", "talk.txt", "-o", "out"],
        cwd=tmp_path,
        NVIDIA_API_KEY="nvapi-fake",
        NIM_BASE_URL=f"http://127.0.0.1:{port}/v1",
        NIM_MAX_RETRIES="0",
    )
    assert proc.returncode == cli.EXIT_NIM
    assert "Traceback" not in proc.stderr
    lines = [l for l in proc.stderr.splitlines() if l.strip()]
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith("error: Could not reach NIM")
    assert f"127.0.0.1:{port}" in lines[0]


@pytest.mark.slow
def test_nim_base_url_from_dotenv_is_honoured(tmp_path):
    # Regression: NIM_BASE_URL in .env used to be ignored because it was read
    # at import time, before the CLI loaded .env.
    port = _closed_port()
    (tmp_path / ".env").write_text(
        f"NVIDIA_API_KEY=nvapi-fake\nNIM_BASE_URL=http://127.0.0.1:{port}/v1\nNIM_MAX_RETRIES=0\n",
        encoding="utf-8",
    )
    (tmp_path / "talk.txt").write_text("Short talk.", encoding="utf-8")
    proc = _run_cli(["summarize", "talk.txt"], cwd=tmp_path)
    assert proc.returncode == cli.EXIT_NIM
    assert f"127.0.0.1:{port}" in proc.stderr
    assert "integrate.api.nvidia.com" not in proc.stderr


def test_missing_media_is_exit_4(tmp_path, no_nim_env, capsys):
    code = cli.main(["transcribe", str(tmp_path / "nope.mp3"), "-o", str(tmp_path / "o")])
    assert code == cli.EXIT_NOT_FOUND
    assert "Media file not found" in capsys.readouterr().err


def test_whisper_load_failure_is_friendly(tmp_path, no_nim_env, capsys, monkeypatch):
    from transcribe_studio import transcribe as transcribe_mod

    media = tmp_path / "clip.wav"
    media.write_bytes(b"RIFF")

    def fail(*a, **k):
        raise transcribe_mod.WhisperUnavailable("could not load the Whisper model 'base' (offline)")

    monkeypatch.setattr(transcribe_mod, "_load_model", fail)
    code = cli.main(["transcribe", str(media), "-o", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == cli.EXIT_WHISPER
    assert err.startswith("error: could not load the Whisper model")
    assert "Traceback" not in err


def test_unexpected_errors_hint_at_debug_and_debug_reraises(tmp_path, no_nim_env, capsys, monkeypatch):
    def explode(*a, **k):
        raise ZeroDivisionError("kaboom")

    monkeypatch.setattr(cli, "transcribe", explode)
    media = tmp_path / "clip.wav"
    media.write_bytes(b"RIFF")
    code = cli.main(["transcribe", str(media), "-o", str(tmp_path / "o")])
    err = capsys.readouterr().err
    assert code == cli.EXIT_PARTIAL
    assert "unexpected ZeroDivisionError: kaboom" in err and "--debug" in err
    with pytest.raises(ZeroDivisionError):
        cli.main(["--debug", "transcribe", str(media), "-o", str(tmp_path / "o")])


def test_version_flag(capsys):
    from transcribe_studio import __version__

    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_chunk_overlap_must_be_smaller_than_chunk_length(tmp_path, no_nim_env, capsys):
    with pytest.raises(SystemExit) as info:
        cli.main(["transcribe", "x.wav", "--chunk-length", "10", "--chunk-overlap", "10"])
    assert info.value.code == 2
    assert "--chunk-overlap" in capsys.readouterr().err


def test_summarize_local_writes_a_labelled_extractive_summary(tmp_path, no_nim_env, capsys):
    talk = tmp_path / "talk.txt"
    talk.write_text("Backups fail silently. We need to test a restore every month. "
                    "Snapshots are not backups. Hay que regar temprano.", encoding="utf-8")
    code = cli.main(["summarize", str(talk), "--local", "-o", str(tmp_path / "o")])
    assert code == 0
    md = (tmp_path / "o" / "summary.md").read_text(encoding="utf-8")
    assert "Offline extractive summary" in md
    assert "- [ ] We need to test a restore every month." in md
    assert "- [ ] Hay que regar temprano." in md
    assert "note:" not in capsys.readouterr().err  # explicit --local: no nagging


def test_summarize_without_a_key_goes_offline_with_a_note(tmp_path, no_nim_env, capsys):
    talk = tmp_path / "talk.txt"
    talk.write_text("One sentence about backups. Another one about restores.", encoding="utf-8")
    assert cli.main(["summarize", str(talk), "-o", str(tmp_path / "o")]) == 0
    assert "NVIDIA_API_KEY not set" in capsys.readouterr().err
    assert (tmp_path / "o" / "summary.md").exists()


# ---------------------------------------------------------------------------
# Transcript-first: everything but `transcribe` works from a saved transcript.json
# ---------------------------------------------------------------------------
SAMPLE = ROOT / "examples" / "sample-podcast.transcript.json"
ARTIFACTS = ("transcript.txt", "transcript.json", "captions.srt", "captions.vtt",
             "summary.md", "chapters.md", "manifest.json")


def test_sample_transcript_is_small_valid_and_bilingual():
    from transcribe_studio.config import load_json

    assert SAMPLE.stat().st_size < 30_000
    tr = load_json(SAMPLE)
    assert len(tr.segments) >= 20 and tr.duration > 180
    assert all(seg.words for seg in tr.segments)  # word timings for word-accurate captions
    assert "backup" in tr.text.lower() and "tomates" in tr.text.lower()


@pytest.mark.slow
def test_all_on_the_sample_needs_no_whisper_and_no_network(tmp_path):
    # faster_whisper is made unimportable: the transcript-first path must not touch it.
    code = (
        "import sys; sys.modules['faster_whisper'] = None\n"
        "from transcribe_studio.cli import main\n"
        f"raise SystemExit(main(['all', r'{SAMPLE}', '--no-nim', '-o', r'{tmp_path / 'demo'}']))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "demo"
    for name in ARTIFACTS:
        assert (out / name).exists(), name
    assert "Captions: " in proc.stdout and "0 overlaps" in proc.stdout

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Offline extractive summary" in summary
    assert "- [ ] We need to test a restore from the offsite copy every month" in summary
    assert "- [ ] Hay que regar temprano por la mañana" in summary
    assert "todo por hoy" not in summary.split("## Action items")[-1]

    chapters = (out / "chapters.md").read_text(encoding="utf-8").splitlines()[2:]
    assert len(chapters) >= 3 and chapters[0].startswith("00:00 ")
    switch = next(line for line in chapters if "Tomates" in line or "Balcón" in line)
    assert switch.startswith("02:")  # the Spanish segment starts just after 2:00


def test_sample_captions_are_strictly_valid(tmp_path, no_nim_env):
    from tests.test_subtitles_words import parse_srt_strict, parse_vtt_strict

    assert cli.main(["subs", str(SAMPLE), "-o", str(tmp_path)]) == 0
    srt = parse_srt_strict((tmp_path / "captions.srt").read_text(encoding="utf-8"))
    vtt = parse_vtt_strict((tmp_path / "captions.vtt").read_text(encoding="utf-8"))
    assert len(srt) == len(vtt) > 20
    assert all(a[1] <= b[0] for a, b in zip(srt, srt[1:]))


def test_subs_with_speaker_labels_from_a_transcript(tmp_path, no_nim_env, capsys):
    assert cli.main(["subs", str(SAMPLE), "--speaker-labels", "-o", str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert "pause heuristic" in captured.err
    assert "Captions: " in captured.out
    srt = (tmp_path / "captions.srt").read_text(encoding="utf-8")
    assert "[Speaker 1] Welcome back" in srt and "[Speaker 2] Thanks, Leo." in srt
    assert "<v Speaker 2>Thanks, Leo." in (tmp_path / "captions.vtt").read_text(encoding="utf-8")


def test_chapters_and_summarize_accept_the_transcript(tmp_path, no_nim_env, capsys):
    assert cli.main(["chapters", str(SAMPLE), "--local", "-o", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("00:00 ")
    assert cli.main(["summarize", str(SAMPLE), "--local", "-o", str(tmp_path)]) == 0
    assert (tmp_path / "summary.md").exists() and (tmp_path / "chapters.md").exists()


def test_outputs_are_byte_identical_across_runs(tmp_path, no_nim_env):
    for run_dir in ("a", "b"):
        assert cli.main(["all", str(SAMPLE), "--no-nim", "-o", str(tmp_path / run_dir)]) == 0
    for name in ("chapters.md", "summary.md", "captions.srt", "captions.vtt", "transcript.txt"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name


def test_rebuilding_from_a_diarized_transcript_keeps_the_speakers(tmp_path, no_nim_env):
    first, second = tmp_path / "first", tmp_path / "second"
    assert cli.main(["all", str(SAMPLE), "--no-nim", "--diarize", "-o", str(first)]) == 0
    assert cli.main(["all", str(first / "transcript.json"), "--no-nim", "-o", str(second)]) == 0
    assert "Speaker 2: Thanks, Leo." in (second / "transcript.txt").read_text(encoding="utf-8")


def test_skip_existing_for_a_single_input(tmp_path, no_nim_env, capsys):
    out = tmp_path / "o"
    assert cli.main(["all", str(SAMPLE), "--no-nim", "-o", str(out)]) == 0
    capsys.readouterr()
    assert cli.main(["all", str(SAMPLE), "--no-nim", "--skip-existing", "-o", str(out)]) == 0
    assert "Up to date, skipped" in capsys.readouterr().out


def test_wrong_inputs_get_friendly_errors(tmp_path, no_nim_env, capsys):
    assert cli.main(["transcribe", str(SAMPLE), "-o", str(tmp_path)]) == cli.EXIT_USAGE
    assert "already a transcript" in capsys.readouterr().err

    notes = tmp_path / "notes.txt"
    notes.write_text("hello", encoding="utf-8")
    assert cli.main(["all", str(notes), "-o", str(tmp_path)]) == cli.EXIT_USAGE
    assert "Use `summarize` for plain text" in capsys.readouterr().err

    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a transcript"}', encoding="utf-8")
    assert cli.main(["subs", str(bad), "-o", str(tmp_path)]) == cli.EXIT_BAD_INPUT
    err = capsys.readouterr().err
    assert err.startswith("error: bad.json: not a transcript") and "Traceback" not in err

    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    assert cli.main(["chapters", str(broken), "--local", "-o", str(tmp_path)]) == cli.EXIT_BAD_INPUT
    assert cli.main(["subs", str(tmp_path / "missing.json")]) == cli.EXIT_NOT_FOUND


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_batch_through_the_cli_with_a_fake_whisper(tmp_path, no_nim_env, capsys, monkeypatch):
    from tests.test_transcribe_ffmpeg import FakeWhisper
    from transcribe_studio import transcribe as transcribe_mod

    media = tmp_path / "media"
    media.mkdir()
    for name, freq in (("ep01.wav", 300), ("ep01.flac", 500), ("ep02.wav", 700)):
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", f"sine=frequency={freq}:duration=6", str(media / name)], check=True)
    (media / "readme.txt").write_text("not media", encoding="utf-8")
    loads = []
    monkeypatch.setattr(transcribe_mod, "_load_model", lambda *a, **k: loads.append(1) or FakeWhisper("en"))

    out = tmp_path / "out"
    assert cli.main(["all", str(media), "--no-nim", "-o", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "[1/3] ep01.flac: ok" in printed and "Batch complete: 3/3 succeeded" in printed
    assert loads == [1]  # one model for the whole batch
    assert sorted(p.name for p in out.iterdir()) == ["batch_report.md", "ep01", "ep01-wav", "ep02"]
    for folder in ("ep01", "ep01-wav", "ep02"):
        for name in ARTIFACTS:
            assert (out / folder / name).exists(), f"{folder}/{name}"

    # Resume: nothing left to do, so Whisper is not even loaded.
    loads.clear()
    assert cli.main(["all", str(media), "--no-nim", "--skip-existing", "-o", str(out)]) == 0
    assert loads == [] and "up to date, skipped" in capsys.readouterr().out
