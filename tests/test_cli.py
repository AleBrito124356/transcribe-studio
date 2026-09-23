"""The command-line interface: exit codes, friendly errors, real subprocesses."""

from __future__ import annotations

import os
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
