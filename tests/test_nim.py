"""NIM client: runtime config, retries/backoff, and error messages.

These tests talk to a real HTTP server (stdlib ``http.server`` on 127.0.0.1
with an ephemeral port), so the actual urllib transport, headers and status
handling are exercised. Nothing leaves the machine.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from transcribe_studio.config import DEFAULT_NIM_BASE_URL, load_env
from transcribe_studio.nim import MissingApiKey, NimClient, NimError, parse_retry_after

NIM_ENV = ("NVIDIA_API_KEY", "NIM_BASE_URL", "NIM_MODEL", "NIM_TIMEOUT", "NIM_MAX_RETRIES")


@pytest.fixture(autouse=True)
def clean_nim_env(monkeypatch):
    """Isolate every test from the developer's real NIM settings and proxies."""
    saved = {k: os.environ.get(k) for k in NIM_ENV}
    for k in NIM_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    yield
    for k, v in saved.items():  # load_env() writes os.environ directly
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class ScriptedNim:
    """A tiny OpenAI-compatible server that replays scripted responses."""

    def __init__(self, script):
        self.script = list(script)  # items: (status, headers, body)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - http.server API
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append({
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "body": body,
                })
                status, headers, payload = (
                    outer.script.pop(0) if outer.script else (500, {}, {"error": "script exhausted"})
                )
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args):  # keep pytest output clean
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def _ok(content):
    return (200, {}, {"choices": [{"message": {"role": "assistant", "content": content}}]})


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_retries_429_then_succeeds_honouring_retry_after():
    sleeps = []
    script = [
        (429, {"Retry-After": "0"}, {"error": {"message": "Too many requests"}}),
        (429, {"Retry-After": "0"}, {"error": {"message": "Too many requests"}}),
        _ok("hola mundo"),
    ]
    with ScriptedNim(script) as srv:
        client = NimClient(api_key="nvapi-test", base_url=srv.base_url, model="m/x",
                           max_retries=3, sleep=sleeps.append)
        reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply == "hola mundo"
    assert len(srv.requests) == 3
    assert sleeps == [0.0, 0.0]  # Retry-After: 0 wins over the exponential backoff
    first = srv.requests[0]
    assert first["path"] == "/v1/chat/completions"
    assert first["auth"] == "Bearer nvapi-test"
    assert first["body"]["model"] == "m/x"
    assert first["body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_5xx_uses_exponential_backoff_with_jitter():
    sleeps = []
    script = [(503, {}, {"detail": "overloaded"}), (502, {}, b"bad gateway"), _ok("ok")]
    with ScriptedNim(script) as srv:
        client = NimClient(api_key="k", base_url=srv.base_url, max_retries=2, backoff=1.0,
                           sleep=sleeps.append, rng=lambda: 1.0)
        assert client.chat([{"role": "user", "content": "x"}]) == "ok"
    # equal jitter with rng()=1.0 -> the full exponential ceiling: 1s, then 2s
    assert sleeps == [1.0, 2.0]


def test_rate_limit_exhausted_gives_clear_error():
    script = [(429, {"Retry-After": "0"}, {"error": "slow down"})] * 3
    with ScriptedNim(script) as srv:
        client = NimClient(api_key="k", base_url=srv.base_url, max_retries=2, sleep=lambda s: None)
        with pytest.raises(NimError) as info:
            client.chat([{"role": "user", "content": "x"}])
    assert len(srv.requests) == 3
    assert info.value.status == 429
    assert "rate-limit" in str(info.value)
    assert "3 attempts" in str(info.value)
    assert not info.value.fatal


@pytest.mark.parametrize("status", [401, 403])
def test_bad_key_fails_fast_with_a_key_hint(status):
    with ScriptedNim([(status, {}, {"detail": "Authentication failed"})]) as srv:
        client = NimClient(api_key="nvapi-wrong", base_url=srv.base_url, sleep=lambda s: None)
        with pytest.raises(NimError) as info:
            client.chat([{"role": "user", "content": "x"}])
    assert len(srv.requests) == 1  # never retried
    msg = str(info.value)
    assert "NVIDIA_API_KEY" in msg and "nvapi-" in msg
    assert info.value.fatal and info.value.status == status
    assert "nvapi-wrong" not in msg  # never echo the key back


def test_unknown_model_is_fatal_and_names_the_model():
    with ScriptedNim([(404, {}, {"detail": "Function not found"})]) as srv:
        client = NimClient(api_key="k", base_url=srv.base_url, model="nope/model")
        with pytest.raises(NimError) as info:
            client.chat([{"role": "user", "content": "x"}])
    assert info.value.fatal
    assert "nope/model" in str(info.value) and "NIM_MODEL" in str(info.value)


def test_other_4xx_is_not_retried():
    with ScriptedNim([(400, {}, {"error": {"message": "max_tokens too large"}})]) as srv:
        client = NimClient(api_key="k", base_url=srv.base_url, sleep=lambda s: None)
        with pytest.raises(NimError, match="max_tokens too large"):
            client.chat([{"role": "user", "content": "x"}])
    assert len(srv.requests) == 1


def test_non_json_200_is_reported():
    with ScriptedNim([(200, {}, b"<html>captive portal</html>")]) as srv:
        client = NimClient(api_key="k", base_url=srv.base_url)
        with pytest.raises(NimError, match="not JSON"):
            client.chat([{"role": "user", "content": "x"}])


def test_network_errors_are_retried_with_backoff(monkeypatch):
    import urllib.error
    import urllib.request

    calls = []

    def refuse(req, timeout):
        calls.append(req.full_url)
        raise urllib.error.URLError(ConnectionRefusedError(10061, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    sleeps, retries = [], []
    client = NimClient(
        api_key="k",
        base_url="http://127.0.0.1:9/v1",
        max_retries=2,
        sleep=sleeps.append,
        rng=lambda: 0.0,
        on_retry=lambda attempt, delay, reason: retries.append((attempt, reason)),
    )
    with pytest.raises(NimError) as info:
        client.chat([{"role": "user", "content": "x"}])
    assert len(calls) == 3
    assert sleeps == [0.5, 1.0]  # equal-jitter floor with rng()=0: half of 1s, half of 2s
    assert retries == [(1, "connection refused"), (2, "connection refused")]
    assert "3 attempts" in str(info.value)


def test_real_closed_port_gives_one_line_error():
    port = _closed_port()
    client = NimClient(api_key="k", base_url=f"http://127.0.0.1:{port}/v1", max_retries=0, timeout=5)
    with pytest.raises(NimError) as info:
        client.chat([{"role": "user", "content": "x"}])
    msg = str(info.value)
    assert f"127.0.0.1:{port}" in msg and "connection refused" in msg
    assert len(msg.splitlines()) == 1
    assert client.requests_sent == 1


def test_env_retry_and_timeout_settings(monkeypatch):
    monkeypatch.setenv("NIM_MAX_RETRIES", "0")
    monkeypatch.setenv("NIM_TIMEOUT", "7.5")
    client = NimClient(api_key="k")
    assert client.max_retries == 0
    assert client.timeout == 7.5
    monkeypatch.setenv("NIM_MAX_RETRIES", "not-a-number")
    assert NimClient(api_key="k").max_retries == 3  # garbage falls back to the default


def test_dotenv_loaded_after_import_changes_base_url_and_model(tmp_path):
    # The CLI imports everything first and calls load_env() afterwards; the
    # settings in .env must still reach the client.
    assert NimClient(api_key="k").base_url == DEFAULT_NIM_BASE_URL
    env = tmp_path / ".env"
    env.write_text(
        "NVIDIA_API_KEY=nvapi-from-dotenv\n"
        "NIM_BASE_URL=http://127.0.0.1:1/v1/\n"
        "NIM_MODEL=meta/llama-3.1-8b-instruct\n",
        encoding="utf-8",
    )
    load_env(env)
    client = NimClient()
    assert client.base_url == "http://127.0.0.1:1/v1"
    assert client.model == "meta/llama-3.1-8b-instruct"
    assert client.api_key == "nvapi-from-dotenv"


def test_missing_key_raises():
    assert not NimClient.available()
    with pytest.raises(MissingApiKey, match="build.nvidia.com"):
        NimClient()


def test_parse_retry_after_forms():
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("-4") == 0.0
    assert parse_retry_after("garbage") is None
    # HTTP-date form, relative to a fixed "now".
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:10 GMT", now=1445412480.0) == pytest.approx(10.0)
