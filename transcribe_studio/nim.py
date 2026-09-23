"""Tiny, resilient OpenAI-compatible client for NVIDIA NIM.

Uses only the standard library (``urllib``) so the package imports and its pure
functions run without any third-party HTTP dependency. NIM speaks the OpenAI
chat-completions protocol, so this is a thin POST wrapper around
``/chat/completions`` with the robustness a free, rate-limited tier needs:

- configuration (``NIM_BASE_URL``, ``NIM_MODEL``, ``NIM_TIMEOUT``,
  ``NIM_MAX_RETRIES``) is read when the client is *built*, not at import, so a
  ``.env`` loaded later still applies;
- HTTP 429 / 408 / 5xx, timeouts and dropped connections are retried with
  exponential backoff and jitter, honouring ``Retry-After``;
- a rejected key (401/403) or unknown model (404) fails fast with a message
  that says what to fix, flagged ``fatal`` so batch runs stop asking.

Get a free key (starts with ``nvapi-``) in about two minutes at
https://build.nvidia.com — no credit card required.
"""

from __future__ import annotations

import email.utils
import http.client
import json
import os
import random
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional

from .config import API_KEY_ENV, env_number, nim_base_url, nim_model

_SIGNUP_HINT = (
    "No NVIDIA NIM API key found.\n"
    f"  1. Set {API_KEY_ENV} in your environment or .env file.\n"
    "  2. Get a free key at https://build.nvidia.com (starts with 'nvapi-').\n"
    "Transcripts, subtitles, offline chapters and the offline extractive summary\n"
    "all work without a key; only NIM summaries, NIM chapters and translation need it."
)

# Statuses worth retrying: request timeout, rate limit, and server-side errors.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class MissingApiKey(RuntimeError):
    """Raised when a NIM feature is requested without an API key configured."""


class NimError(RuntimeError):
    """Raised when the NIM endpoint returns an error or a malformed response.

    ``status`` is the HTTP status when there was one. ``fatal`` marks errors that
    retrying later will not fix (bad key, unknown model), so callers such as a
    batch run can stop calling NIM for the remaining files.
    """

    def __init__(self, message: str, status: Optional[int] = None, fatal: bool = False,
                 attempts: int = 1) -> None:
        super().__init__(message)
        self.status = status
        self.fatal = fatal
        self.attempts = attempts


class _Transient(Exception):
    """One failed attempt that is worth retrying."""

    def __init__(self, reason: str, status: Optional[int] = None,
                 retry_after: Optional[float] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status
        self.retry_after = retry_after


def parse_retry_after(value: Optional[str], now: Optional[float] = None) -> Optional[float]:
    """Parse a ``Retry-After`` header (delta-seconds or HTTP-date) into seconds."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    now = time.time() if now is None else now
    return max(0.0, when.timestamp() - now)


def _describe_network_error(reason: object, timeout: float) -> str:
    if isinstance(reason, ConnectionRefusedError):
        return "connection refused"
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return f"timed out after {timeout:g}s"
    if isinstance(reason, socket.gaierror):
        return "host name could not be resolved (are you offline?)"
    if isinstance(reason, ConnectionResetError):
        return "connection reset by the server"
    return str(reason) or reason.__class__.__name__


def _error_detail(exc: urllib.error.HTTPError) -> str:
    """Best-effort one-line explanation from an error body."""
    try:
        raw = exc.read().decode("utf-8", "replace") if exc.fp else ""
    except OSError:
        raw = ""
    detail = raw.strip()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        data = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            detail = str(err["message"])
        elif isinstance(err, str):
            detail = err
        elif data.get("detail"):
            detail = str(data["detail"])
        elif data.get("message"):
            detail = str(data["message"])
    detail = " ".join(detail.split())
    return detail[:300] or (exc.reason if isinstance(exc.reason, str) else "")


class NimClient:
    """Minimal chat-completions client for NVIDIA NIM with retries."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff: float = 1.0,
        max_backoff: float = 30.0,
        max_retry_after: float = 90.0,
        sleep: Optional[Callable[[float], None]] = None,
        rng: Optional[Callable[[], float]] = None,
        on_retry: Optional[Callable[[int, float, str], None]] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "")
        if not self.api_key:
            raise MissingApiKey(_SIGNUP_HINT)
        self.model = model or nim_model()
        self.base_url = (base_url or nim_base_url()).rstrip("/")
        self.timeout = float(timeout if timeout is not None else env_number("NIM_TIMEOUT", 120.0))
        retries = max_retries if max_retries is not None else env_number("NIM_MAX_RETRIES", 3)
        self.max_retries = max(0, int(retries))
        self.backoff = max(0.0, backoff)
        self.max_backoff = max_backoff
        self.max_retry_after = max_retry_after
        self._sleep = sleep or time.sleep
        self._rng = rng or random.random
        self.on_retry = on_retry
        self.requests_sent = 0

    @classmethod
    def from_env(cls, model: Optional[str] = None, **kwargs) -> "NimClient":
        """Build a client from the environment, raising :class:`MissingApiKey` if unset."""
        return cls(model=model, **kwargs)

    @staticmethod
    def available() -> bool:
        """True if an API key is configured (does not validate it)."""
        return bool(os.environ.get(API_KEY_ENV))

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------
    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with "equal jitter": half fixed, half random."""
        ceiling = min(self.max_backoff, self.backoff * (2 ** attempt))
        return ceiling / 2 + self._rng() * ceiling / 2

    def _post_once(self, data: bytes) -> dict:
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        self.requests_sent += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            code = exc.code
            detail = _error_detail(exc)
            if code in (401, 403):
                raise NimError(
                    f"NIM rejected the API key (HTTP {code}). Check {API_KEY_ENV} in your "
                    "environment or .env: free keys start with 'nvapi-' and can be "
                    "regenerated at https://build.nvidia.com.",
                    status=code,
                    fatal=True,
                ) from None
            if code == 404:
                raise NimError(
                    f"NIM returned HTTP 404 for model '{self.model}' at {self.base_url}. "
                    "Check NIM_MODEL and NIM_BASE_URL.",
                    status=code,
                    fatal=True,
                ) from None
            if code in RETRYABLE_STATUS:
                retry_after = parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
                raise _Transient(f"HTTP {code}" + (f": {detail}" if detail else ""),
                                 status=code, retry_after=retry_after) from None
            raise NimError(f"NIM returned HTTP {code}: {detail}", status=code) from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise NimError(f"TLS certificate check failed for {self.base_url}: {exc.reason}") from None
            raise _Transient(_describe_network_error(exc.reason, self.timeout)) from None
        except (TimeoutError, socket.timeout) as exc:
            raise _Transient(_describe_network_error(exc, self.timeout)) from None
        except (ConnectionError, http.client.HTTPException) as exc:
            raise _Transient(_describe_network_error(exc, self.timeout)) from None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            snippet = " ".join(raw.split())[:120]
            raise NimError(f"NIM answered with something that is not JSON: {snippet!r}") from None

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> str:
        """Send a chat completion and return the assistant message content.

        Transient failures are retried up to ``max_retries`` times. Raises
        :class:`NimError` once retries are exhausted or on a non-retryable error.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        attempts = self.max_retries + 1
        body: Optional[dict] = None
        for attempt in range(attempts):
            try:
                body = self._post_once(data)
                break
            except _Transient as transient:
                if attempt == attempts - 1:
                    raise self._exhausted(transient, attempts) from None
                if transient.retry_after is not None:
                    delay = min(transient.retry_after, self.max_retry_after)
                else:
                    delay = self._backoff_delay(attempt)
                if self.on_retry is not None:
                    self.on_retry(attempt + 1, delay, transient.reason)
                self._sleep(delay)

        try:
            content = body["choices"][0]["message"]["content"]  # type: ignore[index]
        except (KeyError, IndexError, TypeError):
            raise NimError(f"Unexpected NIM response shape: {str(body)[:200]}") from None
        return content or ""

    def _exhausted(self, transient: _Transient, attempts: int) -> NimError:
        tries = f"{attempts} attempt{'s' if attempts != 1 else ''}"
        if transient.status == 429:
            msg = (
                f"NIM kept rate-limiting the request (HTTP 429) after {tries}. The free "
                "tier allows a limited number of requests per minute: wait a minute and "
                "retry, or raise NIM_MAX_RETRIES."
            )
        elif transient.status is not None:
            msg = f"NIM failed with {transient.reason} after {tries}."
        else:
            msg = f"Could not reach NIM at {self.base_url}: {transient.reason} (after {tries})."
        return NimError(msg, status=transient.status, attempts=attempts)

    def complete(self, system: str, user: str, **kwargs) -> str:
        """Convenience wrapper for a single system+user turn."""
        return self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
