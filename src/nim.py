"""Tiny OpenAI-compatible client for NVIDIA NIM.

Uses only the standard library (``urllib``) so the package imports and its pure
functions run without any third-party HTTP dependency. NIM speaks the OpenAI
chat-completions protocol, so this is a thin POST wrapper around
``/chat/completions``.

Get a free key (starts with ``nvapi-``) in about two minutes at
https://build.nvidia.com — no credit card required.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

from .config import API_KEY_ENV, DEFAULT_NIM_MODEL, NIM_BASE_URL

_SIGNUP_HINT = (
    "No NVIDIA NIM API key found.\n"
    f"  1. Set {API_KEY_ENV} in your environment or .env file.\n"
    "  2. Get a free key at https://build.nvidia.com (starts with 'nvapi-').\n"
    "Local transcription and subtitles still work without a key; only the\n"
    "summary, chapters (NIM mode), and translation features need it."
)


class MissingApiKey(RuntimeError):
    """Raised when a NIM feature is requested without an API key configured."""


class NimError(RuntimeError):
    """Raised when the NIM endpoint returns an error or malformed response."""


class NimClient:
    """Minimal chat-completions client for NVIDIA NIM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get(API_KEY_ENV, "")
        if not self.api_key:
            raise MissingApiKey(_SIGNUP_HINT)
        self.model = model or os.environ.get("NIM_MODEL", DEFAULT_NIM_MODEL)
        self.base_url = (base_url or NIM_BASE_URL).rstrip("/")
        self.timeout = timeout

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> "NimClient":
        """Build a client from the environment, raising :class:`MissingApiKey` if unset."""
        return cls(model=model)

    @staticmethod
    def available() -> bool:
        """True if an API key is configured (does not validate it)."""
        return bool(os.environ.get(API_KEY_ENV))

    def chat(
        self,
        messages: List[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ) -> str:
        """Send a chat completion and return the assistant message content."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network path
            detail = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
            raise NimError(f"NIM returned HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise NimError(f"Could not reach NIM at {self.base_url}: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:  # pragma: no cover
            raise NimError(f"Unexpected NIM response shape: {body!r}") from exc

    def complete(self, system: str, user: str, **kwargs) -> str:
        """Convenience wrapper for a single system+user turn."""
        return self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
