"""Shared fixtures and a fake NIM client so tests never touch the network."""

from __future__ import annotations

import os
import sys

import pytest

# Make the repo root importable (so `import src...` works) regardless of the
# directory pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Segment, Word  # noqa: E402


class FakeNim:
    """A stand-in NIM client.

    `responder` is a callable taking the message list and returning the assistant
    string. Records every call for assertions.
    """

    def __init__(self, responder):
        self.responder = responder
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.responder(messages)

    def complete(self, system, user, **kwargs):
        return self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )


@pytest.fixture
def make_segments():
    def _make(spec):
        """spec: list of (start, end, text) tuples -> list[Segment]."""
        segs = []
        for start, end, text in spec:
            words = []
            tokens = text.split()
            if tokens:
                step = (end - start) / len(tokens)
                for i, tok in enumerate(tokens):
                    words.append(Word(start=start + i * step, end=start + (i + 1) * step, word=" " + tok))
            segs.append(Segment(start=start, end=end, text=text, words=words))
        return segs

    return _make


@pytest.fixture
def fake_nim():
    return FakeNim
