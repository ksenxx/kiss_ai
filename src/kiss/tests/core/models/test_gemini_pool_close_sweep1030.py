# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A Gemini model that rebuilds its client closes the pool the old client sent through.

``GeminiModel._build_client`` gives every ``genai.Client`` a tracking
httpx pool of its own, and ``genai.Client.close()`` leaves an httpx
client it was handed open (the SDK lets the caller own it).  So when
``Model._reuse_client`` replaced the client after a key change and
closed the old one, the old pool kept its keep-alive sockets until the
garbage collector reached it.  A real model runs a turn against the
local Gemini endpoint, switches key, and runs another.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from kiss.core.models.gemini_model import GeminiModel
from kiss.tests.core.models.gemini_sse_harness import GeminiScript, chunk, serve, text_part

_MODEL = "gemini-pool-under-test"


@pytest.fixture
def gemini_endpoint() -> Generator[tuple[str, GeminiScript]]:
    """A real local Gemini endpoint for one test."""
    yield from serve()


def _turn(model: GeminiModel, script: GeminiScript, prompt: str, answer: str) -> None:
    """Start a run with *prompt* and check the model answers *answer*."""
    model.initialize(prompt)
    script.play([chunk([text_part(answer)], finish_reason="STOP")])
    content, _ = model.generate()
    assert content == answer


def test_a_key_change_closes_the_replaced_clients_pool(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript],
) -> None:
    """The pool of the client a key change replaced is closed; the new one serves."""
    base_url, script = gemini_endpoint
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(_MODEL, api_key="key-one")
    _turn(model, script, "First run.", "one")
    first_client, first_pool = model.client, model._http_client

    model.api_key = "key-two"
    _turn(model, script, "Second run.", "two")

    assert model.client is not first_client
    assert first_pool.is_closed, "the replaced client's pool kept its sockets open"
    assert model._http_client is not first_pool
    assert not model._http_client.is_closed


def test_an_unchanged_key_keeps_the_open_pool(
    monkeypatch: pytest.MonkeyPatch, gemini_endpoint: tuple[str, GeminiScript],
) -> None:
    """A run with the same key reuses the client and its pool, open."""
    base_url, script = gemini_endpoint
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", base_url)
    model = GeminiModel(_MODEL, api_key="key-one")
    _turn(model, script, "First run.", "one")
    client, pool = model.client, model._http_client

    _turn(model, script, "Second run.", "two")

    assert model.client is client
    assert model._http_client is pool
    assert not pool.is_closed
