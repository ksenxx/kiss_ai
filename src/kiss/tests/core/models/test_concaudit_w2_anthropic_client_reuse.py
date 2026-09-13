# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``AnthropicModel.initialize`` must reuse its SDK client across runs.

``KISSAgent._reset`` keeps one adapter instance alive across sub-sessions
and calls ``initialize()`` at the start of every run.  Building a fresh
``Anthropic`` client each time also builds a fresh httpx connection pool
and abandons the previous one to the garbage collector — the same
repeated work ``OpenAICompatibleBase._ensure_client`` removed for the
OpenAI transports.  The client is rebuilt only when an input it was
built from (the API key, the workspace-id header) actually changed.
"""

from __future__ import annotations

import pytest

from kiss.core.models.anthropic_model import AnthropicModel


def _make() -> AnthropicModel:
    """Build an adapter that never talks to the network in these tests."""
    return AnthropicModel("claude-under-test", api_key="key-1")


def test_initialize_twice_keeps_the_same_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs with unchanged inputs share one client (and its pool)."""
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    model = _make()
    model.initialize("first run")
    first = model.client
    assert first is not None

    model.reset_conversation()
    model.initialize("second run")

    assert model.client is first
    assert model.conversation == [{"role": "user", "content": "second run"}]


def test_changed_workspace_id_rebuilds_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workspace header is read per run; a change must reach the client."""
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    model = _make()
    model.initialize("run without workspace")
    first = model.client
    assert "anthropic-workspace-id" not in first.default_headers

    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_changed")
    model.initialize("run with workspace")

    assert model.client is not first
    assert model.client.default_headers["anthropic-workspace-id"] == "wrkspc_changed"


def test_changed_api_key_rebuilds_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rotated key must not keep authenticating with the old one."""
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    model = _make()
    model.initialize("run one")
    first = model.client

    model.api_key = "key-2"
    model.initialize("run two")

    assert model.client is not first
    assert model.client.api_key == "key-2"
