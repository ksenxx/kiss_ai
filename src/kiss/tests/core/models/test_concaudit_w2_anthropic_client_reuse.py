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

import threading
from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.anthropic_model import AnthropicModel
from kiss.tests.server.pause import Pause


def _make() -> AnthropicModel:
    """Build an adapter that never talks to the network in these tests."""
    return AnthropicModel("claude-under-test", api_key="key-1")


def _set_workspace_id(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Configure the workspace id (the adapter reads the config field) for one test."""
    monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_WORKSPACE_ID", value, raising=False)


def test_initialize_twice_keeps_the_same_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs with unchanged inputs share one client (and its pool)."""
    _set_workspace_id(monkeypatch, "")
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
    _set_workspace_id(monkeypatch, "")
    model = _make()
    model.initialize("run without workspace")
    first = model.client
    assert "anthropic-workspace-id" not in first.default_headers

    _set_workspace_id(monkeypatch, "wrkspc_changed")
    model.initialize("run with workspace")

    assert model.client is not first
    assert model.client.default_headers["anthropic-workspace-id"] == "wrkspc_changed"


def test_changed_api_key_rebuilds_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rotated key must not keep authenticating with the old one."""
    _set_workspace_id(monkeypatch, "")
    model = _make()
    model.initialize("run one")
    first = model.client

    model.api_key = "key-2"
    model.initialize("run two")

    assert model.client is not first
    assert model.client.api_key == "key-2"


def _any_call(_frame_locals: dict[str, Any]) -> bool:
    """Hold whichever thread enters the paused function."""
    return True


def _built_for(client: Any) -> str:
    """The setting *client* was built with: its Bedrock region, or its workspace header."""
    region = getattr(client, "aws_region", None)
    return str(region or client.default_headers["anthropic-workspace-id"])


@pytest.mark.parametrize(
    ("name", "field"),
    [
        ("claude-under-test", "ANTHROPIC_WORKSPACE_ID"),
        ("bedrock/us.anthropic.claude-opus-4-6-v1:0", "AWS_REGION"),
    ],
    ids=["workspace", "bedrock"],
)
def test_a_setting_saved_while_the_client_is_built_keeps_its_own_client(
    monkeypatch: pytest.MonkeyPatch, name: str, field: str
) -> None:
    """The client is built from the very settings its reuse key records.

    The builder read the settings again: a save between the two readings
    left the new setting's client under the old setting's key, and a run
    after the old setting was restored kept sending the new one.
    """
    model = AnthropicModel(name, api_key="key-1", bedrock=name.startswith("bedrock/"))
    monkeypatch.setattr(DEFAULT_CONFIG, field, "setting-a")
    with Pause(AnthropicModel._build_client.__code__, _any_call) as held:
        run = threading.Thread(target=model.initialize, args=("first run",))
        run.start()
        assert held.paused.wait(10), "the client was never built"
        monkeypatch.setattr(DEFAULT_CONFIG, field, "setting-b")
    run.join(10)
    for setting in ("setting-a", "setting-b"):
        monkeypatch.setattr(DEFAULT_CONFIG, field, setting)
        model.initialize("next run")
        assert _built_for(model.client) == setting
