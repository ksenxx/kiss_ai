# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for all channel agents — no mocks or test doubles.

Tests the unauthenticated answers of every channel agent's ``check_*_auth``
and ``clear_*_auth`` tools, plus config persistence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import ChannelConfig
from kiss.tests.agents.third_party_agents.channel_agent_registry import (
    AGENT_IDS,
    ALL_CHANNEL_AGENTS,
    EXPECTED_AUTH_TOOLS,
    get_agent_class,
)
from kiss.tests.agents.third_party_agents.muse_test_utils import tools_by_name

# Every channel's ChannelConfig resolves KISS_HOME lazily, so each test
# starts from an empty per-test home and leaves nothing behind.
pytestmark = pytest.mark.usefixtures("isolated_kiss_home")


def _auth_tool(class_name: str, module_path: str, prefix: str):
    """Return a fresh agent's auth tool whose name starts with *prefix* (``check_``/``clear_``)."""
    tools = tools_by_name(get_agent_class(class_name, module_path)())
    name = next(n for n in EXPECTED_AUTH_TOOLS[class_name] if n.startswith(prefix))
    return tools[name]


@pytest.mark.parametrize("class_name,module_path", ALL_CHANNEL_AGENTS, ids=AGENT_IDS)
def test_check_auth_unauthenticated(class_name: str, module_path: str) -> None:
    """check_*_auth() returns a helpful message when not configured."""
    if class_name == "IMessageAgent" and sys.platform != "darwin":
        pytest.skip("macOS-only agent")
    result = _auth_tool(class_name, module_path, "check_")()
    lower = result.lower()
    assert (
        "not authenticated" in lower
        or "not configured" in lower
        or "no " in lower
        or "authenticate" in lower
    ), f"Expected unauthenticated message, got: {result[:200]}"


@pytest.mark.parametrize("class_name,module_path", ALL_CHANNEL_AGENTS, ids=AGENT_IDS)
def test_clear_auth_when_not_authenticated(class_name: str, module_path: str) -> None:
    """clear_*_auth() works without error even when not authenticated."""
    result = _auth_tool(class_name, module_path, "clear_")()
    assert "cleared" in result.lower() or "removed" in result.lower(), (
        f"Expected cleared message, got: {result[:200]}"
    )


def test_config_roundtrip(tmp_path: Path) -> None:
    """``ChannelConfig`` load/clear tolerate a missing, corrupt or non-object file.

    Every channel module's ``_config`` is a ``ChannelConfig``, so one instance
    pins the behaviour for all of them.
    """
    config = ChannelConfig(tmp_path / "channel", ("token",))
    path = config.path
    config.clear()
    assert config.load() is None
    assert not path.exists()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{bad json!!")
    assert config.load() is None

    path.write_text('"just a string"')
    assert config.load() is None

    config.clear()
    assert not path.exists()
