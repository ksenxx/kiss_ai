# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests for TlonAgent auth-check flakiness (leftover config state).

The flake: ``test_check_auth_unauthenticated[TlonAgent]`` intermittently saw a
leftover ``~/.kiss/third_party_agents/tlon/config.json`` (written by a
concurrently running test that points it at a live local ephemeral-port HTTP
server), so ``check_tlon_auth()`` returned ``{"ok": true, ...}`` instead of the
unauthenticated message.

Fix verified here: ``ChannelConfig.path`` resolves lazily and honours
``$KISS_HOME``, so a test run under ``isolated_kiss_home`` (a fresh temp
``KISS_HOME`` per test) has fully isolated channel config state and a
leftover config from another test cannot be seen at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.third_party_agents.tlon.tlon_sea import TlonAgent, _config
from kiss.tests.agents.third_party_agents.muse_test_utils import tools_by_name

pytestmark = pytest.mark.usefixtures("isolated_kiss_home")


def test_channel_config_path_honours_kiss_home_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ChannelConfig.path follows $KISS_HOME changes made after import."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss_home_a"))
    path_a = _config.path
    assert path_a == (tmp_path / "kiss_home_a" / "third_party_agents" / "tlon" / "config.json")
    _config.save({"ship_url": "http://127.0.0.1:1", "code": "c", "ship": "~zod"})
    assert path_a.exists()
    loaded = _config.load()
    assert loaded is not None and loaded["ship_url"] == "http://127.0.0.1:1"

    monkeypatch.setenv("KISS_HOME", str(tmp_path / "kiss_home_b"))
    path_b = tmp_path / "kiss_home_b" / "third_party_agents" / "tlon" / "config.json"
    assert _config.path == path_b
    assert _config.load() is None, "config must not leak across KISS_HOME dirs"


def test_clear_tlon_auth_resets_backend_and_config() -> None:
    """clear_tlon_auth removes the config file and in-memory ship state."""
    _config.save({"ship_url": "http://127.0.0.1:1", "code": "c", "ship": "~zod"})
    agent = TlonAgent()
    assert agent._is_authenticated()
    tools = tools_by_name(agent)
    result = tools["clear_tlon_auth"]()
    assert "cleared" in result.lower()
    assert _config.load() is None
    assert not agent._is_authenticated()
    assert "not configured" in tools["check_tlon_auth"]().lower()
