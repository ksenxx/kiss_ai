# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared fixtures for channel-agent tests.

Isolates the Slack token directory per test so parallel pytest processes
never race on the real ``~/.kiss/third_party_agents/slack`` path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import kiss.agents.third_party_agents.slack_agent as slack_agent_mod


@pytest.fixture(autouse=True)
def _isolated_slack_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect Slack token storage to a per-test temporary directory.

    ``slack_agent._SLACK_DIR`` is a module global built from ``Path.home()``,
    so tests that save or clear tokens would otherwise touch the real user
    token file and race with concurrent pytest processes. Some test modules
    also import ``_SLACK_DIR`` by value, so their own module binding is
    patched too when present.
    """
    isolated = tmp_path / "slack"
    monkeypatch.setattr(slack_agent_mod, "_SLACK_DIR", isolated)
    for mod_name in (
        "kiss.tests.agents.channels.test_slack_agent",
        "kiss.tests.agents.channels.test_slack_channel_backend",
        "kiss.tests.agents.channels.test_run_once",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_SLACK_DIR"):
            monkeypatch.setattr(mod, "_SLACK_DIR", isolated)
    return isolated
