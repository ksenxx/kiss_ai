# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: gmail/googlechat credential paths honour ``KISS_HOME``.

Previously ``gmail_sea`` and ``googlechat_sea`` built their credential
directories from a module-level ``Path.home() / ".kiss" / ...`` constant,
bypassing the per-process ``KISS_HOME`` isolation that
``src/kiss/tests/conftest.py`` sets up.  Parallel pytest processes therefore
raced on the REAL user files (then ``token.json`` / ``credentials.json``;
now the Composio connection record and the Chat service account key) — the
same class of bug previously fixed for slack tokens and tlon configs.

These tests prove the paths now resolve lazily via ``kiss.core.config.kiss_home()`` so each
pytest process (and each ``KISS_HOME`` change) gets fully isolated state, and
that the ``channel_work`` working-directory default follows suit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.third_party_agents._channel_agent_utils import ChannelRunner, write_private_file
from kiss.agents.third_party_agents._composio_google import connected_account_id, service_dir
from kiss.agents.third_party_agents.googlechat import googlechat_sea
from kiss.agents.third_party_agents.irc.irc_sea import IRCChannelBackend


def test_gmail_connection_isolated_per_kiss_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Gmail connection recorded under one KISS_HOME never leaks into another."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home_a"))
    base_a = tmp_path / "home_a" / "third_party_agents" / "gmail"
    assert service_dir("gmail") == base_a
    write_private_file(base_a / "composio.json", '{"connected_account_id": "ca_a"}')
    assert connected_account_id("gmail") == "ca_a"

    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home_b"))
    assert service_dir("gmail") == tmp_path / "home_b" / "third_party_agents" / "gmail"
    assert connected_account_id("gmail") == "", "connection must not leak across KISS_HOME"


def test_googlechat_paths_honour_kiss_home_lazily(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Google Chat credential paths follow $KISS_HOME changes after import."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home_a"))
    base_a = tmp_path / "home_a" / "third_party_agents" / "googlechat"
    assert googlechat_sea._service_account_path() == base_a / "service_account.json"

    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home_b"))
    base_b = tmp_path / "home_b" / "third_party_agents" / "googlechat"
    assert googlechat_sea._service_account_path() == base_b / "service_account.json"


def test_channel_runner_default_work_dir_honours_kiss_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ChannelRunner's default work_dir resolves under $KISS_HOME.

    The runner is built over a real (unconnected) ``IRCChannelBackend``;
    the work-dir default does not depend on the backend.
    """
    monkeypatch.setenv("KISS_HOME", str(tmp_path / "home_a"))
    runner = ChannelRunner(
        backend=IRCChannelBackend(), channel_name="general", agent_name="Test Agent"
    )
    assert runner._work_dir == str(tmp_path / "home_a" / "channel_work")
