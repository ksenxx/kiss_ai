# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: gmail/googlechat credential paths honour ``KISS_HOME``.

Previously ``gmail_agent`` and ``googlechat_agent`` built their credential
directories from a module-level ``Path.home() / ".kiss" / ...`` constant,
bypassing the per-process ``KISS_HOME`` isolation that
``src/kiss/tests/conftest.py`` sets up.  Parallel pytest processes therefore
raced on the REAL user files (``token.json`` / ``credentials.json``) — the
same class of bug previously fixed for slack tokens and tlon configs.

These tests prove the paths now resolve lazily via ``_kiss_home()`` so each
pytest process (and each ``KISS_HOME`` change) gets fully isolated state, and
that the ``channel_work`` working-directory default follows suit.
"""

from __future__ import annotations

import os
from pathlib import Path

from kiss.agents.third_party_agents import gmail_agent, googlechat_agent
from kiss.agents.third_party_agents._channel_agent_utils import ChannelRunner


class _KissHomeSwap:
    """Temporarily point ``KISS_HOME`` at a given directory."""

    def __init__(self, target: Path) -> None:
        self._saved = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(target)

    def restore(self) -> None:
        """Restore the original ``KISS_HOME`` value."""
        if self._saved is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved


def test_gmail_paths_honour_kiss_home_lazily(tmp_path: Path) -> None:
    """Gmail token/credentials paths follow $KISS_HOME changes after import."""
    swap = _KissHomeSwap(tmp_path / "home_a")
    try:
        base_a = tmp_path / "home_a" / "third_party_agents" / "gmail"
        assert gmail_agent._token_path() == base_a / "token.json"
        assert gmail_agent._credentials_path() == base_a / "credentials.json"

        os.environ["KISS_HOME"] = str(tmp_path / "home_b")
        base_b = tmp_path / "home_b" / "third_party_agents" / "gmail"
        assert gmail_agent._token_path() == base_b / "token.json"
        assert gmail_agent._credentials_path() == base_b / "credentials.json"
        assert not gmail_agent._token_path().exists()
    finally:
        swap.restore()


def test_gmail_token_save_isolated_per_kiss_home(tmp_path: Path) -> None:
    """A token written under one KISS_HOME never leaks into another."""
    swap = _KissHomeSwap(tmp_path / "home_a")
    try:
        gmail_agent._gmail_dir().mkdir(parents=True, exist_ok=True)
        gmail_agent._token_path().write_text('{"token": "test-isolated"}')
        assert gmail_agent._token_path().exists()

        os.environ["KISS_HOME"] = str(tmp_path / "home_b")
        assert not gmail_agent._token_path().exists(), (
            "token must not leak across KISS_HOME dirs"
        )
    finally:
        swap.restore()


def test_googlechat_paths_honour_kiss_home_lazily(tmp_path: Path) -> None:
    """Google Chat credential paths follow $KISS_HOME changes after import."""
    swap = _KissHomeSwap(tmp_path / "home_a")
    try:
        base_a = tmp_path / "home_a" / "third_party_agents" / "googlechat"
        assert googlechat_agent._token_path() == base_a / "token.json"
        assert googlechat_agent._credentials_path() == base_a / "credentials.json"
        assert googlechat_agent._service_account_path() == (
            base_a / "service_account.json"
        )

        os.environ["KISS_HOME"] = str(tmp_path / "home_b")
        base_b = tmp_path / "home_b" / "third_party_agents" / "googlechat"
        assert googlechat_agent._token_path() == base_b / "token.json"
        assert googlechat_agent._credentials_path() == base_b / "credentials.json"
        assert googlechat_agent._service_account_path() == (
            base_b / "service_account.json"
        )
    finally:
        swap.restore()


def test_channel_runner_default_work_dir_honours_kiss_home(tmp_path: Path) -> None:
    """ChannelRunner's default work_dir resolves under $KISS_HOME."""
    swap = _KissHomeSwap(tmp_path / "home_a")
    try:

        class _NullBackend:
            pass

        runner = ChannelRunner(
            backend=_NullBackend(), channel_name="general", agent_name="Test Agent"
        )
        assert runner._work_dir == str(tmp_path / "home_a" / "channel_work")
    finally:
        swap.restore()
