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

import os
from pathlib import Path

from kiss.agents.third_party_agents._channel_agent_utils import ChannelRunner, write_private_file
from kiss.agents.third_party_agents._composio_google import connected_account_id, service_dir
from kiss.agents.third_party_agents.googlechat import googlechat_sea


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


def test_gmail_connection_isolated_per_kiss_home(tmp_path: Path) -> None:
    """A Gmail connection recorded under one KISS_HOME never leaks into another."""
    swap = _KissHomeSwap(tmp_path / "home_a")
    try:
        base_a = tmp_path / "home_a" / "third_party_agents" / "gmail"
        assert service_dir("gmail") == base_a
        write_private_file(base_a / "composio.json", '{"connected_account_id": "ca_a"}')
        assert connected_account_id("gmail") == "ca_a"

        os.environ["KISS_HOME"] = str(tmp_path / "home_b")
        assert service_dir("gmail") == tmp_path / "home_b" / "third_party_agents" / "gmail"
        assert connected_account_id("gmail") == "", "connection must not leak across KISS_HOME"
    finally:
        swap.restore()


def test_googlechat_paths_honour_kiss_home_lazily(tmp_path: Path) -> None:
    """Google Chat credential paths follow $KISS_HOME changes after import."""
    swap = _KissHomeSwap(tmp_path / "home_a")
    try:
        base_a = tmp_path / "home_a" / "third_party_agents" / "googlechat"
        assert googlechat_sea._service_account_path() == (
            base_a / "service_account.json"
        )

        os.environ["KISS_HOME"] = str(tmp_path / "home_b")
        base_b = tmp_path / "home_b" / "third_party_agents" / "googlechat"
        assert googlechat_sea._service_account_path() == (
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
