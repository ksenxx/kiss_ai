# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared fixtures for channel-agent tests.

Isolates the Slack token directory per test so parallel pytest processes
never race on the real ``~/.kiss/third_party_agents/slack`` path, and
provides the per-test ``KISS_HOME`` and refusing-port fixtures that several
test modules used to define locally.
"""

from __future__ import annotations

import socket
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.agents.third_party_agents.slack_sea as slack_agent_mod


@pytest.fixture
def isolated_kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ``KISS_HOME`` at a per-test temp dir so ``~/.kiss`` is never touched.

    ``ChannelConfig.path`` and ``kiss_home()`` resolve the env var lazily,
    so every config written by the test lands under the returned directory.
    """
    home = tmp_path / "kiss_home"
    monkeypatch.setenv("KISS_HOME", str(home))
    return home


@pytest.fixture
def refusing_port() -> Iterator[int]:
    """A localhost TCP port on which every connect is refused for the whole test.

    The port is the local end of an *established* loopback connection
    (``holder`` connected to ``anchor``) that lives for the whole test.  A
    SYN from any other peer matches neither that connection nor a listener,
    so the kernel answers RST at once (``ECONNREFUSED``); and because the
    port is in use, no other socket can bind it or be handed it by
    ``bind(0)`` meanwhile -- unlike the bind/close/reuse-the-number pattern,
    which races with concurrent tests.

    A socket that is merely bound and never listening -- the earlier
    version of this fixture -- is refused only on Linux: macOS silently
    drops SYNs aimed at a CLOSED-state socket, so connects timed out
    instead and every "unreachable server" test failed there.
    """
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as anchor,
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder,
    ):
        anchor.bind(("127.0.0.1", 0))
        anchor.listen(1)
        holder.bind(("127.0.0.1", 0))
        holder.connect(anchor.getsockname())
        yield int(holder.getsockname()[1])


@pytest.fixture(autouse=True)
def _isolated_slack_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Redirect Slack token storage to a per-test temporary directory.

    ``slack_sea._SLACK_DIR`` is a module global built from ``Path.home()``,
    so tests that save or clear tokens would otherwise touch the real user
    token file and race with concurrent pytest processes. Some test modules
    also import ``_SLACK_DIR`` by value, so their own module binding is
    patched too when present.
    """
    isolated = tmp_path / "slack"
    monkeypatch.setattr(slack_agent_mod, "_SLACK_DIR", isolated)
    for mod_name in (
        "kiss.tests.agents.third_party_agents.test_slack_agent",
        "kiss.tests.agents.third_party_agents.test_slack_channel_backend",
        "kiss.tests.agents.third_party_agents.test_run_once",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_SLACK_DIR"):
            monkeypatch.setattr(mod, "_SLACK_DIR", isolated)
    return isolated
