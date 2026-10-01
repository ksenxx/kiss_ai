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
from collections.abc import Iterator
from pathlib import Path

import pytest

import kiss.agents.third_party_agents.slack.slack_sea as slack_agent_mod


@pytest.fixture(autouse=True)
def no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the auth hand-off from opening the developer's real browser.

    Unauthenticated ``check_*_auth()`` / ``authenticate_*()`` calls open
    the provider's sign-in page in the default browser when the process
    has a display; under pytest that page would pop up on a desktop dev
    machine.  Tests that exercise the launcher install a scripted
    ``$BROWSER`` and set ``KISS_HEADLESS=0`` themselves.
    """
    monkeypatch.setenv("KISS_HEADLESS", "1")


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

    ``slack_sea._slack_dir()`` resolves ``$KISS_HOME`` lazily, which already
    keeps parallel pytest processes apart; patching it here additionally
    gives every *test* a fresh, not-yet-created directory, so a token saved
    by one test is never visible to the next.  The returned path is the
    directory the product code will use for the duration of the test.
    """
    isolated = tmp_path / "slack"
    monkeypatch.setattr(slack_agent_mod, "_slack_dir", lambda: isolated)
    return isolated
