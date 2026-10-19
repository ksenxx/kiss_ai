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

import os
import shutil
import socket
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import kiss.agents.third_party_agents.slack.slack_sea as slack_agent_mod
from kiss.tests.agents.third_party_agents.composio_test_utils import (
    FakeComposioServer,
    start_fake_composio,
)
from kiss.tests.agents.third_party_agents.muse_test_utils import setup_muse_env, teardown_muse_env
from kiss.tests.conftest import install_cli_script

# Scripted signal-cli for the ``link`` flow (``test_muse_connect_flows``,
# ``test_browser_handoff``).  A Python program, not a shell script, so the
# same stand-in runs on Windows, where ``install_cli_script`` adds the
# ``.cmd`` shim.
FAKE_SIGNAL_LINK_CLI = """#!/usr/bin/env python3
# Scripted stand-in for signal-cli: ``link`` prints the provisioning URI,
# waits for the phone (a marker file), then announces the linked account.
import os
import sys
import time

state_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
args = sys.argv[1:]


def marker(name):
    return os.path.exists(os.path.join(state_dir, name))


if args[:1] == ["link"]:
    if args[1:2] != ["-n"]:
        print("expected -n", file=sys.stderr)
        sys.exit(2)
    with open(os.path.join(state_dir, "device-name"), "w") as fh:
        fh.write(args[2] + "\\n")
    if marker("no-uri"):
        print("Failed to link", file=sys.stderr)
        sys.exit(3)
    print("sgnl://linkdevice?uuid=abc-123&pub_key=BQ%2Fkey", flush=True)
    if marker("noisy"):
        sys.stderr.write("x" * 2097152)
        sys.stderr.flush()
    while not marker("scanned"):
        if marker("fail"):
            print("Link request timed out", file=sys.stderr)
            sys.exit(1)
        time.sleep(0.1)
    if not marker("silent"):
        print("Associated with: +15550001111")
    sys.exit(0)
if args[:1] == ["listAccounts"]:
    if marker("silent"):
        print("Number: +15550009999")
    sys.exit(0)
print("unknown command " + " ".join(args[:1]), file=sys.stderr)
sys.exit(4)
"""


def install_cli_on_path(test: unittest.TestCase, name: str, source: str) -> Path:
    """Put the stand-in CLI *name* (the program *source*) first on PATH for the rest of *test*.

    Returns the temp directory holding it.  Each mutation registers its
    undo with ``addCleanup`` at once, so PATH is restored and the directory
    removed even when a later line of the caller's ``setUp`` fails.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix=f"kiss-fake-{name}-"))
    test.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
    install_cli_script(tmpdir / name, source)
    test.addCleanup(os.environ.__setitem__, "PATH", os.environ["PATH"])
    os.environ["PATH"] = str(tmpdir) + os.pathsep + os.environ["PATH"]
    return tmpdir


def connect_on_ephemeral_port(config: Any, backend: Any, server_attr: str) -> int:
    """Connect *backend* through its public ``connect()`` on an OS-assigned port.

    The channel's ``authenticate_*`` tool deliberately rejects port ``"0"``,
    so the persisted *config*'s port is rewritten to ``"0"`` directly; the
    public ``connect()`` then binds an ephemeral port in the server it keeps
    (no reserve-close-rebind race) and the really bound port is read back
    from the live server stored on ``backend.<server_attr>``.

    Args:
        config: The channel's ``ChannelConfig`` holding the persisted settings.
        backend: An unconnected channel backend.
        server_attr: Name of the backend attribute holding its HTTP server.

    Returns:
        The port the backend's server is listening on.
    """
    cfg = config.load()
    assert cfg is not None
    cfg["port"] = "0"
    config.save(cfg)
    assert backend.connect() is True
    server = getattr(backend, server_attr)
    assert server is not None
    return int(server.server_address[1])


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
def muse_env(
    request: pytest.FixtureRequest, isolated_kiss_home: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Path]:
    """Enable Muse-auth inside an isolated ``KISS_HOME`` with a live daemon.

    The requesting module's ``MUSE_POLICY`` (usually built with
    ``muse_test_utils.loopback_policy``) is installed as the daemon's
    policy; the daemon is stopped and awaited after the test.
    """
    setup_muse_env(monkeypatch, request.module.MUSE_POLICY)
    yield isolated_kiss_home
    teardown_muse_env()


@pytest.fixture
def composio(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeComposioServer]:
    """Run the local Composio API emulator and point the Composio SDK at it."""
    yield from start_fake_composio(monkeypatch)


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
def _isolated_slack_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
