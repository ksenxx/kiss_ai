# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared Muse-auth helpers for the ``test_muse_*`` modules.

Each Muse test module keeps its own tiny ``muse_env`` fixture (the
per-module policy documents which services get loopback ``extra_hosts``
entries) but delegates the common setup and teardown here:
``setup_muse_env`` enables Muse-auth and writes the policy file inside
the isolated ``KISS_HOME``, and ``teardown_muse_env`` stops the daemon
subprocess and waits for its socket to vanish so the next test (or the
next daemon in the same test) starts from a clean slate.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir, socket_path
from kiss.agents.third_party_agents.muse_auth.client import stop_daemon


def setup_muse_env(monkeypatch: pytest.MonkeyPatch, policy: dict[str, Any]) -> None:
    """Enable Muse-auth and install ``policy`` inside the current ``KISS_HOME``.

    Sets ``KISS_MUSE_AUTH=1`` via ``monkeypatch`` (so it is undone on
    teardown) and writes ``policy`` as ``policy.json`` in the Muse-auth
    directory, creating the directory first.  Callers must already have
    pointed ``KISS_HOME`` at an isolated location (the
    ``isolated_kiss_home`` fixture).

    Args:
        monkeypatch: The test's monkeypatch, used for the env var.
        policy: The Muse-auth policy document to write.
    """
    monkeypatch.setenv("KISS_MUSE_AUTH", "1")
    directory = muse_auth_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "policy.json").write_text(json.dumps(policy))


def teardown_muse_env() -> None:
    """Stop the Muse-auth daemon and wait until its socket disappears."""
    stop_daemon()
    wait_daemon_stopped()


def wait_daemon_stopped(timeout: float = 10.0) -> None:
    """Poll until the Muse-auth daemon socket vanishes (or ``timeout``).

    ``stop_daemon`` only asks the daemon to exit; the subprocess removes
    its socket as it shuts down.  Waiting for the socket to vanish keeps
    a follow-up ``ensure_daemon`` (or the next test) from handshaking
    with the dying daemon.

    Args:
        timeout: Maximum seconds to wait for the socket to disappear.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and socket_path().exists():
        time.sleep(0.05)


def auth_tools(agent: Any) -> dict[str, Any]:
    """Return ``agent``'s auth tools keyed by function name.

    Args:
        agent: Any channel agent exposing ``_get_auth_tools``.

    Returns:
        A mapping from tool function name to the tool callable.
    """
    return {tool.__name__: tool for tool in agent._get_auth_tools()}
