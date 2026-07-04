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

Fixes verified here:
1. ``ChannelConfig.path`` resolves lazily and honours ``$KISS_HOME``, so each
   pytest process (conftest sets a fresh temp ``KISS_HOME``) has fully isolated
   channel config state.
2. ``test_new_channel_agents`` now resets module-level ``ChannelConfig``
   objects (``_config``) before asserting unauthenticated state, so leftover
   config within the same process is also cleared.

No mocks or test doubles: a real in-process HTTP server plays the Urbit ship.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from kiss.agents.third_party_agents.tlon_agent import TlonAgent, _config
from kiss.tests.agents.channels.test_new_channel_agents import (
    _CHANNEL_AGENTS,
)
from kiss.tests.agents.channels.test_new_channel_agents import (
    test_check_auth_unauthenticated as _run_check_auth_unauthenticated,
)

_TLON_INFO = next(ch for ch in _CHANNEL_AGENTS if ch["agent_class"] == "TlonAgent")


class _ShipHandler(BaseHTTPRequestHandler):
    """Minimal Urbit-ship-shaped handler: accepts /~/login."""

    def do_POST(self) -> None:  # noqa: N802
        """Accept any login POST with 204, like a real ship."""
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self.send_response(204)
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silence request logging."""


class _ConfigBackup:
    """Back up and restore the tlon config file around a test."""

    def __init__(self) -> None:
        self._backup: str | None = None
        if _config.path.exists():
            self._backup = _config.path.read_text()

    def restore(self) -> None:
        """Restore the original config file state."""
        if self._backup is not None:
            _config.path.parent.mkdir(parents=True, exist_ok=True)
            _config.path.write_text(self._backup)
        elif _config.path.exists():
            _config.path.unlink()


def test_channel_config_path_honours_kiss_home_lazily(tmp_path: Path) -> None:
    """ChannelConfig.path follows $KISS_HOME changes made after import."""
    saved = os.environ.get("KISS_HOME")
    try:
        os.environ["KISS_HOME"] = str(tmp_path / "kiss_home_a")
        path_a = _config.path
        assert path_a == (
            tmp_path / "kiss_home_a" / "third_party_agents" / "tlon" / "config.json"
        )
        _config.save({"ship_url": "http://127.0.0.1:1", "code": "c", "ship": "~zod"})
        assert path_a.exists()
        loaded = _config.load()
        assert loaded is not None and loaded["ship_url"] == "http://127.0.0.1:1"

        os.environ["KISS_HOME"] = str(tmp_path / "kiss_home_b")
        assert _config.path == (
            tmp_path / "kiss_home_b" / "third_party_agents" / "tlon" / "config.json"
        )
        assert _config.load() is None, "config must not leak across KISS_HOME dirs"
    finally:
        if saved is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = saved


def test_check_auth_unauthenticated_survives_leftover_ephemeral_config() -> None:
    """The exact flaky scenario: leftover config pointing at a live local port.

    A real local HTTP server (ephemeral port) plays the ship, and a leftover
    config referencing it is saved before the auth-check test runs.  The
    test must clear that state and still report unauthenticated.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ShipHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backup = _ConfigBackup()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        _config.save({"ship_url": base, "code": "lidlut-tabwed", "ship": "~zod"})

        # Leftover config makes a fresh agent look authenticated...
        agent = TlonAgent()
        agent.web_use_tool = None
        tools = {t.__name__: t for t in agent._get_tools()}
        assert json.loads(tools["check_tlon_auth"]())["ok"] is True

        # ...but the shared auth-check test now resets it and passes.
        _run_check_auth_unauthenticated(_TLON_INFO)
    finally:
        backup.restore()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_clear_tlon_auth_resets_backend_and_config() -> None:
    """clear_tlon_auth removes the config file and in-memory ship state."""
    backup = _ConfigBackup()
    try:
        _config.save({"ship_url": "http://127.0.0.1:1", "code": "c", "ship": "~zod"})
        agent = TlonAgent()
        agent.web_use_tool = None
        assert agent._is_authenticated()
        tools = {t.__name__: t for t in agent._get_tools()}
        result = tools["clear_tlon_auth"]()
        assert "cleared" in result.lower()
        assert _config.load() is None
        assert not agent._is_authenticated()
        assert "not configured" in tools["check_tlon_auth"]().lower()
    finally:
        backup.restore()
