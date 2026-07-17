# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bughunt-2 integration tests for the sorcar CLI client modules.

Covers two genuine bugs found by auditing
:mod:`kiss.ui.cli.cli_client` and
:mod:`kiss.ui.cli.cli_daemon_bridge`:

1. ``cli_client._request_models`` (used by ``/model list``) performed a
   single blind ``models_q.get(timeout=10)`` that ignored
   ``client._closed`` — the daemon-disconnect early-bail that every
   sibling waiter already honours (``_request_cli_info`` per review
   #8/#25 and the ``/autocommit`` wait loop per review A3).  After a
   daemon disconnect, ``/model list`` therefore froze the REPL for the
   full 10-second timeout even though no reply could ever arrive.

2. ``cli_daemon_bridge._send_envelope`` caches its AF_UNIX writer
   forever, so a ``KISS_SORCAR_SOCK`` change is silently ignored even
   though ``_sock_path()`` explicitly re-reads the environment variable
   on every call (its documented contract, "override via the
   ``KISS_SORCAR_SOCK`` env var for tests").  Events emitted after the
   path changed kept flowing to the OLD daemon socket instead of the
   daemon now listening at the new path.

No mocks / patches / fakes: the tests drive the real ``CliClient``
object and real AF_UNIX listener sockets.
"""

from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
from pathlib import Path

from kiss.core.print_to_console import ConsolePrinter
from kiss.ui.cli import cli_daemon_bridge
from kiss.ui.cli.cli_client import CliClient, _request_models


def _make_closed_client(tmp_path: Path) -> CliClient:
    """Build a real (never-started) ``CliClient`` in the closed state.

    A client whose loop thread never ran behaves exactly like one
    whose daemon connection dropped: ``send`` is a no-op (no running
    loop) and ``_closed`` is set — the state every disconnect-aware
    waiter must bail out of promptly.
    """
    client = CliClient(
        tmp_path / "no-daemon.sock", str(tmp_path), "tab-bh2", ConsolePrinter(),
    )
    client._closed.set()
    return client


class TestRequestModelsHonoursDisconnect:
    """Bug 1: ``_request_models`` must bail early on daemon disconnect."""

    def test_returns_promptly_when_already_closed(self, tmp_path: Path) -> None:
        """A pre-closed client must not block for the full 10 s timeout."""
        client = _make_closed_client(tmp_path)
        start = time.monotonic()
        result = _request_models(client)
        elapsed = time.monotonic() - start
        assert result == []
        assert elapsed < 5.0, (
            f"_request_models blocked {elapsed:.1f}s after daemon "
            f"disconnect — it must honour client._closed like "
            f"_request_cli_info does"
        )

    def test_returns_promptly_when_closed_mid_wait(self, tmp_path: Path) -> None:
        """A disconnect that lands while waiting must unblock the waiter."""
        client = CliClient(
            tmp_path / "no-daemon.sock", str(tmp_path), "tab-bh2b",
            ConsolePrinter(),
        )
        timer = threading.Timer(0.5, client._closed.set)
        timer.start()
        try:
            start = time.monotonic()
            result = _request_models(client)
            elapsed = time.monotonic() - start
        finally:
            timer.cancel()
        assert result == []
        assert elapsed < 5.0, (
            f"_request_models blocked {elapsed:.1f}s after a mid-wait "
            f"daemon disconnect"
        )


class _UdsLineServer:
    """Minimal real AF_UNIX line-collecting server for bridge tests."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.received: list[bytes] = []
        self._lock = threading.Lock()
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(path))
        self._srv.listen(4)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(
                target=self._pump, args=(conn,), daemon=True,
            ).start()

    def _pump(self, conn: socket.socket) -> None:
        buf = b""
        with conn:
            while True:
                try:
                    data = conn.recv(4096)
                except OSError:
                    return
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    with self._lock:
                        self.received.append(line)

    def lines(self) -> list[bytes]:
        with self._lock:
            return list(self.received)

    def wait_for_line(self, timeout: float = 3.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.lines():
                return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        try:
            self._srv.close()
        except OSError:
            pass


class TestBridgeFollowsSockPathChanges:
    """Bug 2: the bridge must reconnect when ``KISS_SORCAR_SOCK`` changes."""

    def test_send_event_follows_env_var_change(self) -> None:
        """Events after a sock-path change must reach the NEW daemon."""
        # Short base dir: AF_UNIX sun_path is capped at ~104 bytes on
        # macOS, and pytest tmp_path can exceed it.
        base = Path(tempfile.mkdtemp(prefix="bh2-", dir="/tmp"))
        server_a = _UdsLineServer(base / "a.sock")
        server_b = _UdsLineServer(base / "b.sock")
        old_env = os.environ.get("KISS_SORCAR_SOCK")
        # Reset the module-level cached writer so a connection cached
        # by an earlier test cannot leak into this one.
        with cli_daemon_bridge._LOCK:
            if cli_daemon_bridge._WRITER is not None:
                try:
                    cli_daemon_bridge._WRITER.close()
                except OSError:
                    pass
                cli_daemon_bridge._WRITER = None
        try:
            os.environ["KISS_SORCAR_SOCK"] = str(server_a.path)
            cli_daemon_bridge.send_event({"n": 1})
            assert server_a.wait_for_line(), (
                "sanity: first event never reached server A"
            )
            os.environ["KISS_SORCAR_SOCK"] = str(server_b.path)
            cli_daemon_bridge.send_event({"n": 2})
            assert server_b.wait_for_line(), (
                "event sent after KISS_SORCAR_SOCK changed still went to "
                "the OLD daemon socket — the cached writer must be "
                "re-resolved against _sock_path() on every send"
            )
        finally:
            if old_env is None:
                os.environ.pop("KISS_SORCAR_SOCK", None)
            else:
                os.environ["KISS_SORCAR_SOCK"] = old_env
            with cli_daemon_bridge._LOCK:
                if cli_daemon_bridge._WRITER is not None:
                    try:
                        cli_daemon_bridge._WRITER.close()
                    except OSError:
                        pass
                    cli_daemon_bridge._WRITER = None
            server_a.close()
            server_b.close()
