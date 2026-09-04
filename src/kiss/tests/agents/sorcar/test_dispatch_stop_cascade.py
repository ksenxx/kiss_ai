# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the dispatch stop cascade in ``daemon_client.run``.

Incident (2026-09-02): a task running on the main working tree
dispatched a gpt-5.6-sol review through ``run_agent`` →
``daemon_client.run`` (tab ``api-…``, ``use_worktree=False``).  The
user pressed Stop; the parent thread — blocked in ``run``'s event-read
loop — got the injected ``KeyboardInterrupt`` and ended with "Task
stopped by user", but nothing stopped the dispatched child task, which
kept running invisibly for another minute.  Because the orphan was a
non-worktree task, its agent state kept ``is_running_non_wt=True`` on
the repository, so pressing the manual Git Commit button was refused
with "A task is still running in this folder; wait for it to finish
before committing." while no task appeared to be running.

The fix: when ``run``'s wait aborts for any reason other than a
``TimeoutError`` (whose documented contract is "the task keeps
running"), the client sends the daemon a ``{"type": "stop"}`` command
for the dispatched task's tab before the existing ``closeTab``.

These tests run a real UNIX-domain-socket daemon stand-in that records
every command the client sends, and drive ``daemon_client.run`` through
each abort path:

* ``KeyboardInterrupt`` injected mid-wait (the incident) → ``stop``
  then ``closeTab``.
* ``TimeoutError`` → ``closeTab`` only, no ``stop``.
* Normal completion → ``closeTab`` only, no ``stop``.
* Daemon drops the connection mid-wait → ``ConnectionError``
  propagates; the best-effort ``stop``/``closeTab`` sends hit the dead
  socket and their ``OSError`` is swallowed (covered end-to-end: the
  client must not mask the ``ConnectionError``).
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar import daemon_client
from kiss.server.task_runner import inject_keyboard_interrupt


class _RecordingDaemon:
    """A real UDS daemon stand-in that records every client command.

    Accepts one connection, reads the initial ``run`` command, sends
    ``{"type": "status", "running": true}``, and then follows the
    scripted *mode*:

    * ``"stream"`` — keeps sending harmless event lines forever (so the
      client's read loop iterates, exactly like a live task streaming
      deltas), while a background reader records every further command
      the client sends.
    * ``"silent"`` — sends nothing further (drives the client into its
      read timeout), while recording further client commands.
    * ``"finish"`` — sends a terminal ``result`` + ``status
      running=false`` pair, then records further client commands.
    * ``"drop"`` — closes the connection immediately after the first
      status event (drives ``ConnectionError`` in the client).
    """

    def __init__(self, sock_path: Path, mode: str) -> None:
        """Bind a UNIX-domain listener at *sock_path* for *mode*."""
        self.mode = mode
        self.commands: list[dict[str, Any]] = []
        self.run_cmd: dict[str, Any] | None = None
        self._stop = threading.Event()
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(str(sock_path))
        self._srv.listen(1)
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _record_commands(self, reader: Any) -> None:
        """Record every newline-framed JSON command from *reader*."""
        while not self._stop.is_set():
            try:
                line = reader.readline()
            except OSError:
                return
            if not line:
                return
            try:
                self.commands.append(json.loads(line.decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    def _serve(self) -> None:
        try:
            conn, _ = self._srv.accept()
        except OSError:
            return
        with conn:
            reader = conn.makefile("rb")
            try:
                run_cmd: dict[str, Any] = json.loads(
                    reader.readline().decode("utf-8"),
                )
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                return
            self.run_cmd = run_cmd
            tab_id = run_cmd.get("tabId", "")

            def send(event: dict[str, Any]) -> bool:
                try:
                    conn.sendall(
                        json.dumps(event).encode("utf-8") + b"\n",
                    )
                    return True
                except OSError:
                    return False

            send({"type": "status", "running": True, "tabId": tab_id})
            if self.mode == "drop":
                return  # ``with conn`` closes the socket → client EOF.
            if self.mode == "finish":
                send({
                    "type": "result",
                    "tabId": tab_id,
                    "taskId": "task-cascade-1",
                    "success": True,
                    "text": "done",
                    "cost": "$0.0100",
                    "total_tokens": 5,
                    "step_count": 1,
                })
                send({"type": "status", "running": False, "tabId": tab_id})
            recorder = threading.Thread(
                target=self._record_commands, args=(reader,), daemon=True,
            )
            recorder.start()
            while not self._stop.is_set():
                if self.mode == "stream":
                    if not send({
                        "type": "text_delta",
                        "text": "…",
                        "tabId": tab_id,
                    }):
                        break
                time.sleep(0.05)
            recorder.join(timeout=5)

    def wait_for_command(self, cmd_type: str, timeout: float = 5.0) -> bool:
        """Poll until a *cmd_type* command was recorded (or timeout)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(c.get("type") == cmd_type for c in self.commands):
                return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        """Shut the listener and the serving thread down."""
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass
        self._thread.join(timeout=5)


def _sock_path() -> Path:
    """Return a fresh UDS path in a private temporary directory."""
    return Path(tempfile.mkdtemp(prefix="kiss_cascade_")) / "daemon.sock"


class TestStopCascadeOnInterrupt:
    """The incident path: parent stopped mid-dispatch → child stopped."""

    def test_keyboard_interrupt_sends_stop_then_closetab(self) -> None:
        """An injected KeyboardInterrupt must cascade a ``stop``.

        Reproduces the 2026-09-02 incident: the calling task's thread
        is force-interrupted (the daemon's ``_force_stop_thread``
        mechanism) while blocked in ``daemon_client.run``'s read loop.
        Before the fix the daemon received only ``closeTab`` and the
        dispatched task kept running, keeping the repository flagged
        busy for the manual Git Commit button.
        """
        path = _sock_path()
        daemon = _RecordingDaemon(path, mode="stream")
        outcome: dict[str, Any] = {}

        def call() -> None:
            try:
                daemon_client.run(
                    "long child task", sock_path=path, timeout=60.0,
                )
                outcome["exc"] = None
            except BaseException as exc:  # noqa: BLE001 — capture for assert
                outcome["exc"] = exc

        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        try:
            # Let the client connect and enter its read loop.
            deadline = time.monotonic() + 5
            while daemon.run_cmd is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert daemon.run_cmd is not None, "client never sent run"
            time.sleep(0.2)
            tid = worker.ident
            assert tid is not None
            assert inject_keyboard_interrupt(tid) == 1
            worker.join(timeout=10)
            assert not worker.is_alive(), "client wait never aborted"
            assert isinstance(outcome["exc"], KeyboardInterrupt)
            assert daemon.wait_for_command("stop"), (
                "the interrupted dispatch never sent a stop for its "
                "child task — the orphan keeps the repo busy"
            )
            assert daemon.wait_for_command("closeTab")
            types = [c.get("type") for c in daemon.commands]
            assert types.index("stop") < types.index("closeTab")
            stop_cmd = next(
                c for c in daemon.commands if c.get("type") == "stop"
            )
            assert stop_cmd.get("tabId") == daemon.run_cmd.get("tabId")
            # The stop must be qualified by the SAME client-minted run
            # token the run was submitted with, so the daemon can
            # reject it if the tab was reused by a newer run
            # (gpt-5.6-sol review finding).
            assert stop_cmd.get("taskId") == daemon.run_cmd.get("taskId")
            assert stop_cmd.get("taskId")
        finally:
            daemon.close()


class TestNoStopOnTimeout:
    """A timeout keeps the documented fire-and-forget contract."""

    def test_timeout_sends_only_closetab(self) -> None:
        """``TimeoutError`` must NOT stop the still-running child."""
        path = _sock_path()
        daemon = _RecordingDaemon(path, mode="silent")
        try:
            with pytest.raises(TimeoutError):
                daemon_client.run(
                    "slow child task", sock_path=path, timeout=0.5,
                )
            assert daemon.wait_for_command("closeTab")
            assert not any(
                c.get("type") == "stop" for c in daemon.commands
            ), "a timeout must leave the dispatched task running"
        finally:
            daemon.close()


class TestNoStopOnSuccess:
    """A normally finished dispatch must not send a spurious stop."""

    def test_success_sends_only_closetab(self) -> None:
        """A terminal result must close the tab without a ``stop``."""
        path = _sock_path()
        daemon = _RecordingDaemon(path, mode="finish")
        try:
            result = daemon_client.run(
                "quick child task", sock_path=path, timeout=30.0,
            )
            assert result.success is True
            assert result.text == "done"
            assert daemon.wait_for_command("closeTab")
            assert not any(
                c.get("type") == "stop" for c in daemon.commands
            ), "a finished dispatch must not send a stop"
        finally:
            daemon.close()


class TestConnectionDropStillRaises:
    """A dead daemon socket must not mask the ``ConnectionError``."""

    def test_connection_error_propagates_despite_failed_stop(self) -> None:
        """EOF mid-wait → ``ConnectionError``; failed sends swallowed.

        Covers the ``except OSError: pass`` branch of the stop send:
        the daemon closed the connection, so both the cascade ``stop``
        and the ``closeTab`` hit a dead socket, and the original
        ``ConnectionError`` must still reach the caller.
        """
        path = _sock_path()
        daemon = _RecordingDaemon(path, mode="drop")
        try:
            with pytest.raises(ConnectionError, match="closed the connection"):
                daemon_client.run(
                    "doomed child task", sock_path=path, timeout=10.0,
                )
        finally:
            daemon.close()
