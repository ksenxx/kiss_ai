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

These tests run a real local-WSS daemon stand-in that records every
command the client sends, and drive ``daemon_client.run`` through each
abort path:

* ``KeyboardInterrupt`` injected mid-wait (the incident) → ``stop``
  then ``closeTab``.
* ``TimeoutError`` → ``closeTab`` only, no ``stop``.
* Normal completion → ``closeTab`` only, no ``stop``.
* Daemon drops the connection mid-wait → ``ConnectionError``
  propagates; the best-effort ``stop``/``closeTab`` sends hit the dead
  connection and their error is swallowed (covered end-to-end: the
  client must not mask the ``ConnectionError``).
"""

from __future__ import annotations

import asyncio
import gc
import json
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection

from kiss.agents.sorcar import daemon_client
from kiss.server.task_runner import inject_keyboard_interrupt
from kiss.tests.local_ws import fake_daemon


class _RecordingDaemon:
    """A real local-WSS daemon stand-in that records every client command.

    Served by :func:`fake_daemon` on an event loop of its own thread, so
    the synchronous ``daemon_client.run`` under test can block the test
    thread.  Reads the initial ``run`` command, sends
    ``{"type": "status", "running": true}``, and then follows the
    scripted *mode*:

    * ``"stream"`` — keeps sending harmless events forever (so the
      client's read loop iterates, exactly like a live task streaming
      deltas), while a reader records every further command the client
      sends.
    * ``"silent"`` — sends nothing further (drives the client into its
      read timeout), while recording further client commands.
    * ``"finish"`` — sends a terminal ``result`` + ``status
      running=false`` pair, then records further client commands.
    * ``"drop"`` — closes the connection immediately after the first
      status event (drives ``ConnectionError`` in the client).
    """

    def __init__(self, mode: str) -> None:
        """Start serving *mode*; returns once the endpoint file is written."""
        self.mode = mode
        self.endpoint_file = Path(tempfile.mkdtemp(prefix="kiss_cascade_")) / "sorcar-local.json"
        self.commands: list[dict[str, Any]] = []
        self.run_cmd: dict[str, Any] | None = None
        # Set once the recorder saw the client close: every command the
        # client sent before closing is recorded by then.
        self.disconnected = threading.Event()
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._closed = asyncio.Event()
        self._thread = threading.Thread(
            target=self._loop.run_until_complete, args=(self._serve(),), daemon=True,
        )
        self._thread.start()
        assert self._ready.wait(10), "the recording daemon never came up"

    async def _serve(self) -> None:
        """Keep the fake daemon up until :meth:`close`."""
        async with fake_daemon(
            self.endpoint_file.parent, self._handle, endpoint_file=self.endpoint_file,
        ):
            self._ready.set()
            await self._closed.wait()

    async def _record_commands(self, ws: ServerConnection) -> None:
        """Record every JSON command the client sends on *ws*."""
        try:
            async for message in ws:
                self.commands.append(json.loads(message))
        except ConnectionClosed:
            pass
        finally:
            self.disconnected.set()

    async def _handle(self, ws: ServerConnection) -> None:
        """Serve one authenticated client connection."""
        try:
            run_cmd: dict[str, Any] = json.loads(await ws.recv())
        except (ConnectionClosed, json.JSONDecodeError):
            return
        self.run_cmd = run_cmd
        tab_id = run_cmd.get("tabId", "")

        async def send(event: dict[str, Any]) -> bool:
            try:
                await ws.send(json.dumps(event))
                return True
            except ConnectionClosed:
                return False

        await send({"type": "status", "running": True, "tabId": tab_id})
        if self.mode == "drop":
            return  # Returning from the handler closes the connection.
        if self.mode == "finish":
            await send({
                "type": "result",
                "tabId": tab_id,
                "taskId": "task-cascade-1",
                "success": True,
                "text": "done",
                "cost": "$0.0100",
                "total_tokens": 5,
                "step_count": 1,
            })
            await send({"type": "status", "running": False, "tabId": tab_id})
        recorder = asyncio.ensure_future(self._record_commands(ws))
        try:
            while not self._closed.is_set():
                if self.mode == "stream" and not await send({
                    "type": "text_delta",
                    "text": "…",
                    "tabId": tab_id,
                }):
                    break
                await asyncio.sleep(0.05)
        finally:
            await asyncio.wait({recorder}, timeout=5)

    def wait_for_command(self, cmd_type: str, timeout: float = 5.0) -> bool:
        """Poll until a *cmd_type* command was recorded (or timeout)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(c.get("type") == cmd_type for c in self.commands):
                return True
            time.sleep(0.02)
        return False

    def close(self) -> None:
        """Stop serving, close every connection and remove the endpoint folder."""
        self._loop.call_soon_threadsafe(self._closed.set)
        self._thread.join(timeout=10)
        # Before the loop close, which raises if the thread outlived the join.
        shutil.rmtree(self.endpoint_file.parent, ignore_errors=True)
        self._loop.close()


class TestStopCascadeOnInterrupt:
    """The incident path: parent stopped mid-dispatch → child stopped."""

    @pytest.mark.parametrize("parent_task_id", ["", "parent-task-1"])
    def test_keyboard_interrupt_sends_stop(self, parent_task_id: str) -> None:
        """An injected KeyboardInterrupt must cascade a ``stop``.

        Reproduces the 2026-09-02 incident: the calling task's thread
        is force-interrupted (the daemon's ``_force_stop_thread``
        mechanism) while blocked in ``daemon_client.run``'s read loop.
        Before the fix the daemon received only ``closeTab`` and the
        dispatched task kept running, keeping the repository flagged
        busy for the manual Git Commit button.

        The stop is unconfirmed when the client leaves, so a top-level
        run keeps its tab (the task may still be running), while a
        sub-agent run, whose tab no surface shows, closes it after the
        stop.
        """
        daemon = _RecordingDaemon(mode="stream")
        outcome: dict[str, Any] = {}

        def call() -> None:
            try:
                daemon_client.run(
                    "long child task", endpoint_file=daemon.endpoint_file, timeout=60.0,
                    parent_task_id=parent_task_id,
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
            # The abort path ends with ``ws.close()``, whose close
            # handshake waits up to the library's 10 s close_timeout
            # when the daemon's loop is slow (a loaded 8-split run).
            worker.join(timeout=30)
            assert not worker.is_alive(), (
                "client wait never aborted; the worker is at:\n"
                + "".join(traceback.format_stack(sys._current_frames()[tid]))
            )
            assert isinstance(outcome["exc"], KeyboardInterrupt)
            assert daemon.wait_for_command("stop"), (
                "the interrupted dispatch never sent a stop for its "
                "child task — the orphan keeps the repo busy"
            )
            assert daemon.disconnected.wait(10)
            types = [c.get("type") for c in daemon.commands]
            assert ("closeTab" in types) == bool(parent_task_id), types
            if parent_task_id:
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

    @pytest.mark.parametrize("parent_task_id", ["", "parent-task-1"])
    def test_timeout_sends_no_stop(self, parent_task_id: str) -> None:
        """``TimeoutError`` must NOT stop the still-running child.

        The task keeps running, so a top-level run keeps its tab on
        every surface; only a sub-agent run (no surface shows its tab)
        closes it.
        """
        daemon = _RecordingDaemon(mode="silent")
        try:
            with pytest.raises(TimeoutError):
                daemon_client.run(
                    "slow child task", endpoint_file=daemon.endpoint_file, timeout=0.5,
                    parent_task_id=parent_task_id,
                )
            assert daemon.disconnected.wait(10)
            types = [c.get("type") for c in daemon.commands]
            assert types == (["closeTab"] if parent_task_id else []), (
                "a timeout must leave the dispatched task running"
            )
        finally:
            daemon.close()


class TestNoStopOnSuccess:
    """A normally finished dispatch must not send a spurious stop."""

    def test_success_sends_only_closetab(self) -> None:
        """A terminal result must close the tab without a ``stop``."""
        daemon = _RecordingDaemon(mode="finish")
        try:
            result = daemon_client.run(
                "quick child task", endpoint_file=daemon.endpoint_file, timeout=30.0,
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
    """A dead daemon connection must not mask the ``ConnectionError``."""

    @pytest.mark.parametrize("parent_task_id", ["", "parent-task-1"])
    def test_connection_error_propagates_despite_failed_stop(
        self, parent_task_id: str,
    ) -> None:
        """EOF mid-wait → ``ConnectionError``; failed sends swallowed.

        Covers the ``except OSError: pass`` branches of the stop and
        ``closeTab`` sends: the daemon closed the connection, so the
        cascade ``stop`` (and, for a sub-agent run, the ``closeTab``)
        hit a dead connection, and the original ``ConnectionError``
        must still reach the caller.
        """
        daemon = _RecordingDaemon(mode="drop")
        try:
            with pytest.raises(ConnectionError, match="closed the connection"):
                daemon_client.run(
                    "doomed child task", endpoint_file=daemon.endpoint_file, timeout=10.0,
                    parent_task_id=parent_task_id,
                )
        finally:
            daemon.close()


def _run_capturing(endpoint_file: Path, outcome: dict[str, Any]) -> None:
    """Run a dispatch with a *cancel* event (0.5 s wakes); store how it ended."""
    try:
        daemon_client.run(
            "long child task", endpoint_file=endpoint_file, timeout=60.0,
            cancel=threading.Event(),
        )
        outcome["exc"] = None
    except BaseException as exc:  # noqa: BLE001 — capture for assert
        outcome["exc"] = exc


def _new_client_connection(before: list[ClientConnection]) -> ClientConnection:
    """Return the live websockets client connection not in *before*."""
    for obj in gc.get_objects():
        if isinstance(obj, ClientConnection) and all(obj is not old for old in before):
            return obj
    raise AssertionError("the client never connected")


def _blocked_in_assembler_get(tid: int, assembler: object) -> bool:
    """Whether thread *tid*'s innermost frame is *assembler*'s ``get``."""
    frame = sys._current_frames().get(tid)
    return (
        frame is not None
        and frame.f_code.co_name == "get"
        and frame.f_locals.get("self") is assembler
    )


class TestStopWhileReadingAFrame:
    """A Stop that lands while the client takes a frame must not hang it.

    CPython raises an injected async exception at the first check after
    a call returns, which can fall between a lock's ``__enter__``
    returning and the ``with`` block's cleanup being armed.  When the
    daemon's Stop landed while the task thread was acquiring the
    websockets frame-assembler mutex inside ``ws.recv()``, the mutex
    stayed held: ``run``'s ``finally`` then blocked for ever in
    ``ws.close()``, which takes the same mutex (seen as a load-only
    failure of ``test_keyboard_interrupt_sends_stop``).
    """

    def test_stop_while_waiting_for_the_frame_mutex(self) -> None:
        """The run aborts and sends its ``stop`` though the Stop met the mutex.

        Forces the interleaving with the real objects: the test holds
        the client connection's assembler mutex until the thread running
        ``daemon_client.run`` is blocked on it in ``Assembler.get`` (or
        2 s pass, when only a helper thread reads the socket), injects
        the Stop into that thread, and releases the mutex.
        """
        daemon = _RecordingDaemon(mode="stream")
        outcome: dict[str, Any] = {}
        before = [o for o in gc.get_objects() if isinstance(o, ClientConnection)]
        worker = threading.Thread(
            target=_run_capturing, args=(daemon.endpoint_file, outcome), daemon=True,
        )
        worker.start()
        try:
            deadline = time.monotonic() + 5
            while daemon.run_cmd is None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert daemon.run_cmd is not None, "client never sent run"
            assembler = _new_client_connection(before).recv_messages
            tid = worker.ident
            assert tid is not None
            with assembler.mutex:
                deadline = time.monotonic() + 2
                while (
                    not _blocked_in_assembler_get(tid, assembler)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                assert inject_keyboard_interrupt(tid) == 1
            worker.join(timeout=30)
            assert not worker.is_alive(), (
                "the stopped client never finished; the worker is at:\n"
                + "".join(traceback.format_stack(sys._current_frames()[tid]))
            )
            assert isinstance(outcome["exc"], KeyboardInterrupt)
            assert daemon.wait_for_command("stop")
        finally:
            daemon.close()
