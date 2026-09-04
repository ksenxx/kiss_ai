# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03 (server, fix round): shutdown/stop racing ``thread.start()``.

``_cmd_run`` registers the run's :class:`AgentState` (worker thread
installed, not started) under ``STATE_LOCK``, releases the lock,
publishes the tab in the shared registry (a disk write), broadcasts
``clear``, and only then calls ``thread.start()``.  Two callers race
that window:

* ``RemoteAccessServer._stop_active_agent_tasks`` (SIGTERM graceful
  shutdown, ``stop_async``) selects the state via ``busy()`` — which
  deliberately counts a created-but-unstarted worker as live — and
  used to call ``thread.join()`` unconditionally, which raises
  ``RuntimeError: cannot join thread before it is started``.  Worse,
  the in-flight ``_cmd_run`` then reached ``thread.start()`` AFTER the
  shutdown sweep, so the run's untrusted setup executed with no
  watchdog and its history row was abandoned (review Finding 1).
* ``_stop_task`` accepts the stop and arms a start-waiting watchdog,
  but the watchdog gives up after 30 s; a ``_cmd_run`` stalled longer
  than that in the registry write then started the run with nothing
  left to enforce the accepted stop (review Finding 2).

The fix is a start/cancel handshake owned by ``_cmd_run``: immediately
before ``thread.start()`` it atomically (under ``STATE_LOCK``, the
same lock the sweep and ``_stop_task`` flag under) observes the
stop/shutdown state and routes a pre-cancelled run straight through
terminal cancellation — the worker thread is never started and the
run's untrusted setup never executes.

Everything here is real: a ``RemoteAccessServer`` serving its Unix
domain socket, TWO independent client connections, real ``run`` /
``stop`` commands, the real shutdown sweep.  The pre-start window is
held open deterministically by parking the production registry
publication (:class:`TabRegistry` subclass whose ``update_tab`` parks
then delegates) — the disk write that keeps the window open in
production.  The agent script's getter writes a marker file on entry,
so "user setup never executed" is directly observable.  No LLM is ever
invoked: a run that does slip through blocks in the getter and raises
when the test releases it.

Branch-coverage notes:

* ``wait_for_thread_start``'s deadline give-up remains unreachable
  without stalling ``thread.start()`` for the full deadline; with the
  handshake in place the give-up is harmless (the accepted stop is
  honored by ``_cmd_run`` itself) and the arm stays uncovered by
  design (``pragma: no cover`` at the call sites).
* The handshake's normal arm (no stop pending → ``thread.start()``)
  is exercised by every other run test in this suite.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from typing import Any
from unittest import TestCase

from kiss.server import agent_state
from kiss.server.tab_registry import TabRegistry
from kiss.server.web_server import RemoteAccessServer

_STOP_LABEL = "Task stopped by user"
_SHUTDOWN_LABEL = "Task interrupted by server restart/shutdown"

_BLOCKING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    _DIR = pathlib.Path(__file__).resolve().parent


    def get_prompt():
        \"\"\"Block until released; raise so no model is ever invoked.\"\"\"
        (_DIR / "entered").write_text("1", encoding="utf-8")
        deadline = time.time() + 60
        while time.time() < deadline:
            if (_DIR / "release").exists():
                raise RuntimeError("released before the stop landed")
            time.sleep(0.02)
        raise RuntimeError("timed out waiting for the stop")
    """
)


class _ParkingTabRegistry(TabRegistry):
    """Real registry that parks the FIRST ``update_tab`` of one tab.

    ``_cmd_run`` publishes the tab in the registry between registering
    the run's state (unstarted worker installed) and
    ``thread.start()`` — parking here holds the production pre-start
    window open without rebuilding any state by hand.
    """

    def __init__(self, path: Path, park_tab: str) -> None:
        super().__init__(path)
        self.park_tab = park_tab
        self.parked = threading.Event()
        self.release = threading.Event()

    def update_tab(self, tab_id: str, **kwargs: Any) -> tuple[bool, list[str]]:
        """Park the designated tab's first update, then delegate."""
        if tab_id == self.park_tab and not self.parked.is_set():
            self.parked.set()
            if not self.release.wait(timeout=60):
                raise TimeoutError("parked too long in update_tab")
        return super().update_tab(tab_id, **kwargs)


class _RaisingTabRegistry(TabRegistry):
    """Real registry whose ``update_tab`` fails for one tab.

    A registry disk write can genuinely fail (unwritable KISS dir);
    ``_cmd_run`` must then disown the registered-but-unstarted worker
    so the tab is not bricked by a phantom ``task_thread``.
    """

    def __init__(self, path: Path, fail_tab: str) -> None:
        super().__init__(path)
        self.fail_tab = fail_tab

    def update_tab(self, tab_id: str, **kwargs: Any) -> tuple[bool, list[str]]:
        """Raise for the designated tab, else delegate."""
        if tab_id == self.fail_tab:
            raise RuntimeError("registry write failed")
        return super().update_tab(tab_id, **kwargs)


class _UdsClient:
    """Newline-delimited JSON client of the server's Unix domain socket."""

    def __init__(self, sock_path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(sock_path)
        self.events: list[dict[str, Any]] = []
        self._cond = threading.Condition()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        buf = b""
        while True:
            try:
                chunk = self.sock.recv(65536)
            except OSError:
                chunk = b""
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if line.strip():
                    with self._cond:
                        self.events.append(json.loads(line))
                        self._cond.notify_all()

    def send(self, cmd: dict[str, Any]) -> None:
        self.sock.sendall((json.dumps(cmd) + "\n").encode())

    def wait_for(
        self,
        event_type: str,
        tab_id: str,
        timeout: float = 30.0,
        **fields: Any,
    ) -> dict[str, Any]:
        """Return the first *event_type* event for *tab_id* matching *fields*."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for ev in self.events:
                    if (
                        ev.get("type") == event_type
                        and ev.get("tabId") == tab_id
                        and all(ev.get(k) == v for k, v in fields.items())
                    ):
                        return ev
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(
                        f"no {event_type!r} for {tab_id!r}; got "
                        f"{[e.get('type') for e in self.events]}",
                    )
                self._cond.wait(remaining)

    def close(self) -> None:
        self.sock.close()


class TestPrestartShutdownAndStopHandshake(TestCase):
    """A run must never start after a shutdown sweep or an accepted stop."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit0903-uds-prestart-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.script = self.tmp / "agent.py"
        self.script.write_text(_BLOCKING_SCRIPT, encoding="utf-8")
        self.sock_path = str(self.tmp / "sorcar.sock")
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.remote = RemoteAccessServer(
            uds_path=self.sock_path, work_dir=str(self.work_dir),
        )
        self.remote._printer._loop = self.loop
        self.remote._loop = self.loop
        self.registry: _ParkingTabRegistry | None = None
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.remote._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=10)
        self.client1 = _UdsClient(self.sock_path)
        self.client2 = _UdsClient(self.sock_path)

    def tearDown(self) -> None:
        (self.tmp / "release").write_text("1", encoding="utf-8")
        if self.registry is not None:
            self.registry.release.set()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        self.client1.close()
        self.client2.close()

        async def _drain() -> None:
            # No ``wait_closed()``: on Python 3.14 it also waits for
            # every handler coroutine, which ``_drain_tasks`` below
            # already does — with a cancellation fallback for
            # stragglers, so teardown always terminates.
            self.uds_server.close()
            for writer in list(self.remote._printer._uds_writers):
                writer.close()
            await self.remote._drain_tasks(set(self.remote._uds_handler_tasks))

        asyncio.run_coroutine_threadsafe(_drain(), self.loop).result(timeout=15)
        self.loop.call_soon_threadsafe(self.loop.stop)
        agent_state.agent_states.clear()

    def _park_registry(self, tab_id: str) -> _ParkingTabRegistry:
        """Swap in a registry that parks *tab_id*'s run publication."""
        registry = _ParkingTabRegistry(self.tmp / "tabs.json", tab_id)
        self.registry = registry
        self.remote._vscode_server.tab_registry = registry
        return registry

    def _send_run(self, tab_id: str, run_token: str) -> None:
        self.client1.send({
            "type": "run",
            "prompt": f"prestart {tab_id}",
            "tabId": tab_id,
            "taskId": run_token,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
            "agentPath": str(self.script),
        })

    def test_shutdown_sweep_prestart_run_never_starts(self) -> None:
        """Finding 1: the sweep must neither crash nor let the run start."""
        tab_id = "prestart-shutdown-tab"
        registry = self._park_registry(tab_id)
        self._send_run(tab_id, "tok-shutdown")
        self.assertTrue(
            registry.parked.wait(timeout=30),
            "the run never reached the registry publication window",
        )

        sweep_error: list[BaseException] = []

        def _sweep() -> None:
            try:
                self.remote._stop_active_agent_tasks()
            except BaseException as exc:  # noqa: BLE001 — recorded for assertion
                sweep_error.append(exc)

        sweeper = threading.Thread(target=_sweep, daemon=True)
        sweeper.start()

        # The sweep must have flagged the pre-start state before the
        # window is released, so the handshake decides against start.
        deadline = time.monotonic() + 30
        state = None
        while time.monotonic() < deadline:
            with agent_state.STATE_LOCK:
                state = agent_state.find_by_tab(tab_id)
                if state is not None and state.interrupted_by_shutdown:
                    break
            time.sleep(0.02)
        self.assertIsNotNone(state, "the run's state was never registered")
        assert state is not None
        self.assertTrue(
            state.interrupted_by_shutdown,
            "the shutdown sweep never flagged the pre-start run",
        )

        registry.release.set()
        sweeper.join(timeout=30)
        self.assertFalse(sweeper.is_alive(), "shutdown sweep never returned")
        self.assertEqual(
            sweep_error,
            [],
            "BUG: the shutdown sweep crashed on the not-yet-started "
            f"worker thread: {sweep_error!r}",
        )

        # BOTH clients see the terminal cancellation of the swept run.
        for client in (self.client1, self.client2):
            result = client.wait_for("result", tab_id)
            self.assertEqual(result["text"], _SHUTDOWN_LABEL)
            self.assertFalse(result["success"])
            client.wait_for("status", tab_id, running=False)

        # The run's untrusted setup never executed: no watchdog-free
        # start slipped through after the sweep.
        self.assertFalse(
            (self.tmp / "entered").exists(),
            "BUG: the run started AFTER the shutdown sweep and executed "
            "its untrusted setup with no watchdog",
        )
        self.assertFalse(state.thread_alive())

    def test_registry_failure_disowns_the_unstarted_worker(self) -> None:
        """A pre-start crash still releases the tab's phantom thread.

        The handshake sits inside ``_cmd_run``'s existing try; a
        registry failure before it must keep unwinding through the
        disown cleanup (``task_thread``/``stop_event`` cleared) so the
        tab accepts the next run instead of queueing behind a thread
        that will never start.
        """
        tab_id = "prestart-raise-tab"
        server = self.remote._vscode_server
        server.tab_registry = _RaisingTabRegistry(
            self.tmp / "tabs-raise.json", tab_id,
        )
        with self.assertRaises(RuntimeError):
            server._cmd_run({
                "type": "run",
                "prompt": "raising registry",
                "tabId": tab_id,
                "workDir": str(self.work_dir),
                "useWorktree": False,
                "useParallel": False,
                "autoCommit": False,
                "agentPath": str(self.script),
            })
        with agent_state.STATE_LOCK:
            state = agent_state.find_by_tab(tab_id)
        self.assertIsNotNone(state)
        assert state is not None
        self.assertIsNone(state.task_thread)
        self.assertIsNone(state.stop_event)
        self.assertFalse((self.tmp / "entered").exists())

    def test_accepted_prestart_stop_is_honored_without_starting(self) -> None:
        """Finding 2: an accepted stop is honored by the handshake itself."""
        tab_id = "prestart-stop-tab"
        run_token = "tok-stop"
        registry = self._park_registry(tab_id)
        self._send_run(tab_id, run_token)
        self.assertTrue(
            registry.parked.wait(timeout=30),
            "the run never reached the registry publication window",
        )

        # The stop arrives on the SECOND connection (the launcher's
        # handler is busy inside the parked run command) — exactly how
        # daemon_client's abort-cascade stop reaches the daemon.
        self.client2.send({
            "type": "stop", "tabId": tab_id, "taskId": run_token,
        })
        ack = self.client2.wait_for("stop_ack", tab_id)
        self.assertTrue(ack["accepted"], "pre-start stop was not accepted")

        registry.release.set()
        result = self.client1.wait_for("result", tab_id)
        self.assertEqual(result["text"], _STOP_LABEL)
        self.assertFalse(result["success"])
        self.client1.wait_for("status", tab_id, running=False)

        # The accepted stop was honored by _cmd_run's handshake: the
        # worker never started, so the untrusted getter never ran —
        # this holds even when the stop watchdog's start-wait gave up.
        self.assertFalse(
            (self.tmp / "entered").exists(),
            "BUG: the accepted stop was only enforced by the watchdog's "
            "injection INSIDE the untrusted setup; the run must never "
            "have started",
        )
