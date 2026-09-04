# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-03 (server-core): a Stop landing in untrusted setup code.

``_stop_task``'s watchdog cancels a run by injecting an asynchronous
``KeyboardInterrupt`` into the task thread.  The untrusted-code
loaders — ``apply_agent_overrides`` (agent-script ``get_X()`` getters)
and ``load_tools_file`` (``get_tools()``) — execute caller-supplied
Python on that thread and convert EVERY raise, ``BaseException``
included, into their diagnostic error type.  An injected stop landing
while such a getter runs was therefore swallowed:

* the run was reported ``"Task failed: AgentFileError: get_prompt()
  ... raised: KeyboardInterrupt"`` (or the ``ToolsFileError``
  equivalent) instead of ``"Task stopped by user"``;
* ``_cancel_outcome`` never ran, so the stop was never acknowledged
  (``AgentState.stop_acknowledged`` stayed ``False``) and the
  watchdog's 5-second retry could land a SECOND interrupt in the
  result broadcasting / persistence should they take that long.

Everything here is real: a ``RemoteAccessServer`` serving its Unix
domain socket, ``run`` / ``stop`` commands sent as ``daemon_client``
would (the ``run`` carries a client-minted ``taskId`` run token and
the ``stop`` repeats it, exercising the run-token guard end to end),
the real worker thread and the real watchdog.  The interrupt lands
deterministically inside the untrusted getter because the getter
blocks (polling for a release file) until the watchdog's injection
arrives.  No LLM is ever invoked: releasing the getter makes it raise,
so a run that survives the stop still dies in setup.

Branch-coverage notes for the fix (``_stop_interrupt_wrapped``):

* ``id(cur) in seen`` (cause/context cycle guard) is unreachable
  without hand-building an exception whose ``__context__`` chain is
  cyclic — real raises never produce one — so it stays uncovered by
  design.
* ``state.interrupted_by_shutdown`` as the stop-requested half is the
  shutdown path (``_stop_active_agent_tasks``), already exercised by
  ``test_audit0902_fix2_server_setup_stop_ack.py``'s shutdown case
  for the unwrapped interrupt; the wrapped variant differs only in
  the flag consulted.
* ``_run_task``'s ``state is None`` re-resolve in its outer catch is
  reachable only when the interrupt lands in the two-statement window
  between ``apply_agent_overrides`` returning and the state
  resolution in the try — an interrupt INSIDE the overrides is always
  wrapped, so no deterministic test can park there; the arm stays
  uncovered by design.
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
from kiss.server.agent_state import AgentState
from kiss.server.task_runner import _stop_interrupt_wrapped
from kiss.server.web_server import RemoteAccessServer

_STOP_LABEL = "Task stopped by user"


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


_BLOCKING_GETTER = textwrap.dedent(
    """
    import pathlib
    import time

    _DIR = pathlib.Path(__file__).resolve().parent


    def {getter}():
        \"\"\"Block until interrupted; raise if released or timed out.\"\"\"
        (_DIR / "entered-{marker}").write_text("1", encoding="utf-8")
        deadline = time.time() + 60
        while time.time() < deadline:
            if (_DIR / "release").exists():
                raise RuntimeError("released before the stop landed")
            time.sleep(0.02)
        raise RuntimeError("timed out waiting for the stop")
    """
)

_BROKEN_GETTER = textwrap.dedent(
    """
    def get_prompt():
        \"\"\"Raise immediately — a genuinely broken agent script.\"\"\"
        raise ValueError("script bug")
    """
)


class TestStopWrappedInterrupt(TestCase):
    """An injected stop swallowed by a loader is still a user stop."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-audit0903-wrap-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.sock_path = str(self.tmp / "sorcar.sock")
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.remote = RemoteAccessServer(
            uds_path=self.sock_path, work_dir=str(self.work_dir),
        )
        self.remote._printer._loop = self.loop
        self.remote._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.remote._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=10)
        self.client = _UdsClient(self.sock_path)

    def tearDown(self) -> None:
        # Release any still-blocked getter so its worker thread exits
        # (the getter raises on release, failing the run in setup, so
        # no model call can ever happen).
        (self.tmp / "release").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        self.client.close()

        async def _drain() -> None:
            # Await the accepted handlers' ``finally`` blocks (closed
            # client sockets unblock their ``readline``) instead of
            # stopping the loop under them, which leaked pending
            # ``_uds_handler`` tasks ("Task was destroyed but it is
            # pending!") into later tests.
            self.uds_server.close()
            for writer in list(self.remote._printer._uds_writers):
                writer.close()
            await self.remote._drain_tasks(set(self.remote._uds_handler_tasks))

        asyncio.run_coroutine_threadsafe(_drain(), self.loop).result(timeout=15)
        self.loop.call_soon_threadsafe(self.loop.stop)
        agent_state.agent_states.clear()

    def _write_script(self, name: str, getter: str, marker: str) -> str:
        path = self.tmp / name
        path.write_text(
            _BLOCKING_GETTER.format(getter=getter, marker=marker),
            encoding="utf-8",
        )
        return str(path)

    def _wait_marker(self, marker: str) -> None:
        deadline = time.monotonic() + 30
        entered = self.tmp / f"entered-{marker}"
        while not entered.exists():
            if time.monotonic() > deadline:
                raise AssertionError(f"{entered} never appeared")
            time.sleep(0.02)

    def _run_and_stop(self, tab_id: str, marker: str, **run_fields: Any) -> None:
        """Send a real ``run``, park it in the getter, and stop it."""
        run_token = f"tok-{tab_id}"
        self.client.send({
            "type": "run",
            "prompt": f"wrapped-stop {tab_id}",
            "tabId": tab_id,
            "taskId": run_token,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
            **run_fields,
        })
        self._wait_marker(marker)
        # The stop mirrors daemon_client's frames: same tab, same
        # client-minted run token — the run-token guard must accept it.
        self.client.send({
            "type": "stop", "tabId": tab_id, "taskId": run_token,
        })
        ack = self.client.wait_for("stop_ack", tab_id)
        self.assertTrue(ack["accepted"])
        result = self.client.wait_for("result", tab_id)
        self.assertEqual(
            result["text"],
            _STOP_LABEL,
            "BUG: the loader wrapped the stop's KeyboardInterrupt into "
            f"a task error: {result['text']!r}",
        )
        self.assertFalse(result["success"])
        # The terminal status the daemon_client stop confirmation
        # waits on must still arrive.
        self.client.wait_for("status", tab_id, running=False)

    def test_stop_during_agent_script_getter_is_a_user_stop(self) -> None:
        """KI inside ``get_prompt()`` (AgentFileError site, ``_run_task``)."""
        script = self._write_script("agent.py", "get_prompt", "agent")
        self._run_and_stop("wrap-agent-tab", "agent", agentPath=script)

    def test_stop_during_tools_file_get_tools_is_a_user_stop(self) -> None:
        """KI inside ``get_tools()`` (ToolsFileError site, ``_run_task_inner``)."""
        tools = self._write_script("tools.py", "get_tools", "tools")
        self._run_and_stop("wrap-tools-tab", "tools", toolsFile=tools)

    def test_broken_script_without_stop_stays_a_task_error(self) -> None:
        """No stop requested → a raising getter keeps its diagnostic.

        Covers ``_stop_interrupt_wrapped``'s stop-not-requested early
        return: the run's stop event exists but is never set, so the
        loader's error must surface verbatim, not as a cancellation.
        """
        script = self.tmp / "broken.py"
        script.write_text(_BROKEN_GETTER, encoding="utf-8")
        tab_id = "wrap-broken-tab"
        self.client.send({
            "type": "run",
            "prompt": "broken script",
            "tabId": tab_id,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "useParallel": False,
            "autoCommit": False,
            "agentPath": str(script),
        })
        result = self.client.wait_for("result", tab_id)
        self.assertIn("AgentFileError", result["text"])
        self.assertIn("script bug", result["text"])
        self.assertFalse(result["success"])
        self.client.wait_for("status", tab_id, running=False)


class TestStopInterruptWrappedPredicate(TestCase):
    """Remaining branches of ``_stop_interrupt_wrapped``, on real objects.

    The UDS tests above cover the two production catch sites; the
    predicate's other decision branches are exercised here directly —
    real :class:`AgentState` objects and real exception chains, no
    doubles.  The wire timing needed to land a NON-interrupt script
    failure while a stop is pending (release racing the watchdog's
    one-second injection) is inherently flaky, so the
    stop-pending-but-no-interrupt branch is pinned here instead.
    """

    def test_predicate_branches(self) -> None:
        wrapped_ki = Exception("loader diagnostic")
        wrapped_ki.__cause__ = KeyboardInterrupt()
        plain_error = Exception("loader diagnostic")
        plain_error.__cause__ = ValueError("script bug")

        # No stop requested: never a cancellation, KI in chain or not.
        idle = AgentState("audit0903-idle", stop_event=threading.Event())
        self.assertFalse(_stop_interrupt_wrapped(wrapped_ki, idle))

        # A state whose run already ended (stop event cleared).
        cleared = AgentState("audit0903-cleared")
        self.assertFalse(_stop_interrupt_wrapped(wrapped_ki, cleared))

        # Stop requested, but the failure does not wrap the interrupt:
        # the script's own error keeps its diagnostic.
        stopped = AgentState("audit0903-stopped", stop_event=threading.Event())
        assert stopped.stop_event is not None
        stopped.stop_event.set()
        self.assertFalse(_stop_interrupt_wrapped(plain_error, stopped))

        # Stop requested and the interrupt sits in the cause chain.
        self.assertTrue(_stop_interrupt_wrapped(wrapped_ki, stopped))

        # The interrupt may also arrive via __context__ (a getter that
        # caught it and raised something else).
        contextual = Exception("secondary")
        contextual.__context__ = KeyboardInterrupt()
        self.assertTrue(_stop_interrupt_wrapped(contextual, stopped))

        # Shutdown-flagged states count as stop-requested too.
        shutdown = AgentState("audit0903-shutdown")
        shutdown.interrupted_by_shutdown = True
        self.assertTrue(_stop_interrupt_wrapped(wrapped_ki, shutdown))
        self.assertFalse(_stop_interrupt_wrapped(plain_error, shutdown))
