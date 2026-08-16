# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the server-core audit findings.

Every test here drives a **real** :class:`RemoteAccessServer` bound to
a temporary Unix-domain socket and an ephemeral TCP port, with
persistence redirected to a scratch sqlite database, and speaks the
real newline-delimited JSON wire protocol over a real client socket.
Nothing is mocked, patched or doubled: the agent states the handlers
consult are the real :mod:`kiss.server.agent_state` objects, the
worker threads are real threads, and the git commands are real git
processes.

Findings covered (audit ``08-server-web.md`` / ``09-server-rest.md``):

* **F08-1** — ``_cmd_run`` refused a run while a merge is in flight by
  broadcasting only an ``error``, leaving the optimistically raised
  ``status running:true`` set forever, so the tab's composer stayed
  disabled until the tab was closed.
* **F08-2** — the shutdown sweep and the two liveness snapshots gated
  on ``is_task_active`` alone, although a worker that has started but
  not yet raised the flag is a real, live task
  (:meth:`AgentState.thread_alive`).
* **F08-5** — ``VSCodeServer.__init__`` cleared the *process-global*
  agent-state registry, so merely constructing a second server
  detached the first server's live tasks.
* **F08-6** — ``VSCodeServer._tab_opened_task_ids`` was a mutable
  **class** attribute shadowed by an instance attribute: dead, and a
  latent cross-instance shared-state hazard.
* **F08-7** — ``_handle_submit`` rebuilt the ``run`` command without
  ``connId``, so a task launched from the browser recorded ``""``
  while the identical task launched from VS Code recorded the real
  connection id.
* **R09-8** — ``generateCommitMessage`` spawned an unbounded thread
  per click with no in-flight dedup, so N clicks ran N generations
  (N billed LLM calls) racing to stamp the same tab.

No test makes a paid LLM call: runs are submitted with a model name
that is not in ``get_available_models()``, which makes the real worker
return at ``task_runner``'s "No model available" guard before any
agent work, and the commit-message tests block inside real ``git`` so
the diff comes back empty (the "no staged changes" branch) instead of
reaching the model.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.agents.sorcar.persistence as th
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
    _snapshot_active_tabs,
)

_UNAVAILABLE_MODEL = "kiss-test-no-such-model"
"""Model name guaranteed to be absent from ``get_available_models()``.

Makes the real worker thread hit ``task_runner``'s "No model
available." guard and return before contacting any provider.
"""


def _find_free_port() -> int:
    """Return an available TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate checks."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _redirect_persistence(tmpdir: str) -> tuple[Any, Any, Any]:
    """Point the persistence layer at a scratch DB under *tmpdir*."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved


def _restore_persistence(saved: tuple[Any, Any, Any]) -> None:
    """Undo :func:`_redirect_persistence`."""
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _git(cwd: Path, *args: str) -> None:
    """Run a git command in *cwd*, raising on failure."""
    subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )


class _ServerHarness(IsolatedAsyncioTestCase):
    """Real ``RemoteAccessServer`` on a temp UDS + ephemeral port."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.port = _find_free_port()
        self.work_dir = Path(self.tmpdir) / "repo"
        self.work_dir.mkdir()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
            work_dir=str(self.work_dir),
        )
        await self.server.start_async()
        self._stopped = False
        self._writers: list[asyncio.StreamWriter] = []
        self._workers: list[tuple[threading.Event, threading.Thread]] = []

    async def asyncTearDown(self) -> None:
        for writer in self._writers:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
        for release, thread in self._workers:
            release.set()
            thread.join(timeout=5)
        if not self._stopped:
            await self.server.stop_async()
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open one real UDS connection (simulates one window)."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )
        self._writers.append(writer)
        return reader, writer

    async def _send(
        self, writer: asyncio.StreamWriter, cmd: dict[str, Any],
    ) -> None:
        """Write one newline-framed JSON command."""
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        predicate: Callable[[dict[str, Any]], bool],
        max_events: int = 200,
        timeout: float = 15.0,
    ) -> dict[str, Any]:
        """Read frames until *predicate* matches, else fail the test."""
        for _ in range(max_events):
            line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            assert line, "UDS closed unexpectedly"
            msg = json.loads(line.decode("utf-8"))
            assert isinstance(msg, dict)
            if predicate(msg):
                return msg
        raise AssertionError("predicate never matched")

    async def _await_dispatch(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
    ) -> None:
        """Block until every command sent so far has been dispatched.

        ``_uds_handler`` dispatches this connection's commands
        sequentially, so the reply to a trailing ``activeTasksQuery``
        cannot be produced until the preceding commands' handlers have
        returned.  (It says nothing about *broadcast* delivery, which
        is scheduled onto the loop from the executor thread — use
        :meth:`_collect_frames` for that.)
        """
        await self._send(writer, {"type": "activeTasksQuery"})
        await self._drain_until(
            reader, lambda m: m.get("type") == "activeTasksResponse",
        )

    async def _collect_frames(
        self,
        reader: asyncio.StreamReader,
        stop: Callable[[dict[str, Any]], bool],
        timeout: float = 10.0,
    ) -> list[dict[str, Any]]:
        """Read frames until *stop* matches or *timeout* elapses."""
        seen: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return seen
            try:
                line = await asyncio.wait_for(
                    reader.readline(), timeout=remaining,
                )
            except TimeoutError:
                return seen
            if not line:
                return seen
            msg = json.loads(line.decode("utf-8"))
            seen.append(msg)
            if stop(msg):
                return seen

    def _register_starting_task(
        self,
        tab_id: str,
        task_id: str,
        chat_id: str = "",
    ) -> tuple[agent_state.AgentState, threading.Event]:
        """Register a real state in the ``_cmd_run`` startup window.

        Reproduces the exact interleaving of F08-2: ``_cmd_run`` has
        registered the state and started the worker thread, but the
        worker has not yet reached ``state.is_task_active = True``.
        The worker is a real thread that honours the state's real
        ``stop_event``, so a correct shutdown sweep stops and joins it.
        """
        stop_event = threading.Event()
        exited = threading.Event()
        state = agent_state.AgentState(
            task_id,
            chat_id=chat_id or f"chat-{task_id}",
            tab_id=tab_id,
            server_owned=True,
            stop_event=stop_event,
        )
        state.last_user_prompt = f"prompt for {task_id}"

        def _worker() -> None:
            stop_event.wait(timeout=30)
            exited.set()

        thread = threading.Thread(
            target=_worker, name=f"h-worker-{task_id}", daemon=True,
        )
        state.task_thread = thread
        agent_state.register(state)
        thread.start()
        self._workers.append((stop_event, thread))
        return state, exited


class TestRunRefusedDuringMergeClearsStatus(_ServerHarness):
    """F08-1: a refused run must lower the client's running flag."""

    async def _refusal_frames(
        self, reader: asyncio.StreamReader, tab_id: str,
    ) -> list[dict[str, Any]]:
        """Collect frames until the refusal is fully delivered.

        The refusal is two events — the running flag is lowered and
        the reason is shown — so reading stops only once BOTH have
        arrived (or the bound expires, which is how a missing one is
        reported).
        """
        pending = {"status", "error"}

        def _stop(msg: dict[str, Any]) -> bool:
            if msg.get("tabId") != tab_id:
                return False
            if msg.get("type") == "status" and msg.get("running") is False:
                pending.discard("status")
            elif msg.get("type") == "error":
                pending.discard("error")
            return not pending

        return await self._collect_frames(reader, _stop, timeout=5.0)

    def _claim_merge(self, tab_id: str) -> agent_state.AgentState:
        """Take the tab's merge claim exactly as ``merge_flow`` does."""
        state = agent_state.AgentState(
            "task-under-merge",
            chat_id="chat-under-merge",
            tab_id=tab_id,
            server_owned=True,
        )
        agent_state.register(state)
        with agent_state.STATE_LOCK:
            state.is_merging = True
        return state

    async def test_submit_during_merge_emits_status_false(self) -> None:
        """The browser ``submit`` dialect must un-disable the composer."""
        tab_id = "tab-merge-submit"
        state = self._claim_merge(tab_id)
        reader, writer = await self._connect()

        await self._send(writer, {
            "type": "submit",
            "prompt": "follow-up while merging",
            "tabId": tab_id,
            "model": _UNAVAILABLE_MODEL,
        })
        frames = await self._refusal_frames(reader, tab_id)

        errors = [
            f for f in frames
            if f.get("type") == "error"
            and "merge is in progress" in str(f.get("text", ""))
        ]
        self.assertTrue(errors, f"no merge refusal error in {frames}")
        self.assertTrue(
            any(
                f.get("type") == "status"
                and f.get("running") is True
                and f.get("tabId") == tab_id
                for f in frames
            ),
            "the optimistic running:true was not observed",
        )
        cleared = [
            f for f in frames
            if f.get("type") == "status"
            and f.get("running") is False
            and f.get("tabId") == tab_id
        ]
        self.assertTrue(
            cleared,
            "BUG F08-1: the refused run never broadcast "
            "status running:false, so the tab's input stays disabled "
            f"forever; frames were {[f.get('type') for f in frames]}",
        )
        self.assertTrue(state.is_merging, "the merge claim was destroyed")

    async def test_run_during_merge_emits_status_false(self) -> None:
        """The VS Code ``run`` dialect goes through the same guard."""
        tab_id = "tab-merge-run"
        self._claim_merge(tab_id)
        reader, writer = await self._connect()

        await self._send(writer, {
            "type": "run",
            "prompt": "follow-up while merging",
            "tabId": tab_id,
            "model": _UNAVAILABLE_MODEL,
        })
        frames = await self._refusal_frames(reader, tab_id)

        self.assertTrue(
            [
                f for f in frames
                if f.get("type") == "error"
                and "merge is in progress" in str(f.get("text", ""))
            ],
            f"no merge refusal error in {frames}",
        )
        self.assertTrue(
            [
                f for f in frames
                if f.get("type") == "status"
                and f.get("running") is False
                and f.get("tabId") == tab_id
            ],
            "BUG F08-1: the VS Code host raised running:true before "
            "sending run, and the refusal never cleared it",
        )

    async def test_accepted_run_is_not_refused(self) -> None:
        """A tab with no merge claim still starts its task."""
        tab_id = "tab-no-merge"
        reader, writer = await self._connect()
        await self._send(writer, {
            "type": "run",
            "prompt": "a normal task",
            "tabId": tab_id,
            "model": _UNAVAILABLE_MODEL,
        })
        result = await self._drain_until(
            reader,
            lambda m: m.get("type") == "result" and m.get("tabId") == tab_id,
        )
        self.assertIn("No model available", str(result.get("text", "")))


class TestStartupWindowIsLive(_ServerHarness):
    """F08-2: a started-but-unflagged worker counts as a live task."""

    async def test_active_tasks_query_reports_starting_task(self) -> None:
        """``activeTasksQuery`` must not answer 0 during the window.

        The VS Code dependency installer restarts the daemon when the
        reply says ``count: 0``; answering 0 for a task whose worker
        has just been started kills it.
        """
        self._register_starting_task("tab-window", "task-window")
        reader, writer = await self._connect()

        await self._send(writer, {"type": "activeTasksQuery"})
        resp = await self._drain_until(
            reader, lambda m: m.get("type") == "activeTasksResponse",
        )
        self.assertEqual(
            resp.get("count"), 1,
            "BUG F08-2: a task whose worker started but has not yet "
            "raised is_task_active was reported as idle, so the "
            "extension would restart the daemon on top of it",
        )
        self.assertIn("tab-window(task=task-window)", resp.get("tabs", []))

    async def test_shutdown_sweep_stops_starting_task(self) -> None:
        """``stop_async`` must stop and join a task in the window.

        The embedder shutdown path sweeps exactly once, so a task
        skipped by the sweep is abandoned outright: no stop event, no
        join, no cleanup ``finally``.
        """
        state, exited = self._register_starting_task("tab-shut", "task-shut")

        await self.server.stop_async()
        self._stopped = True

        self.assertTrue(
            exited.is_set(),
            "BUG F08-2: the shutdown sweep skipped the starting task, "
            "so its worker was never signalled and its history row is "
            "stranded at the abrupt-failure sentinel",
        )
        thread = state.task_thread
        assert thread is not None
        self.assertFalse(thread.is_alive())
        self.assertTrue(state.interrupted_by_shutdown)

    async def test_shutdown_sweep_ignores_finished_states(self) -> None:
        """A state with no live worker is neither active nor swept."""
        state = agent_state.AgentState(
            "task-done", tab_id="tab-done", server_owned=True,
        )
        agent_state.register(state)
        self.assertEqual(_snapshot_active_tabs(), [])

        await self.server.stop_async()
        self._stopped = True

        self.assertFalse(state.interrupted_by_shutdown)

    async def test_ready_restores_tab_of_starting_task(self) -> None:
        """A browser connecting in the window still gets the tab.

        ``_cmd_run`` registers the tab in the shared tab registry
        synchronously, BEFORE the worker thread starts, so a client
        whose ``ready`` lands inside the startup window (worker alive,
        ``is_task_active`` not yet raised) must still receive a
        ``tabs_state`` snapshot binding the tab to its chat.
        """
        self._register_starting_task(
            "tab-restore", "task-restore", chat_id="chat-restore",
        )
        # What _cmd_run does synchronously before starting the worker:
        self.server._vscode_server._registry_update_tab(
            "tab-restore",
            chat_id="chat-restore",
            title="starting task",
            create=True,
        )
        ctx = _no_verify_ssl()
        async with connect(f"wss://127.0.0.1:{self.port}/ws", ssl=ctx) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            auth = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
            self.assertEqual(auth.get("type"), "auth_ok")

            await ws.send(json.dumps({"type": "ready", "tabId": "tab-new"}))
            rows: list[dict[str, Any]] = []
            found = False
            # The ``tabs_state`` snapshot is broadcast from a worker
            # thread (``ready_tab_sync`` via ``asyncio.to_thread``)
            # through ``run_coroutine_threadsafe``, while direct
            # replies are awaited on the event loop; the per-endpoint
            # send lock serializes only sends already in flight, so a
            # later direct reply may legitimately hit the wire before
            # the scheduled snapshot.  Read until the snapshot with the
            # running tab arrives instead of using another reply as an
            # ordering barrier.
            for _ in range(200):
                try:
                    ev = json.loads(
                        await asyncio.wait_for(ws.recv(), timeout=15),
                    )
                except TimeoutError:
                    break
                if ev.get("type") != "tabs_state":
                    continue
                rows.extend(ev.get("tabs", []))
                if any(
                    r.get("tabId") == "tab-restore"
                    and r.get("chatId") == "chat-restore"
                    for r in rows
                ):
                    found = True
                    break

        self.assertTrue(
            found,
            "BUG F08-2: a client connecting during the startup window "
            f"silently omitted the running tab; got {rows}",
        )


class TestSecondServerPreservesRegistry(_ServerHarness):
    """F08-5: constructing a server must not detach live tasks."""

    async def test_construction_keeps_live_states_and_drops_dead(
        self,
    ) -> None:
        """Only states with no live owner may be evicted."""
        live, exited = self._register_starting_task("tab-live", "task-live")
        live.is_task_active = True
        dead = agent_state.AgentState(
            "task-dead", tab_id="tab-dead", server_owned=True,
        )
        agent_state.register(dead)

        RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            url_file=Path(self.tmpdir) / "remote-url-2.json",
            uds_path=Path(self.tmpdir) / "sorcar-2.sock",
            work_dir=str(self.work_dir),
        )

        self.assertIs(
            agent_state.get("task-live"), live,
            "BUG F08-5: constructing a second server cleared the "
            "process-global registry and detached a live task",
        )
        self.assertIn("tab-live(task=task-live)", _snapshot_active_tabs())
        self.assertIsNone(
            agent_state.get("task-dead"),
            "a finished state must still be evicted by a new server",
        )

        await self.server.stop_async()
        self._stopped = True
        self.assertTrue(
            exited.is_set(),
            "BUG F08-5: the first server could no longer stop the task "
            "it owns, so the worker is killed at exit",
        )


class TestNoSharedClassLevelTabMap(unittest.TestCase):
    """F08-6: the tab→task map must not be a mutable class default."""

    def tearDown(self) -> None:
        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()

    def test_tab_opened_task_ids_is_instance_only(self) -> None:
        """Two servers must never share one tab→task dict."""
        self.assertNotIn(
            "_tab_opened_task_ids", vars(VSCodeServer),
            "BUG F08-6: the dead mutable class attribute is back; any "
            "read before __init__ finishes would silently share one "
            "dict across every server in the process",
        )
        first = VSCodeServer()
        second = VSCodeServer()
        first._tab_opened_task_ids["tab-a"] = "task-a"
        self.assertEqual(second._tab_opened_task_ids, {})
        self.assertEqual(first._tab_opened_task_ids, {"tab-a": "task-a"})


class TestSubmitRecordsConnId(_ServerHarness):
    """F08-7: ``submit`` and ``run`` must record the same connection."""

    async def _launch(
        self, writer: asyncio.StreamWriter,
        reader: asyncio.StreamReader,
        cmd_type: str,
        tab_id: str,
    ) -> agent_state.AgentState:
        """Run one task to completion and return its agent state."""
        await self._send(writer, {
            "type": cmd_type,
            "prompt": "record my connection",
            "tabId": tab_id,
            "model": _UNAVAILABLE_MODEL,
        })
        await self._drain_until(
            reader,
            lambda m: m.get("type") == "status"
            and m.get("running") is False
            and m.get("tabId") == tab_id,
        )
        state = agent_state.find_by_tab(tab_id)
        assert state is not None, f"no state registered for {tab_id}"
        return state

    async def test_submit_and_run_agree_on_conn_id(self) -> None:
        """Both dialects on one connection record that connection."""
        reader, writer = await self._connect()
        await self._send(writer, {"type": "ready", "tabId": "tab-run"})
        await self._drain_until(
            reader, lambda m: m.get("type") == "configData",
        )

        via_run = await self._launch(writer, reader, "run", "tab-run")
        via_submit = await self._launch(
            writer, reader, "submit", "tab-submit",
        )

        self.assertTrue(
            via_run.conn_id, "the run dialect lost its connection id",
        )
        self.assertEqual(
            via_submit.conn_id, via_run.conn_id,
            "BUG F08-7: _handle_submit rebuilt the run command without "
            "connId, so a browser-launched task records \"\" while an "
            "identical VS Code-launched task records the real id",
        )


class TestCommitMessageGenerationDedup(_ServerHarness):
    """R09-8: concurrent clicks must not run concurrent generations."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.calls_log = Path(self.tmpdir) / "diff-calls.log"
        self.release_file = Path(self.tmpdir) / "diff-release"
        blocker = Path(self.tmpdir) / "blocking-diff.sh"
        blocker.write_text(
            "#!/bin/sh\n"
            f'echo call >> "{self.calls_log}"\n'
            f'while [ ! -f "{self.release_file}" ]; do sleep 0.02; done\n'
            "exit 0\n",
        )
        blocker.chmod(0o755)
        _git(self.work_dir, "init", "-q")
        _git(self.work_dir, "config", "user.email", "h@example.com")
        _git(self.work_dir, "config", "user.name", "Kiss Test")
        _git(self.work_dir, "config", "diff.external", str(blocker))
        (self.work_dir / "file.txt").write_text("one\n")
        _git(self.work_dir, "add", "file.txt")
        _git(self.work_dir, "-c", "diff.external=", "commit", "-q", "-m", "init")
        (self.work_dir / "file.txt").write_text("two\n")
        _git(self.work_dir, "add", "file.txt")

    async def asyncTearDown(self) -> None:
        self.release_file.touch()
        await super().asyncTearDown()

    def _call_count(self) -> int:
        """Number of real ``git`` external-diff invocations so far."""
        try:
            return len(self.calls_log.read_text().splitlines())
        except FileNotFoundError:
            return 0

    async def _wait_for_calls(self, want: int, timeout: float) -> int:
        """Poll the invocation log until *want* calls or *timeout*."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            count = self._call_count()
            if count >= want:
                return count
            await asyncio.sleep(0.02)
        return self._call_count()

    async def test_burst_of_clicks_runs_one_generation(self) -> None:
        """Five clicks must produce one generation, then re-arm."""
        tab_id = "tab-commit"
        reader, writer = await self._connect()
        for _ in range(5):
            await self._send(writer, {
                "type": "generateCommitMessage",
                "tabId": tab_id,
                "workDir": str(self.work_dir),
            })
        # Barrier: the connection is served sequentially, so once the
        # probe is answered all five clicks have been handled (and,
        # before the fix, all five worker threads spawned).
        await self._await_dispatch(reader, writer)

        self.assertGreaterEqual(
            await self._wait_for_calls(1, timeout=15), 1,
            "the first generation never reached git",
        )
        self.assertEqual(
            await self._wait_for_calls(2, timeout=2.0), 1,
            "BUG R09-8: duplicate clicks spawned concurrent commit "
            "message generations — each one bills an LLM call and "
            "races to stamp the same tab",
        )

        self.release_file.touch()
        first = await self._drain_until(
            reader,
            lambda m: m.get("type") == "commitMessage"
            and m.get("tabId") == tab_id,
        )
        self.assertIn("No staged changes", str(first.get("error", "")))

        # The in-flight guard must be released once the generation
        # finishes, so a later click still works.
        os.remove(self.release_file)
        self.calls_log.unlink(missing_ok=True)
        await self._send(writer, {
            "type": "generateCommitMessage",
            "tabId": tab_id,
            "workDir": str(self.work_dir),
        })
        self.assertEqual(
            await self._wait_for_calls(1, timeout=15), 1,
            "the in-flight guard was never cleared, so the tab can "
            "never generate a commit message again",
        )
        self.release_file.touch()
        await self._drain_until(
            reader,
            lambda m: m.get("type") == "commitMessage"
            and m.get("tabId") == tab_id,
        )

    async def test_other_tab_is_not_blocked(self) -> None:
        """The guard is per tab, not global."""
        reader, writer = await self._connect()
        for tab_id in ("tab-one", "tab-two"):
            await self._send(writer, {
                "type": "generateCommitMessage",
                "tabId": tab_id,
                "workDir": str(self.work_dir),
            })
        await self._await_dispatch(reader, writer)
        self.assertEqual(
            await self._wait_for_calls(2, timeout=15), 2,
            "a generation on one tab must not block another tab",
        )
        self.release_file.touch()
        seen: set[str] = set()

        def _collect(msg: dict[str, Any]) -> bool:
            if msg.get("type") == "commitMessage":
                seen.add(str(msg.get("tabId", "")))
            return {"tab-one", "tab-two"} <= seen

        await self._drain_until(reader, _collect)


if __name__ == "__main__":
    unittest.main()
