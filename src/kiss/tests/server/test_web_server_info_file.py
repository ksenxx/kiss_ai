# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``getInfoFile`` command.

The remote webapp's task-info panel (docked on desktop, a drawer on
mobile) and the VS Code extension's editor-tab chat panels poll
``getInfoFile`` so the info subpanel / Task Info view can mirror
``tmp/PROGRESS.md`` under the active tab's work dir.  These tests
drive the REAL production paths: a live :class:`RemoteAccessServer`
over WSS for the read / signature / fallback behaviors, and a real
Unix-domain-socket connection for the direct reply the VS Code
extension's forwarded polls receive on the same UDS connection.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import ssl
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server.web_server import _OPEN_FILE_MAX_BYTES, RemoteAccessServer


def _find_free_port() -> int:
    """Find an available TCP port."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        port: int = s.getsockname()[1]
        return port


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestGetInfoFileOverWss(IsolatedAsyncioTestCase):
    """``getInfoFile`` over a live WSS connection."""

    async def asyncSetUp(self) -> None:
        import kiss.agents.sorcar.persistence as _persistence

        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        self._persistence_dir = Path(
            tempfile.mkdtemp(prefix="kiss_infofile_test_")
        )
        _persistence._KISS_DIR = self._persistence_dir
        _persistence._DB_PATH = self._persistence_dir / "sorcar.db"
        _persistence._db_conn = None

        self.port = _find_free_port()
        self._orig_config = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": ""})

        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            work_dir=tempfile.mkdtemp(),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

        import kiss.agents.sorcar.persistence as _persistence

        if _persistence._db_conn is not None:
            try:
                _persistence._db_conn.close()
            except Exception:
                pass
            _persistence._db_conn = None
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence

    async def _get_info_file(
        self, ws: Any, cmd_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Send one ``getInfoFile`` and return its ``infoFile`` reply."""
        await ws.send(json.dumps({"type": "getInfoFile", **cmd_fields}))
        deadline = asyncio.get_event_loop().time() + 5
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            ev: dict[str, Any] = json.loads(raw)
            if ev.get("type") == "infoFile":
                return ev
        raise AssertionError("no infoFile reply received")

    async def test_missing_file_replies_empty(self) -> None:
        """A workdir without tmp/PROGRESS.md replies exists=false, no text."""
        work_dir = self.server.work_dir
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-1"}
            )
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["content"], "")
        self.assertEqual(reply["sig"], "")
        self.assertEqual(reply["workDir"], work_dir)
        self.assertEqual(reply["tabId"], "t-1")
        self.assertNotIn("unchanged", reply)

    async def test_existing_file_replies_content_and_sig(self) -> None:
        """An existing tmp/PROGRESS.md is sent back with a non-empty sig."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("# Status\n\nAll good.\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-2", "token": "tok-7"}
            )
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "# Status\n\nAll good.\n")
        self.assertNotEqual(reply["sig"], "")
        st = info.stat()
        self.assertEqual(
            reply["sig"], f"{info}:{st.st_mtime_ns}:{st.st_size}"
        )
        self.assertEqual(reply["token"], "tok-7")

    async def test_matching_known_sig_replies_unchanged(self) -> None:
        """A poll whose knownSig matches skips the content re-send."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("stable\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            first = await self._get_info_file(ws, {"workDir": work_dir})
            second = await self._get_info_file(
                ws, {"workDir": work_dir, "knownSig": first["sig"]}
            )
        self.assertEqual(first["content"], "stable\n")
        self.assertIs(second["unchanged"], True)
        self.assertIs(second["exists"], True)
        self.assertEqual(second["sig"], first["sig"])
        self.assertNotIn("content", second)

    async def test_changed_file_replies_new_content(self) -> None:
        """A stale knownSig gets the rewritten file and a new sig."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("before\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            first = await self._get_info_file(ws, {"workDir": work_dir})
            info.write_text("after: longer content\n")
            second = await self._get_info_file(
                ws, {"workDir": work_dir, "knownSig": first["sig"]}
            )
        self.assertEqual(first["content"], "before\n")
        self.assertEqual(second["content"], "after: longer content\n")
        self.assertNotEqual(second["sig"], first["sig"])
        self.assertNotIn("unchanged", second)

    async def test_deleted_file_replies_empty_again(self) -> None:
        """Deleting the file flips the reply back to exists=false."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("soon gone\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            first = await self._get_info_file(ws, {"workDir": work_dir})
            info.unlink()
            second = await self._get_info_file(
                ws, {"workDir": work_dir, "knownSig": first["sig"]}
            )
        self.assertIs(second["exists"], False)
        self.assertEqual(second["content"], "")
        self.assertEqual(second["sig"], "")

    async def test_empty_workdir_falls_back_to_daemon_dir(self) -> None:
        """No workDir resolves tmp/PROGRESS.md under the daemon work dir."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("fallback\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {})
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "fallback\n")
        self.assertEqual(reply["workDir"], "")

    async def test_worktree_copy_wins_while_task_runs_there(self) -> None:
        """A tab with a recorded worktree reads the worktree's copy.

        A worktree-mode task maintains its ``tmp/PROGRESS.md`` inside
        the worktree, not the tab's workDir, so once the tab's
        ``worktree_created`` event recorded the worktree dir the poll
        must serve the worktree's file — and switching sources must
        change the sig even for byte-identical contents (the sig is
        path-prefixed), so the client repaints.
        """
        work_dir = self.server.work_dir
        main_copy = Path(work_dir) / "tmp" / "PROGRESS.md"
        main_copy.parent.mkdir(parents=True)
        main_copy.write_text("main tree\n")
        wt_dir = tempfile.mkdtemp(prefix="kiss_wt_test_")
        wt_copy = Path(wt_dir) / "tmp" / "PROGRESS.md"
        wt_copy.parent.mkdir(parents=True)
        wt_copy.write_text("worktree progress\n")
        # The production recording path: broadcast tracking of the
        # tab's worktree_created event.
        self.server._printer._track_worktree_event(
            {"type": "worktree_created", "worktreeWorkDir": wt_dir},
            "t-wt",
            None,
        )
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            wt_reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-wt"}
            )
            plain_reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-other"}
            )
        self.assertIs(wt_reply["exists"], True)
        self.assertEqual(wt_reply["content"], "worktree progress\n")
        self.assertTrue(wt_reply["sig"].startswith(str(wt_copy) + ":"))
        self.assertIs(plain_reply["exists"], True)
        self.assertEqual(plain_reply["content"], "main tree\n")
        self.assertNotEqual(plain_reply["sig"], wt_reply["sig"])

    async def test_missing_worktree_copy_falls_back_to_workdir(self) -> None:
        """A recorded worktree without the file falls back to workDir."""
        work_dir = self.server.work_dir
        main_copy = Path(work_dir) / "tmp" / "PROGRESS.md"
        main_copy.parent.mkdir(parents=True)
        main_copy.write_text("only in main\n")
        wt_dir = tempfile.mkdtemp(prefix="kiss_wt_test_")
        self.server._printer._track_worktree_event(
            {"type": "worktree_created", "worktreeWorkDir": wt_dir},
            "t-wt2",
            None,
        )
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-wt2"}
            )
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "only in main\n")
        self.assertTrue(reply["sig"].startswith(str(main_copy) + ":"))

    def _register_task(
        self,
        task_id: str,
        tab_id: str,
        work_dir: str,
        start_ms: int,
        *,
        active: bool = True,
    ) -> Any:
        """Register a real agent state as the task-runner does for a run.

        Mirrors ``_TaskRunnerMixin._run_task_inner``: a
        :class:`WorktreeSorcarAgent` stamped with the run's
        ``_task_start_ms`` and (as ``RelentlessAgent._reset`` does once
        the run begins) its effective ``work_dir``, installed in the
        agent-state registry under the launching tab.  The state is
        unregistered on test teardown.
        """
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
        from kiss.server import agent_state

        agent = WorktreeSorcarAgent(f"info-file-test {task_id}")
        if work_dir:
            agent.work_dir = work_dir
        agent._task_start_ms = start_ms
        state = agent_state.AgentState(
            task_id,
            agent=agent,
            tab_id=tab_id,
            server_owned=True,
            is_task_active=active,
        )
        agent_state.register(state)
        self.addCleanup(agent_state.unregister, task_id, state)
        return state

    @staticmethod
    def _write_aged(path: Path, text: str, age_seconds: float) -> None:
        """Write *text* to *path* and back-date its mtime by *age_seconds*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        old = time.time() - age_seconds
        os.utime(path, (old, old))

    async def test_previous_tasks_file_is_hidden_until_task_rewrites_it(
        self,
    ) -> None:
        """A tmp/PROGRESS.md older than the running task is not shown.

        The reported bug: the subpanel mirrored the PROGRESS.md a
        PREVIOUS task left in the directory until the running task
        overwrote it.  A file whose mtime predates the running task's
        start is treated as missing; once the task rewrites it, the
        new contents are served.
        """
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        self._write_aged(info, "# previous task\n", age_seconds=3600)
        self._register_task(
            "task-cur", "t-run", work_dir, int(time.time() * 1000),
        )
        # As in production, the launching tab is also subscribed to its
        # own task's stream (register_task_ui), and the subscriber set
        # of its previous, already torn-down task may still linger.
        self.server._printer.subscribe_tab("task-cur", "t-run")
        self.server._printer.subscribe_tab("task-gone", "t-run")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            stale = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-run"}
            )
            info.write_text("# current task\n\nstep 1\n")
            fresh = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-run"}
            )
            # A stale knownSig from the fresh read still short-circuits.
            again = await self._get_info_file(
                ws,
                {"workDir": work_dir, "tabId": "t-run", "knownSig": fresh["sig"]},
            )
        self.assertIs(stale["exists"], False)
        self.assertEqual(stale["content"], "")
        self.assertEqual(stale["sig"], "")
        self.assertIs(fresh["exists"], True)
        self.assertEqual(fresh["content"], "# current task\n\nstep 1\n")
        self.assertIs(again["unchanged"], True)

    async def test_file_written_just_before_start_tolerates_clock_skew(
        self,
    ) -> None:
        """An mtime within the 1s tolerance before the start is accepted."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        self._write_aged(info, "coarse clock\n", age_seconds=0.5)
        self._register_task(
            "task-skew", "t-skew", work_dir, int(time.time() * 1000),
        )
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-skew"}
            )
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "coarse clock\n")

    async def test_running_tasks_work_dir_is_the_only_source(self) -> None:
        """The running task's effective work dir wins over every fallback.

        A worktree-mode run's ``work_dir`` is its worktree; the tab's
        workDir (the main checkout) and even a stale recorded worktree
        hold OTHER tasks' files and must never be served — not even
        when the running task has not written its own copy yet.
        """
        work_dir = self.server.work_dir
        main_copy = Path(work_dir) / "tmp" / "PROGRESS.md"
        main_copy.parent.mkdir(parents=True)
        main_copy.write_text("main checkout, other task\n")
        old_wt = tempfile.mkdtemp(prefix="kiss_wt_old_")
        (Path(old_wt) / "tmp").mkdir()
        (Path(old_wt) / "tmp" / "PROGRESS.md").write_text("old worktree\n")
        self.server._printer._track_worktree_event(
            {"type": "worktree_created", "worktreeWorkDir": old_wt},
            "t-wt3",
            None,
        )
        task_wt = tempfile.mkdtemp(prefix="kiss_wt_cur_")
        self._register_task(
            "task-wt3", "t-wt3", task_wt, int(time.time() * 1000),
        )
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            nothing_yet = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-wt3"}
            )
            own = Path(task_wt) / "tmp" / "PROGRESS.md"
            own.parent.mkdir()
            own.write_text("running task's progress\n")
            served = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-wt3"}
            )
        self.assertIs(nothing_yet["exists"], False)
        self.assertEqual(nothing_yet["content"], "")
        self.assertIs(served["exists"], True)
        self.assertEqual(served["content"], "running task's progress\n")
        self.assertTrue(served["sig"].startswith(str(own) + ":"))

    async def test_viewer_tab_resolves_task_through_subscription(
        self,
    ) -> None:
        """A tab viewing a task launched elsewhere reads THAT task's file.

        The same chat opened from another client is subscribed to the
        running task's stream (``JsonPrinter.subscribe_tab``) without
        owning an agent state; the poll must still find the task via
        the subscription.  A viewer tab whose OWN previous task is idle
        (its state lingers for the pending worktree) must prefer the
        active task it is watching.
        """
        work_dir = self.server.work_dir
        task_dir = tempfile.mkdtemp(prefix="kiss_task_dir_")
        (Path(task_dir) / "tmp").mkdir()
        (Path(task_dir) / "tmp" / "PROGRESS.md").write_text("watched task\n")
        self._register_task(
            "task-live", "t-launcher", task_dir, int(time.time() * 1000),
        )
        # The viewer's own finished task (idle, lingering state) wrote
        # a fresh-looking file of its own that must NOT be shown.
        idle_dir = tempfile.mkdtemp(prefix="kiss_idle_dir_")
        (Path(idle_dir) / "tmp").mkdir()
        (Path(idle_dir) / "tmp" / "PROGRESS.md").write_text("idle own task\n")
        self._register_task(
            "task-idle", "t-viewer", idle_dir, int(time.time() * 1000) - 5000,
            active=False,
        )
        self.server._printer.subscribe_tab("task-live", "t-viewer")
        self.server._printer.subscribe_tab("task-live", "t-pure-viewer")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            viewer = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-viewer"}
            )
            pure = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-pure-viewer"}
            )
        self.assertEqual(viewer["content"], "watched task\n")
        self.assertEqual(pure["content"], "watched task\n")

    async def test_idle_state_without_active_task_still_serves_its_dir(
        self,
    ) -> None:
        """With no active task, the tab's lingering state names the dir.

        Right after a task ends (before the client's status flip stops
        the poll) the state is inactive but still the tab's only
        task: its work dir is served rather than the workDir fallback.
        """
        work_dir = self.server.work_dir
        main_copy = Path(work_dir) / "tmp" / "PROGRESS.md"
        main_copy.parent.mkdir(parents=True)
        main_copy.write_text("main\n")
        done_dir = tempfile.mkdtemp(prefix="kiss_done_dir_")
        (Path(done_dir) / "tmp").mkdir()
        (Path(done_dir) / "tmp" / "PROGRESS.md").write_text("just finished\n")
        self._register_task(
            "task-done", "t-done", done_dir, int(time.time() * 1000) - 60000,
            active=False,
        )
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-done"}
            )
        self.assertEqual(reply["content"], "just finished\n")

    async def test_agent_without_start_stamp_is_not_gated(self) -> None:
        """A task agent with no start timestamp serves its dir ungated.

        Sub-agents (``ChatSorcarAgent`` children of ``run_parallel``)
        carry no ``_task_start_ms``; their work dir is the parent's,
        whose PROGRESS.md predates the child — it must still show.
        """
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        self._write_aged(info, "parent progress\n", age_seconds=3600)
        self._register_task("task-sub", "t-sub", work_dir, 0)
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-sub"}
            )
        self.assertEqual(reply["content"], "parent progress\n")

    async def test_agent_without_work_dir_falls_back_to_tab_dirs(self) -> None:
        """Before ``_reset`` the agent has no work dir: tab dirs, gated.

        During task classification / worktree setup the fresh agent has
        no ``work_dir`` yet (``_reset`` assigns it), so the recorded
        worktree / workDir candidates apply — still gated by the start
        stamp, so the previous task's file in the main checkout stays
        hidden.
        """
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        self._write_aged(info, "previous task\n", age_seconds=3600)
        self._register_task(
            "task-early", "t-early", "", int(time.time() * 1000),
        )
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            hidden = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-early"}
            )
            info.write_text("written by the running task\n")
            shown = await self._get_info_file(
                ws, {"workDir": work_dir, "tabId": "t-early"}
            )
        self.assertIs(hidden["exists"], False)
        self.assertEqual(shown["content"], "written by the running task\n")

    async def test_directory_named_progress_md_replies_empty(self) -> None:
        """tmp/PROGRESS.md that is a directory is treated as missing."""
        work_dir = self.server.work_dir
        (Path(work_dir) / "tmp" / "PROGRESS.md").mkdir(parents=True)
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {"workDir": work_dir})
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["content"], "")
        self.assertEqual(reply["sig"], "")

    async def test_fifo_named_progress_md_replies_empty_without_hanging(
        self,
    ) -> None:
        """A FIFO planted at tmp/PROGRESS.md is rejected, not read.

        The handler opens the path with ``O_NONBLOCK`` and checks the
        descriptor's ``fstat`` for a regular file, so a FIFO can
        neither hang the worker thread on open nor be streamed as
        content.
        """
        work_dir = self.server.work_dir
        fifo = Path(work_dir) / "tmp" / "PROGRESS.md"
        fifo.parent.mkdir(parents=True)
        os.mkfifo(fifo)
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {"workDir": work_dir})
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["content"], "")
        self.assertEqual(reply["sig"], "")

    async def test_oversized_file_replies_empty(self) -> None:
        """A file above _OPEN_FILE_MAX_BYTES is treated as missing."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        with info.open("wb") as f:
            f.truncate(_OPEN_FILE_MAX_BYTES + 1)
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {"workDir": work_dir})
        self.assertIs(reply["exists"], False)
        self.assertEqual(reply["content"], "")

    async def test_non_utf8_bytes_are_replaced_not_fatal(self) -> None:
        """Undecodable bytes degrade to U+FFFD instead of an error."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_bytes(b"ok \xff\xfe bytes\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(ws, {"workDir": work_dir})
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "ok \ufffd\ufffd bytes\n")

    async def test_malformed_fields_are_blanked(self) -> None:
        """Non-string workDir / tabId / knownSig echo back as ''."""
        work_dir = self.server.work_dir
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("typed\n")
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl()
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await asyncio.wait_for(ws.recv(), timeout=5)
            reply = await self._get_info_file(
                ws, {"workDir": 123, "tabId": ["x"], "knownSig": 7, "token": 5}
            )
        # The non-string workDir falls back to the daemon work dir,
        # where the file exists; the echoes are blanked.
        self.assertIs(reply["exists"], True)
        self.assertEqual(reply["content"], "typed\n")
        self.assertEqual(reply["workDir"], "")
        self.assertEqual(reply["tabId"], "")
        self.assertEqual(reply["token"], "")


class TestGetInfoFileOverUds(unittest.TestCase):
    """A UDS-delivered ``getInfoFile`` gets a direct ``infoFile`` reply.

    Editor-tab chat panels of the VS Code extension (UDS clients) poll
    ``getInfoFile`` to fill the secondary sidebar's Task Info view, so
    the command is served on both transports and the reply must come
    back on the requesting UDS connection.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.sock_path = os.path.join(self.tmp.name, "sorcar-test.sock")
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True
        )
        self.loop_thread.start()
        self.server = RemoteAccessServer(
            uds_path=self.sock_path,
            url_file=os.path.join(self.tmp.name, "remote-url.json"),
        )
        self.server._printer._loop = self.loop
        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path
            ),
            self.loop,
        ).result(timeout=5)

    def tearDown(self) -> None:
        async def _shutdown() -> None:
            self.uds_server.close()
            await self.uds_server.wait_closed()

        concurrent.futures.wait(
            [asyncio.run_coroutine_threadsafe(_shutdown(), self.loop)],
            timeout=5,
        )
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        self.tmp.cleanup()

    def test_uds_get_info_file_gets_direct_reply(self) -> None:
        """A UDS getInfoFile is answered with a direct ``infoFile``.

        A chat editor panel polls under its root tab id — a tab the
        registry may not know (no task ran yet) — so the reply resolves
        through the workDir fallback and must land on the REQUESTING
        UDS connection, echoing the poll's tab id and token.
        """
        work_dir = os.path.join(self.tmp.name, "wd")
        info = Path(work_dir) / "tmp" / "PROGRESS.md"
        info.parent.mkdir(parents=True)
        info.write_text("editor panel progress\n")

        async def _talk() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(
                self.sock_path
            )
            try:
                writer.write(
                    json.dumps(
                        {
                            "type": "getInfoFile",
                            "workDir": work_dir,
                            "tabId": "editor-panel-tab",
                            "knownSig": "",
                            "token": "9",
                        }
                    ).encode()
                    + b"\n"
                )
                await writer.drain()
                line = await asyncio.wait_for(reader.readline(), timeout=10)
                event: dict[str, Any] = json.loads(line)
                return event
            finally:
                writer.close()
                await writer.wait_closed()

        event = asyncio.run_coroutine_threadsafe(_talk(), self.loop).result(
            timeout=15
        )
        self.assertEqual(event.get("type"), "infoFile")
        self.assertIs(event.get("exists"), True)
        self.assertEqual(event.get("content"), "editor panel progress\n")
        self.assertEqual(event.get("tabId"), "editor-panel-tab")
        self.assertEqual(event.get("token"), "9")


if __name__ == "__main__":
    unittest.main()
