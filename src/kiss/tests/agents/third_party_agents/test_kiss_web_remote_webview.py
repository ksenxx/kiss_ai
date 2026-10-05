# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a third-party agent launched via ``kiss.server.sorcar.run`` is
visible and interactable from a remote webview.

Wires up a real :class:`RemoteAccessServer` with a temporary local
endpoint (the same WSS transport the production ``kiss-web`` daemon
serves to browser/VS Code webviews), launches a
``third_party_agents.slack_sea.SlackAgent`` through
``run_agent_via_kiss_web`` (i.e. through the ``kiss.server.sorcar.run``
API against that daemon), and asserts:

1. a remote webview connection to the SAME daemon receives the task's
   live events (``clear`` / ``status running=True`` / ``prompt``)
   stamped with the API launch's tab id — i.e. the agent task can be
   *opened* remotely; and
2. an ``appendUserMessage`` command sent from the webview lands in the
   running tab's ``pending_user_messages`` queue and is echoed back as
   a ``prompt`` event — i.e. the agent can be *interacted with*
   remotely.

The LLM-driving ``RelentlessAgent.run`` is stubbed (returns canned
YAML after the test releases it) so the full server/transport path is
exercised without paid API calls.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, cast

import yaml

from kiss.agents.sorcar import channel_workspace, local_endpoint
from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.third_party_agents._kiss_web_launcher import (
    run_agent_via_kiss_web,
)
from kiss.core import vscode_config
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.local_ws import LocalReader, LocalWriter, open_local_connection

STUB_SUMMARY = "remote webview stub done"


class TestRemoteWebviewInteraction(unittest.TestCase):
    """Third-party agent tasks are open/interactable via remote webview.

    The daemon under test is reached over its local WSS channel (the
    launcher's private endpoint file).
    """

    def setUp(self) -> None:
        # Every global mutation registers its restoration with
        # ``addCleanup`` immediately: cleanups run (in LIFO order) even
        # when ``setUp`` itself fails partway, unlike ``tearDown``.
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-tp-webview-")
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.endpoint_file = str(Path(self.tmpdir) / "sorcar-local.json")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", "-q"], cwd=self.repo,
            capture_output=True, check=False, timeout=60,
        )

        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "history.db"
        _persistence._db_conn = None
        self.addCleanup(self._restore_persistence)
        self._saved_config_override = (
            vars(vscode_config).get("CONFIG_DIR"),
            vars(vscode_config).get("CONFIG_PATH"),
        )
        vscode_config.CONFIG_DIR = kiss_dir
        vscode_config.CONFIG_PATH = kiss_dir / "config.json"
        self.addCleanup(self._restore_vscode_config)

        self.addCleanup(self._join_tasks_and_clear_agents)
        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()
        self.addCleanup(self._stop_loop)

        self.server = RemoteAccessServer(
            local_endpoint_file=self.endpoint_file, work_dir=self.repo,
        )

        self._viewer_writer: LocalWriter | None = None
        self._reader_task: concurrent.futures.Future[None] | None = None
        asyncio.run_coroutine_threadsafe(
            self.server.start_private_async(), self.loop,
        ).result(timeout=30)
        self.addCleanup(self._shutdown_server)

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run
        self.addCleanup(self._restore_run)

    def _restore_run(self) -> None:
        self._parent_class.run = self._original_run

    def _shutdown_server(self) -> None:
        async def _shutdown() -> None:
            try:
                if self._viewer_writer is not None:
                    self._viewer_writer.close()
                    await self._viewer_writer.wait_closed()
            except Exception:
                pass
            ws_server = self.server._ws_server
            if ws_server is not None:
                ws_server.close()
                await ws_server.wait_closed()
            local_endpoint.remove_endpoint_if_owned(
                self.server._local_endpoint_file, self.server._local_token,
            )
            pending = [
                t for t in asyncio.all_tasks()
                if t is not asyncio.current_task()
            ]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        try:
            asyncio.run_coroutine_threadsafe(
                _shutdown(), self.loop,
            ).result(timeout=5)
        except Exception:
            pass

    def _stop_loop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()

    def _join_tasks_and_clear_agents(self) -> None:
        # The daemon answers the launcher before its task thread has
        # finished its bookkeeping on the test's history.db; join those
        # threads before the DB is closed and the tmpdir removed.
        for state in agent_state.snapshot():
            if state.task_thread is not None:
                state.task_thread.join(timeout=30)
        agent_state.agent_states.clear()

    def _restore_persistence(self) -> None:
        # ``_close_db()`` stops the event writer and invalidates every
        # thread's cached connection; a raw ``close()`` under another
        # thread's running ``db.execute`` crashes the process (SIGSEGV).
        _persistence._close_db()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence

    def _restore_vscode_config(self) -> None:
        saved_dir, saved_path = self._saved_config_override
        if saved_dir is None:
            if "CONFIG_DIR" in vars(vscode_config):
                delattr(vscode_config, "CONFIG_DIR")
        else:
            vscode_config.CONFIG_DIR = saved_dir
        if saved_path is None:
            if "CONFIG_PATH" in vars(vscode_config):
                delattr(vscode_config, "CONFIG_PATH")
        else:
            vscode_config.CONFIG_PATH = saved_path

    def _open_viewer(self) -> tuple[
        LocalWriter, list[dict[str, Any]], threading.Event,
    ]:
        """Open a remote-webview local connection and drain its inbox."""

        async def _open() -> tuple[
            LocalReader, LocalWriter,
        ]:
            return await open_local_connection(self.server)

        reader, writer = asyncio.run_coroutine_threadsafe(
            _open(), self.loop,
        ).result(timeout=5)
        self._viewer_writer = writer

        received: list[dict[str, Any]] = []
        got = threading.Event()

        async def _drain() -> None:
            while True:
                line = await reader.readline()
                if not line:
                    return
                try:
                    received.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                got.set()

        self._reader_task = asyncio.run_coroutine_threadsafe(
            _drain(), self.loop,
        )
        return writer, received, got

    def _wait_for_local_client(
        self, expected_count: int, timeout: float = 5.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._printer._ws_lock:
                if len(
                    self.server._printer._local_clients
                ) >= expected_count:
                    return
            time.sleep(0.02)
        raise AssertionError("local viewer connection never registered")

    def _send_from_viewer(self, cmd: dict[str, Any]) -> None:
        """Send a JSON command over the viewer's local connection."""
        writer = self._viewer_writer
        assert writer is not None

        async def _send() -> None:
            writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
            await writer.drain()

        asyncio.run_coroutine_threadsafe(_send(), self.loop).result(
            timeout=5,
        )

    @staticmethod
    def _events_for_tab(
        received: list[dict[str, Any]], tab_id: str, ev_type: str,
    ) -> list[dict[str, Any]]:
        return [
            e for e in list(received)
            if e.get("type") == ev_type and e.get("tabId") == tab_id
        ]

    def test_launched_agent_open_and_interact_via_remote_webview(
        self,
    ) -> None:
        from kiss.agents.third_party_agents.slack.slack_sea import SlackAgent

        release = threading.Event()
        started = threading.Event()
        drained_messages: list[str] = []

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            started.set()
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            assert printer is not None, "the launched agent has no printer"
            # Drain through the printer bridge a real agent's pre-step
            # hook uses.  It CLEARS the queue: a message merely peeked
            # at would still be pending when this run ends, and the
            # task runner re-submits undrained prompts as the tab's
            # next run -- a real-model run on the slack SEA that
            # outlives the test (seen blocked in ask_user_question on
            # Windows) and keeps the ``default`` channel workspace held
            # for every later test in the process.
            deadline = time.time() + 30
            while time.time() < deadline and not release.is_set():
                queued = printer.drain_pending_user_messages()
                if queued:
                    drained_messages.extend(queued)
                    release.set()
                    break
                time.sleep(0.05)
            release.wait(timeout=30)
            raw: str = yaml.safe_dump(
                {
                    "success": True,
                    "is_continue": False,
                    "summary": STUB_SUMMARY,
                },
                sort_keys=False,
            )
            printer.print(
                raw,
                type="result",
                step_count=1,
                total_tokens=10,
                cost="$0.0010",
            )
            return raw

        self._parent_class.run = stub_run

        _writer, received, _got = self._open_viewer()
        self._wait_for_local_client(1)

        agent = SlackAgent()
        out: dict[str, Any] = {}

        def launch() -> None:
            out["result"] = run_agent_via_kiss_web(
                agent,
                "remote webview task",
                work_dir=self.repo,
                endpoint_file=self.endpoint_file,
            )

        t = threading.Thread(target=launch, daemon=True)
        t.start()
        try:
            assert started.wait(timeout=30), "agent run never started"

            tab_id = ""
            deadline = time.time() + 10
            worker: threading.Thread | None = None
            while time.time() < deadline and not tab_id:
                for st in agent_state.snapshot():
                    if st.tab_id.startswith("api-") and st.is_task_active:
                        tab_id, worker = st.tab_id, st.task_thread
                        break
                time.sleep(0.02)
            assert tab_id, "API launch never appeared in the registry"
            assert worker is not None, "the API launch has no task thread"

            deadline = time.time() + 10
            while time.time() < deadline:
                if self._events_for_tab(received, tab_id, "status"):
                    break
                time.sleep(0.05)
            status_events = self._events_for_tab(
                received, tab_id, "status",
            )
            assert any(
                e.get("running") is True for e in status_events
            ), "remote webview never saw status running=True for the task"
            assert self._events_for_tab(received, tab_id, "clear"), (
                "remote webview never saw the task's clear event"
            )
            prompt_events = self._events_for_tab(
                received, tab_id, "prompt",
            )
            assert any(
                "remote webview task" in str(e.get("text", ""))
                for e in prompt_events
            ), "remote webview never saw the task's prompt event"

            self._send_from_viewer({
                "type": "appendUserMessage",
                "tabId": tab_id,
                "prompt": "follow-up from the webview",
            })

            deadline = time.time() + 10
            while time.time() < deadline and not drained_messages:
                time.sleep(0.05)
            assert drained_messages == ["follow-up from the webview"], (
                "appendUserMessage from the remote webview never reached "
                "the running third-party agent's message queue"
            )

            echoes: list[dict[str, Any]] = []
            deadline = time.time() + 10
            while time.time() < deadline:
                echoes = [
                    e for e in self._events_for_tab(
                        received, tab_id, "prompt",
                    )
                    if "follow-up from the webview" in str(
                        e.get("text", ""),
                    )
                ]
                if echoes:
                    break
                time.sleep(0.05)
            assert echoes, (
                "the webview never received the prompt echo for its "
                "appendUserMessage"
            )
        finally:
            release.set()
            t.join(timeout=30)

        parsed = yaml.safe_load(out.get("result") or "")
        assert parsed and parsed.get("success") is True
        assert parsed.get("summary") == STUB_SUMMARY
        ended: list[dict[str, Any]] = []
        deadline = time.time() + 10
        while time.time() < deadline:
            ended = [
                e for e in self._events_for_tab(received, tab_id, "status")
                if e.get("running") is False
            ]
            if ended:
                break
            time.sleep(0.05)
        assert ended, "remote webview never saw status running=False"
        # The drained follow-up must not come back as a second run on
        # the tab.  The task runner re-submits leftovers as the LAST
        # step of the worker thread's cleanup (after ``running=False``),
        # so wait for that thread itself; a re-submitted run would then
        # show as a second ``clear`` event, a fresh task thread and the
        # slack SEA's ``default`` channel workspace held again.
        worker.join(timeout=30)
        assert not worker.is_alive(), "the launched run's task thread did not finish"
        time.sleep(1.0)  # a re-dispatched run would have started by now
        assert len(self._events_for_tab(received, tab_id, "clear")) == 1, (
            "an undrained follow-up was re-submitted as a new run"
        )
        assert dict(channel_workspace._ACTIVE_WORKSPACES) == {}
        assert not any(
            st.task_thread is not None for st in agent_state.snapshot()
        ), "a task thread outlived the launched run"


if __name__ == "__main__":
    unittest.main()
