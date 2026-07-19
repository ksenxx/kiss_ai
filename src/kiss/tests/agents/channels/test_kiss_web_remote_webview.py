# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a third-party agent launched via ``kiss.server.sorcar.run`` is
visible and interactable from a remote webview.

Wires up a real :class:`RemoteAccessServer` on a temporary UDS path
(the same transport the production ``kiss-web`` daemon serves to
browser/VS Code webviews), launches a
``third_party_agents.slack_agent.SlackAgent`` through
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

from kiss.agents.sorcar import persistence as _persistence
from kiss.agents.sorcar.running_agent_state import _RunningAgentState
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.third_party_agents._kiss_web_launcher import (
    run_agent_via_kiss_web,
)
from kiss.core import vscode_config
from kiss.server.web_server import RemoteAccessServer

STUB_SUMMARY = "remote webview stub done"


class TestRemoteWebviewInteraction(unittest.TestCase):
    """Third-party agent tasks are open/interactable via remote webview."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-tp-webview-")
        self.sock_path = str(Path(self.tmpdir) / "sorcar.sock")
        self.repo = str(Path(self.tmpdir) / "repo")
        Path(self.repo).mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "init", "-q"], cwd=self.repo,
            capture_output=True, check=False,
        )

        kiss_dir = Path(self.tmpdir) / ".kiss"
        kiss_dir.mkdir(parents=True, exist_ok=True)
        self._saved_persistence = (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        )
        _persistence._KISS_DIR = kiss_dir
        _persistence._DB_PATH = kiss_dir / "sorcar.db"
        _persistence._db_conn = None
        self._saved_config_override = (
            vars(vscode_config).get("CONFIG_DIR"),
            vars(vscode_config).get("CONFIG_PATH"),
        )
        vscode_config.CONFIG_DIR = kiss_dir
        vscode_config.CONFIG_PATH = kiss_dir / "config.json"

        self.loop = asyncio.new_event_loop()
        self.loop_thread = threading.Thread(
            target=self.loop.run_forever, daemon=True,
        )
        self.loop_thread.start()

        self.server = RemoteAccessServer(
            uds_path=self.sock_path, work_dir=self.repo,
        )
        self.server._printer._loop = self.loop
        # ``_run_cmd`` (used by the UDS dispatcher for webview
        # commands) requires the server loop.
        self.server._loop = self.loop

        self.uds_server: asyncio.Server = asyncio.run_coroutine_threadsafe(
            asyncio.start_unix_server(
                self.server._uds_handler, path=self.sock_path,
            ),
            self.loop,
        ).result(timeout=5)

        self._viewer_writer: asyncio.StreamWriter | None = None
        self._reader_task: concurrent.futures.Future[None] | None = None

        self._parent_class = cast(Any, SorcarAgent.__mro__[1])
        self._original_run = self._parent_class.run

    def tearDown(self) -> None:
        self._parent_class.run = self._original_run

        async def _shutdown() -> None:
            try:
                if self._viewer_writer is not None:
                    self._viewer_writer.close()
                    await self._viewer_writer.wait_closed()
            except Exception:
                pass
            with self.server._printer._ws_lock:
                writers = list(self.server._printer._uds_writers)
            for writer in writers:
                try:
                    writer.close()
                except Exception:
                    pass
            self.uds_server.close()
            await self.uds_server.wait_closed()
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
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop_thread.join(timeout=5)
        self.loop.close()
        _RunningAgentState.running_agent_states.clear()

        if _persistence._db_conn is not None:
            _persistence._db_conn.close()
        (
            _persistence._DB_PATH,
            _persistence._db_conn,
            _persistence._KISS_DIR,
        ) = self._saved_persistence
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
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_viewer(self) -> tuple[
        asyncio.StreamWriter, list[dict[str, Any]], threading.Event,
    ]:
        """Open a remote-webview UDS connection and drain its inbox."""

        async def _open() -> tuple[
            asyncio.StreamReader, asyncio.StreamWriter,
        ]:
            return await asyncio.open_unix_connection(
                self.sock_path, limit=16 * 1024 * 1024,
            )

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

    def _wait_for_uds_writer(
        self, expected_count: int, timeout: float = 5.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.server._printer._ws_lock:
                if len(
                    self.server._printer._uds_writers
                ) >= expected_count:
                    return
            time.sleep(0.02)
        raise AssertionError("UDS viewer connection never registered")

    def _send_from_viewer(self, cmd: dict[str, Any]) -> None:
        """Send a JSON command over the viewer's UDS connection."""
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
        from kiss.agents.third_party_agents.slack_agent import SlackAgent

        release = threading.Event()
        started = threading.Event()
        drained_messages: list[str] = []

        def stub_run(self_agent: Any, **kwargs: Any) -> str:
            started.set()
            # Poll the tab's pending_user_messages like the real
            # pre-step hook would, so the interaction assertion sees
            # the message from the agent's side of the queue.
            deadline = time.time() + 30
            while time.time() < deadline and not release.is_set():
                tab_id = getattr(self_agent, "_tab_id", "")
                with _RunningAgentState._registry_lock:
                    state = _RunningAgentState.running_agent_states.get(
                        tab_id,
                    )
                    if state is not None and state.pending_user_messages:
                        drained_messages.extend(
                            state.pending_user_messages,
                        )
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
            printer = kwargs.get("printer") or getattr(
                self_agent, "printer", None,
            )
            if printer is not None:  # pragma: no branch
                printer.print(
                    raw,
                    type="result",
                    step_count=1,
                    total_tokens=10,
                    cost="$0.0010",
                )
            return raw

        self._parent_class.run = stub_run

        # 1. A remote webview connects BEFORE the task starts.
        _writer, received, _got = self._open_viewer()
        self._wait_for_uds_writer(1)

        # 2. Launch a third-party agent via the API-based launcher
        # against the SAME daemon the webview is connected to.
        agent = SlackAgent()
        out: dict[str, Any] = {}

        def launch() -> None:
            out["result"] = run_agent_via_kiss_web(
                agent,
                "remote webview task",
                work_dir=self.repo,
                sock_path=self.sock_path,
            )

        t = threading.Thread(target=launch, daemon=True)
        t.start()
        try:
            assert started.wait(timeout=30), "agent run never started"

            # The API launch registered its tab on the daemon; find it
            # via the live registry (the tab id is minted by
            # ``kiss.server.sorcar.run``).
            tab_id = ""
            deadline = time.time() + 10
            while time.time() < deadline and not tab_id:
                with _RunningAgentState._registry_lock:
                    for tid, st in (
                        _RunningAgentState.running_agent_states.items()
                    ):
                        if tid.startswith("api-") and st.is_task_active:
                            tab_id = tid
                            break
                time.sleep(0.02)
            assert tab_id, "API launch never appeared in the registry"

            # 3. OPEN: the remote webview must receive the task's live
            # events stamped with the launch's tab id.
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

            # 4. INTERACT: send appendUserMessage from the webview.
            self._send_from_viewer({
                "type": "appendUserMessage",
                "tabId": tab_id,
                "prompt": "follow-up from the webview",
            })

            # The running agent must observe the queued message.
            deadline = time.time() + 10
            while time.time() < deadline and not drained_messages:
                time.sleep(0.05)
            assert drained_messages == ["follow-up from the webview"], (
                "appendUserMessage from the remote webview never reached "
                "the running third-party agent's message queue"
            )

            # And the webview gets the prompt echo for its message.
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
        # After completion the webview sees the task end.
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


if __name__ == "__main__":
    unittest.main()
