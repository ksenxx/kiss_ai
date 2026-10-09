# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Server simplification (server.py / sorcar.py): replay order and terminals.

``_replay_session`` used to have two bodies: one for a chat with a
``task_history`` row and one for a chat whose task was still in its
setup window (no row yet).  The no-row body broadcast the viewer's
``status running:true`` and its ``task_events`` transcript BEFORE
publishing the tab to the shared registry, so a mirroring client could
receive events for a tab it had not yet seen in ``tabs_state``.  Both
cases now run through one body and publish the tab first.  Everything
here is real: a real ``VSCodeServer``, a run submitted through the real
``_cmd_run`` and parked in a real SEA getter (so it never reaches an
LLM), the real replay.

``ServerApi.terminal_open`` now forks the terminal's shell in a worker
thread instead of on the event loop; the end-to-end check drives it
through a real ``RemoteAccessServer`` over WSS from a remote
(password-authenticated) connection while a second connection keeps
being served.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

from websockets.asyncio.client import connect

from kiss.core.vscode_config import CONFIG_PATH, save_config
from kiss.server import agent_state
from kiss.server.server import VSCodeServer
from kiss.server.web_server import RemoteAccessServer
from kiss.tests.conftest import posix_only
from kiss.tests.server._memory_printer import MemoryPrinter

_BLOCKING_SCRIPT = textwrap.dedent(
    """
    import pathlib
    import time

    from kiss.agents.seas.base.base_sea import BaseSea

    _DIR = pathlib.Path(__file__).resolve().parent


    class Sea(BaseSea):
        def settings(self, settings):
            \"\"\"Block until released, then raise (the task ends in setup).\"\"\"
            (_DIR / "entered").write_text("1", encoding="utf-8")
            deadline = time.time() + 60
            while time.time() < deadline:
                if (_DIR / "release").exists():
                    raise RuntimeError("released")
                time.sleep(0.02)
            raise RuntimeError("timed out waiting for the release")
    """
)

_PASSWORD = "simplify-server-b-password"


def _wait(predicate: Any, timeout: float) -> bool:
    """Poll *predicate* every 20 ms until it holds or *timeout* passes."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestReplayPublishesTabBeforeEvents(TestCase):
    """A pre-row running chat's replay publishes the tab first."""

    def setUp(self) -> None:
        os.environ.setdefault("KISS_WORKDIR", "/tmp")
        agent_state.agent_states.clear()
        self.tmp = Path(tempfile.mkdtemp(prefix="kiss-simplify-server-b-"))
        self.work_dir = self.tmp / "wd"
        self.work_dir.mkdir()
        self.script = self.tmp / "agent.py"
        self.script.write_text(_BLOCKING_SCRIPT, encoding="utf-8")
        self.printer = MemoryPrinter()
        self.server = VSCodeServer(printer=self.printer)
        self.server.work_dir = str(self.work_dir)

    def tearDown(self) -> None:
        (self.tmp / "release").write_text("1", encoding="utf-8")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and any(
            s.task_thread is not None and s.task_thread.is_alive()
            for s in agent_state.agent_states.values()
        ):
            time.sleep(0.05)
        agent_state.agent_states.clear()

    def _start_parked_run(self, tab_id: str, chat_id: str) -> None:
        self.server._cmd_run(
            {
                "type": "run",
                "prompt": "parked in setup",
                "tabId": tab_id,
                "taskId": f"tok-{tab_id}",
                "chatId": chat_id,
                "workDir": str(self.work_dir),
                "useWorktree": False,
                "isParallel": False,
                "autoCommit": False,
                "seaPath": str(self.script),
            }
        )
        self.assertTrue(
            _wait((self.tmp / "entered").exists, 30.0),
            "the run never reached the SEA getter",
        )

    def test_tabs_state_precedes_status_and_transcript(self) -> None:
        launcher, viewer, chat_id = "launcher-tab", "viewer-tab", "chat-prerow"
        self._start_parked_run(launcher, chat_id)
        first = len(self.printer.emitted)

        self.server._replay_session(chat_id, viewer)

        emitted = list(self.printer.emitted)[first:]
        publish_idx = next(
            i
            for i, ev in enumerate(emitted)
            if ev.get("type") == "tabs_state"
            and any(t.get("tabId") == viewer for t in ev.get("tabs", []))
        )
        status_idx = next(
            i
            for i, ev in enumerate(emitted)
            if ev.get("type") == "status" and ev.get("tabId") == viewer
        )
        events_idx = next(
            i
            for i, ev in enumerate(emitted)
            if ev.get("type") == "task_events" and ev.get("tabId") == viewer
        )
        self.assertLess(publish_idx, status_idx)
        self.assertLess(publish_idx, events_idx)
        self.assertLess(status_idx, events_idx)
        self.assertTrue(emitted[status_idx]["running"])
        # No row yet: the heading is the prompt stamped on the live state.
        self.assertEqual(emitted[events_idx]["task"], "parked in setup")
        self.assertEqual(emitted[events_idx]["chat_id"], chat_id)
        self.assertEqual(emitted[events_idx]["extra"], "")

    def test_nothing_live_and_no_row_reports_not_running(self) -> None:
        viewer, chat_id = "idle-viewer", "chat-without-rows"
        first = len(self.printer.emitted)

        self.server._replay_session(chat_id, viewer)

        emitted = list(self.printer.emitted)[first:]
        statuses = [
            ev for ev in emitted if ev.get("type") == "status" and ev.get("tabId") == viewer
        ]
        self.assertEqual([ev["running"] for ev in statuses], [False])
        self.assertFalse(
            [ev for ev in emitted if ev.get("type") == "task_events" and ev.get("tabId") == viewer],
            "an empty chat must not replay an empty transcript",
        )
        publish_idx = next(
            i
            for i, ev in enumerate(emitted)
            if ev.get("type") == "tabs_state"
            and any(t.get("tabId") == viewer for t in ev.get("tabs", []))
        )
        self.assertLess(publish_idx, emitted.index(statuses[0]))


def _pick_free_port() -> int:
    """Return an OS-assigned free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    """Permissive SSL context for the dev self-signed cert."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


@posix_only("terminal tabs need a pty")
class TestTerminalOpenOverWss(IsolatedAsyncioTestCase):
    """``terminalOpen`` from a remote connection opens a real shell."""

    async def asyncSetUp(self) -> None:
        self._port = _pick_free_port()
        self._orig_config: str | None = None
        if CONFIG_PATH.exists():
            self._orig_config = CONFIG_PATH.read_text()
        save_config({"remote_password": _PASSWORD})
        self._server = RemoteAccessServer(
            host="127.0.0.1",
            port=self._port,
            work_dir=tempfile.mkdtemp(),
            use_tunnel=False,
        )
        await self._server.start_async()

    async def asyncTearDown(self) -> None:
        await self._server.stop_async()
        if self._orig_config is not None:
            CONFIG_PATH.write_text(self._orig_config)
        elif CONFIG_PATH.exists():
            CONFIG_PATH.unlink()

    async def _remote_connection(self) -> Any:
        ws = await connect(
            f"wss://127.0.0.1:{self._port}/ws",
            ssl=_no_verify_ssl(),
        )
        await ws.send(json.dumps({"type": "auth", "password": _PASSWORD}))
        await self._recv_type(ws, "auth_ok")
        return ws

    async def _recv_type(self, ws: Any, expected: str) -> dict[str, Any]:
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            if isinstance(msg, dict) and msg.get("type") == expected:
                return msg

    async def test_terminal_opens_while_another_connection_is_served(self) -> None:
        tab_id = "term-tab-1"
        opener = await self._remote_connection()
        other = await self._remote_connection()
        try:
            await opener.send(
                json.dumps(
                    {
                        "type": "terminalOpen",
                        "tab_id": tab_id,
                        "cols": 80,
                        "rows": 24,
                    }
                )
            )
            # The other connection's command is dispatched and answered
            # while the opener's shell is being forked.
            await other.send(json.dumps({"type": "ping"}))
            await self._recv_type(other, "pong")

            opened = await self._recv_type(opener, "terminalOpened")
            self.assertEqual(opened["tab_id"], tab_id)
            self.assertFalse(opened["attached"])
            self.assertTrue(opened["shell"])

            await opener.send(
                json.dumps(
                    {
                        "type": "terminalInput",
                        "tab_id": tab_id,
                        "data": "echo kiss-$((40+2))\n",
                    }
                )
            )
            output = ""
            while "kiss-42" not in output:
                data = await self._recv_type(opener, "terminalData")
                output += data["data"]
            await opener.send(json.dumps({"type": "terminalClose", "tab_id": tab_id}))
            await self._recv_type(opener, "terminalExit")
        finally:
            await opener.close()
            await other.close()
