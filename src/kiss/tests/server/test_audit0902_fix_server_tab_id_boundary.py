# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""E2E: a wire ``tabId`` is normalised ONCE at the API boundary.

``TabRegistry`` strips surrounding whitespace from every tab id it
stores, so a client that sends ``"  tab-1 "`` sees the canonical
``"tab-1"`` in every ``tabs_state`` snapshot.  The command handlers,
however, kept using the raw string: ``_cmd_run`` stored it in
``AgentState.tab_id`` and ``_tab_chat_views``, and ``_close_tab``
looked the raw/canonical id up exactly.  One wire tab thus acquired
two identities — closing the canonical id removed the registry row
but left the ``AgentState`` (and any worktree it owns) behind.

These tests drive ``openTab`` / ``run`` / ``closeTab`` through a real
``wss://`` connection and a real UDS connection and assert that after
closing with the canonical id NO backend bookkeeping for the tab
remains, under either spelling.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import connect

import kiss.core.vscode_config as vc
from kiss.server import agent_state
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert

RAW_TAB = "  tab-1 "
CANON_TAB = "tab-1"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _no_verify_ssl() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


class TestTabIdNormalisedAtBoundary(IsolatedAsyncioTestCase):
    """Whitespace-padded tab ids never create a second backend identity."""

    async def asyncSetUp(self) -> None:
        agent_state.agent_states.clear()
        self.tmpdir = Path(tempfile.mkdtemp(prefix="kiss-tabid-boundary-"))
        self._saved_cfg = (vc.CONFIG_DIR, vc.CONFIG_PATH)
        vc.CONFIG_DIR = self.tmpdir / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"
        self.work_dir = self.tmpdir / "repo"
        self.work_dir.mkdir()
        certfile, keyfile = self.tmpdir / "cert.pem", self.tmpdir / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.port = _free_port()
        self.uds_path = self.tmpdir / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=self.tmpdir / "remote-url.json",
            uds_path=self.uds_path,
            work_dir=str(self.work_dir),
        )
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        agent_state.agent_states.clear()
        vc.CONFIG_DIR, vc.CONFIG_PATH = self._saved_cfg
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    async def _recv_until(recv: Any, reply_type: str) -> dict[str, Any]:
        while True:
            msg = json.loads(await asyncio.wait_for(recv(), 30))
            if msg.get("type") == reply_type:
                return dict(msg)

    async def _wait_task_threads_dead(self) -> None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            with agent_state.STATE_LOCK:
                alive = [
                    st for st in agent_state.agent_states.values()
                    if st.thread_alive()
                ]
            if not alive:
                return
            await asyncio.sleep(0.05)
        self.fail("task thread did not finish")

    def _tab_ids_with_state(self) -> set[str]:
        with agent_state.STATE_LOCK:
            return {st.tab_id for st in agent_state.agent_states.values()}

    def _assert_no_backend_trace(self) -> None:
        backend = self.server._vscode_server
        self.assertEqual(self._tab_ids_with_state(), set())
        self.assertFalse(backend.tab_registry.has_tab(CANON_TAB))
        self.assertNotIn(RAW_TAB, backend._tab_chat_views)
        self.assertNotIn(CANON_TAB, backend._tab_chat_views)

    def _run_cmd(self, tab_id: str) -> dict[str, Any]:
        # A model name that does not exist makes the real task thread
        # fail fast without any network call; the AgentState created by
        # ``_cmd_run`` survives the failure (tabs keep their state until
        # closed), which is exactly the leak this test guards.
        return {
            "type": "run",
            "prompt": "audit0902 tab-id boundary probe",
            "tabId": tab_id,
            "workDir": str(self.work_dir),
            "useWorktree": False,
            "autoCommit": False,
            "model": "no-such-model-audit0902",
        }

    async def test_wss_open_run_close_with_padded_tab_id(self) -> None:
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await self._recv_until(ws.recv, "auth_ok")

            await ws.send(json.dumps(
                {"type": "openTab", "tabId": RAW_TAB, "title": "padded"},
            ))
            tabs = await self._recv_until(ws.recv, "tabs_state")
            self.assertEqual(
                [t["tabId"] for t in tabs["tabs"]], [CANON_TAB],
            )

            await ws.send(json.dumps(self._run_cmd(RAW_TAB)))
            clear = await self._recv_until(ws.recv, "clear")
            # The run's own events must already speak the canonical id.
            self.assertEqual(clear["tabId"], CANON_TAB)
            await self._wait_task_threads_dead()
            self.assertEqual(self._tab_ids_with_state(), {CANON_TAB})
            self.assertIn(CANON_TAB, self.server._vscode_server._tab_chat_views)
            self.assertNotIn(RAW_TAB, self.server._vscode_server._tab_chat_views)

            # Close with the canonical id every client actually shows.
            await ws.send(json.dumps({"type": "closeTab", "tabId": CANON_TAB}))
            tabs = await self._recv_until(ws.recv, "tabs_state")
            self.assertEqual(tabs["tabs"], [])
        self._assert_no_backend_trace()

    async def test_wss_close_with_padded_id_drops_canonical_state(self) -> None:
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await self._recv_until(ws.recv, "auth_ok")
            await ws.send(json.dumps(self._run_cmd(CANON_TAB)))
            await self._recv_until(ws.recv, "clear")
            await self._wait_task_threads_dead()
            self.assertEqual(self._tab_ids_with_state(), {CANON_TAB})
            await ws.send(json.dumps({"type": "closeTab", "tabId": RAW_TAB}))
            tabs = await self._recv_until(ws.recv, "tabs_state")
            self.assertEqual(tabs["tabs"], [])
        self._assert_no_backend_trace()

    async def test_validation_error_reply_carries_canonical_tab_id(self) -> None:
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await self._recv_until(ws.recv, "auth_ok")
            # ``run`` without its required ``prompt`` is rejected; the
            # error must address the tab by the id every client shows.
            await ws.send(json.dumps({"type": "run", "tabId": RAW_TAB}))
            error = await self._recv_until(ws.recv, "error")
            self.assertEqual(error["tabId"], CANON_TAB)
            self.assertIn("missing prompt", error["text"])
            # A padded id that is ONLY whitespace canonicalises to no
            # tab at all: the error is sent without a ``tabId``.
            await ws.send(json.dumps({"type": "run", "tabId": "   "}))
            error = await self._recv_until(ws.recv, "error")
            self.assertNotIn("tabId", error)
        self._assert_no_backend_trace()

    async def test_non_string_tab_id_is_left_for_the_handler(self) -> None:
        async with connect(
            f"wss://127.0.0.1:{self.port}/ws", ssl=_no_verify_ssl(),
        ) as ws:
            await ws.send(json.dumps({"type": "auth", "password": ""}))
            await self._recv_until(ws.recv, "auth_ok")
            await ws.send(json.dumps(
                {"type": "checkPaths", "paths": ["x"], "tabId": 7},
            ))
            reply = await self._recv_until(ws.recv, "pathsExist")
            self.assertEqual(reply["tabId"], "")
        self._assert_no_backend_trace()

    async def test_uds_padded_tab_id_registers_one_local_tab(self) -> None:
        reader, writer = await asyncio.open_unix_connection(str(self.uds_path))

        async def recv() -> str:
            line = await reader.readline()
            self.assertTrue(line, "UDS connection closed")
            return line.decode("utf-8")

        try:
            writer.write(
                (json.dumps({"type": "openTab", "tabId": RAW_TAB}) + "\n")
                .encode("utf-8"),
            )
            await writer.drain()
            tabs = await self._recv_until(recv, "tabs_state")
            self.assertEqual([t["tabId"] for t in tabs["tabs"]], [CANON_TAB])
            printer = self.server._printer
            with printer._ws_lock:
                local_tabs = dict(printer._local_uds_tab_counts)
            self.assertEqual(local_tabs, {CANON_TAB: 1})

            writer.write(
                (json.dumps({"type": "closeTab", "tabId": CANON_TAB}) + "\n")
                .encode("utf-8"),
            )
            await writer.drain()
            tabs = await self._recv_until(recv, "tabs_state")
            self.assertEqual(tabs["tabs"], [])
            with printer._ws_lock:
                local_tabs = dict(printer._local_uds_tab_counts)
            self.assertEqual(local_tabs, {})
        finally:
            writer.close()
            await writer.wait_closed()
        self._assert_no_backend_trace()
