# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Parity tests: the remote webapp must behave like the VS Code extension.

The remote webapp (browser over WSS) and the VS Code extension (UDS)
share one frontend (``media/main.js``) and one backend dispatch path
(:meth:`RemoteAccessServer._dispatch_client_command`).  These tests
lock in the behaviours that previously diverged between the two:

* ``submit`` must forward the webview's ``autoCommit`` toggle into the
  backend ``run`` command (the extension's ``_startTask`` always did;
  the web server used to drop it).
* ``runUpdate`` must locate and run ``~/kiss_ai/install.sh`` exactly
  like the extension's ``_runUpdate()`` / ``installerPath.js`` — and
  must never surface as an "Unknown command" error broadcast.
* ``sizeReport`` (the webview's reply to the extension-only
  ``measureSize`` request) must be silently ignored, like the other
  VS Code-only webview messages.

All tests drive the server through a real UDS client connection, the
same newline-delimited JSON protocol browsers speak over WSS — both
transports now share :meth:`_dispatch_client_command` verbatim.
"""

from __future__ import annotations

import asyncio
import json
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class TestWebExtensionParity(IsolatedAsyncioTestCase):
    """End-to-end parity tests over the shared dispatch path."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)

        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        # Point the updater paths at per-test temp locations so the
        # tests never touch the real ~/kiss_ai or ~/.kiss/update.log.
        self.install_root = Path(self.tmpdir) / "kiss_ai"
        self.server._install_root = self.install_root
        self.server._update_log_path = Path(self.tmpdir) / "update.log"
        await self.server.start_async()

    async def asyncTearDown(self) -> None:
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await asyncio.open_unix_connection(
            str(self.uds_path), limit=16 * 1024 * 1024,
        )

    async def _send(
        self, writer: asyncio.StreamWriter, cmd: dict[str, Any],
    ) -> None:
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _read_event(
        self, reader: asyncio.StreamReader, timeout: float = 2.0,
    ) -> dict[str, Any]:
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        assert line, "UDS closed unexpectedly"
        msg = json.loads(line.decode("utf-8"))
        assert isinstance(msg, dict)
        return msg

    async def _drain_until(
        self,
        reader: asyncio.StreamReader,
        wanted_type: str,
        max_events: int = 50,
        timeout: float = 2.0,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Read events until *wanted_type*; return (event, all_seen)."""
        seen: list[dict[str, Any]] = []
        for _ in range(max_events):
            msg = await self._read_event(reader, timeout=timeout)
            seen.append(msg)
            if msg.get("type") == wanted_type:
                return msg, seen
        raise AssertionError(
            f"did not observe a {wanted_type!r} event within "
            f"{max_events} messages; saw {[m.get('type') for m in seen]}",
        )

    @staticmethod
    def _assert_no_unknown_command(events: list[dict[str, Any]]) -> None:
        for ev in events:
            if ev.get("type") == "error" and "Unknown command" in str(
                ev.get("text", ""),
            ):
                raise AssertionError(f"Unknown-command error broadcast: {ev}")

    async def test_run_update_missing_script_reports_extension_error(
        self,
    ) -> None:
        """``runUpdate`` with no install.sh mirrors the extension's error.

        The extension shows ``Cannot update KISS Sorcar: install.sh not
        found in ~/kiss_ai.``; the web server must broadcast the same
        wording as an ``error`` event — NOT the previous
        ``Unknown command: runUpdate`` broadcast.
        """
        reader, writer = await self._connect()
        try:
            await self._send(writer, {"type": "runUpdate"})
            err, seen = await self._drain_until(reader, "error")
            self._assert_no_unknown_command(seen[:-1])
            text = str(err.get("text", ""))
            self.assertIn("Cannot update KISS Sorcar", text)
            self.assertIn("install.sh not found", text)
            self.assertIn(str(self.install_root), text)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_run_update_executes_install_script(self) -> None:
        """``runUpdate`` runs install.sh and broadcasts a notice.

        Mirrors the extension's behaviour of announcing "An update of
        KISS Sorcar is getting installed…" and executing the script
        (in a terminal there; as a detached subprocess here).
        """
        self.install_root.mkdir(parents=True, exist_ok=True)
        marker = self.install_root / "marker.txt"
        script = self.install_root / "install.sh"
        script.write_text(
            "#!/bin/bash\necho updated > marker.txt\necho done\n",
        )
        reader, writer = await self._connect()
        try:
            await self._send(writer, {"type": "runUpdate"})
            notice, seen = await self._drain_until(reader, "notice")
            self._assert_no_unknown_command(seen)
            self.assertIn(
                "An update of KISS Sorcar is getting installed",
                str(notice.get("text", "")),
            )
            # The script runs detached; poll for its side effect.
            for _ in range(100):
                if marker.is_file():
                    break
                await asyncio.sleep(0.05)
            self.assertTrue(marker.is_file(), "install.sh did not run")
            self.assertEqual(marker.read_text().strip(), "updated")
            # Script stdout was captured in the update log.
            log = self.server._update_log_path
            for _ in range(100):
                if log.is_file() and "done" in log.read_text():
                    break
                await asyncio.sleep(0.05)
            self.assertIn("done", log.read_text())
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_size_report_is_silently_ignored(self) -> None:
        """``sizeReport`` must not produce an Unknown-command error."""
        reader, writer = await self._connect()
        try:
            await self._send(
                writer,
                {"type": "sizeReport", "innerWidth": 100, "screenWidth": 200},
            )
            # activeTasksQuery acts as a synchronisation barrier: its
            # response arrives after any (hypothetical) error from the
            # preceding sizeReport would have been broadcast.
            await self._send(writer, {"type": "activeTasksQuery"})
            _, seen = await self._drain_until(reader, "activeTasksResponse")
            self._assert_no_unknown_command(seen)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def test_submit_forwards_auto_commit_to_run(self) -> None:
        """A webapp ``submit`` with ``autoCommit: true`` reaches the task.

        The VS Code extension's ``_startTask`` forwards the toggle in
        the ``run`` command and ``task_runner`` flips
        ``tab.auto_commit_mode``; the web server's submit → run
        translation must do the same.  Uses a stub agent (the pattern
        ``_run_task`` explicitly supports for tests) so no LLM call is
        made, and a fake API key so a model is "available".
        """
        from kiss.agents.sorcar.running_agent_state import _RunningAgentState
        from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
        from kiss.core import config as config_module
        from kiss.core.models.model_info import get_available_models

        keys = config_module.DEFAULT_CONFIG
        saved_key = keys.ANTHROPIC_API_KEY
        keys.ANTHROPIC_API_KEY = "test-anthropic-key"
        try:
            available = get_available_models()
            self.assertTrue(available, "no model available with fake key")
            model = next(m for m in available if m.startswith("claude-"))

            tab_id = "tab-parity-autocommit"
            vscode = self.server._vscode_server
            tab = vscode._get_tab(tab_id)
            agent = WorktreeSorcarAgent("Sorcar VS Code")
            ran = threading.Event()

            def fake_run(**kwargs: Any) -> None:
                ran.set()

            agent.run = fake_run  # type: ignore[assignment]
            tab.agent = agent
            tab.stop_event = threading.Event()
            tab.user_answer_queue = queue.Queue()

            work_dir = Path(self.tmpdir) / "work"
            work_dir.mkdir(parents=True, exist_ok=True)

            reader, writer = await self._connect()
            try:
                await self._send(writer, {
                    "type": "submit",
                    "tabId": tab_id,
                    "prompt": "do a thing",
                    "model": model,
                    "workDir": str(work_dir),
                    "attachments": [],
                    "useWorktree": False,
                    "useParallel": False,
                    "autoCommit": True,
                })
                # The submit handler echoes setTaskText before running.
                _, seen = await self._drain_until(reader, "setTaskText")
                self._assert_no_unknown_command(seen)
                # auto_commit_mode is set on the worker thread just
                # before the agent runs; wait for the stub to start.
                self.assertTrue(
                    await asyncio.get_running_loop().run_in_executor(
                        None, ran.wait, 10.0,
                    ),
                    "stub agent.run never started",
                )
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    with _RunningAgentState._registry_lock:
                        live = _RunningAgentState.running_agent_states.get(
                            tab_id,
                        )
                    if live is not None and live.auto_commit_mode:
                        break
                    await asyncio.sleep(0.05)
                with _RunningAgentState._registry_lock:
                    live = _RunningAgentState.running_agent_states.get(tab_id)
                assert live is not None
                self.assertTrue(
                    live.auto_commit_mode,
                    "autoCommit was dropped on the submit → run path",
                )
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
        finally:
            keys.ANTHROPIC_API_KEY = saved_key
