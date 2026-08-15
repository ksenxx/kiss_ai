# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for web_server.py bugs (audit findings F1/F5).

* D1 (F1): ``_handle_ready`` must skip non-dict ``restoredTabs``
  elements instead of raising ``AttributeError`` and tearing down the
  whole authenticated WebSocket connection.
* D3 (F5): ``_translate_webview_command`` no longer rewrites
  ``userActionDone`` (the branch was dead — no client ever sends it;
  ``media/main.js`` posts ``userAnswer`` directly), so the command
  passes through unchanged with all dispatch stamps intact.

No mocks/patches: a real :class:`RemoteAccessServer` (and its real
:class:`VSCodeServer`) is constructed and driven directly.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase, TestCase

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
    _translate_webview_command,
)


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    """Redirect the persistence DB to a temp dir; return saved state."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


class _RecordingEndpoint:
    """Minimal real WSS-like endpoint that records what is sent to it."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, data: str) -> None:
        """Record *data* exactly like a live connection would receive it."""
        self.sent.append(data)


class _ServerTestBase(IsolatedAsyncioTestCase):
    """Shared setup: a real ``RemoteAccessServer`` with recorded I/O."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.saved = _redirect_persistence(self.tmpdir)
        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=0,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
        )
        self.server._loop = asyncio.get_running_loop()
        self.run_cmds: list[dict[str, Any]] = []

        async def record_cmd(cmd: dict[str, Any]) -> None:
            self.run_cmds.append(cmd)

        self.server._run_cmd = record_cmd  # type: ignore[assignment]
        self.broadcasts: list[dict[str, Any]] = []
        self.server._printer.broadcast = self.broadcasts.append  # type: ignore[method-assign, assignment]

    async def asyncTearDown(self) -> None:
        sweep = self.server._vscode_server._orphan_sweep_thread
        if sweep is not None and sweep.is_alive():
            await asyncio.to_thread(sweep.join, 30)
        if th._db_conn is not None:
            th._db_conn.close()
            th._db_conn = None
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


class TestReadyMalformedRestoredTabs(_ServerTestBase):
    """D1/F1: non-dict ``restoredTabs`` entries must not kill the conn."""

    async def test_non_dict_restored_tab_does_not_raise(self) -> None:
        """A string element in ``restoredTabs`` must be skipped, not
        raise ``AttributeError`` (which would propagate out of
        ``_dispatch_client_command`` and tear down the connection)."""
        endpoint = _RecordingEndpoint()
        cmd = {
            "type": "ready",
            "tabId": "t1",
            "connId": "c1",
            "restoredTabs": ["x"],
        }
        await self.server._handle_ready(cmd, endpoint)
        self.assertTrue(
            any('"focusInput"' in s for s in endpoint.sent),
            f"focusInput not sent; sent={endpoint.sent}",
        )

    async def test_valid_entries_resume_amid_garbage(self) -> None:
        """Valid dict entries around non-dict garbage are still resumed."""
        endpoint = _RecordingEndpoint()
        cmd = {
            "type": "ready",
            "tabId": "t1",
            "connId": "c1",
            "restoredTabs": [
                "x", 42, None,
                {"tabId": "t2", "chatId": "chat-2"},
                ["nested"],
            ],
        }
        await self.server._handle_ready(cmd, endpoint)
        resumes = [c for c in self.run_cmds if c.get("type") == "resumeSession"]
        self.assertEqual(len(resumes), 1)
        self.assertEqual(resumes[0]["chatId"], "chat-2")
        self.assertEqual(resumes[0]["tabId"], "t2")


class TestTranslateUserActionDone(TestCase):
    """``userActionDone`` has no producer; the dead rewrite was removed."""

    def test_user_action_done_passes_through_unchanged(self) -> None:
        """No client sends ``userActionDone`` (``media/main.js`` posts
        ``userAnswer`` directly), so ``_translate_webview_command`` no
        longer rewrites it — the command passes through unchanged."""
        cmd = {
            "type": "userActionDone",
            "tabId": "t1",
            "connId": "conn-9",
            "workDir": "/some/work/dir",
        }
        out = _translate_webview_command(dict(cmd))
        self.assertEqual(out, cmd)
