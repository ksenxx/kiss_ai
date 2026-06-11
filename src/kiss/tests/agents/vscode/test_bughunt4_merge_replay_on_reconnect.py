# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt iteration 4: merge review lost forever on browser reload.

In the standalone web server the per-hunk merge review is owned by the
server-side :class:`_WebMergeState` (``_merge_states[tab_id]``).  The
``merge_data`` event that opens the review UI is tab-stamped, so
``WebPrinter.broadcast`` forwards it to currently-connected clients
only — it is never recorded or persisted.

When the browser reloads (or the WebSocket drops and reconnects) in
the middle of a review, ``_handle_ready`` re-claims the tab (cancels
the deferred close, resumes the chat session) but never re-emits the
in-flight merge review.  The reloaded page therefore shows no merge
UI, while the server still holds an unresolved ``_WebMergeState`` and
the backend tab is stuck ``is_merging`` — the user can never finish
(or even see) the review, so ``all-done`` / ``_finish_merge`` /
the autocommit prompt never fire.

The VS Code extension does not have this problem: its TypeScript
``MergeManager`` lives in the extension host and survives webview
reloads.  The web client's only source of truth is the server, so the
server must replay the review on reconnect.

This test drives a real :class:`RemoteAccessServer` over a real
``wss://`` connection (no mocks): it opens a review through the real
broadcast path, resolves one hunk through the real action handler,
then reconnects with ``ready.restoredTabs`` and asserts the new
connection receives the ``merge_data`` + ``merge_started`` +
``merge_nav`` replay (including the already-resolved hunk).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import ssl
import tempfile
import time
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from websockets.asyncio.client import ClientConnection, connect

import kiss.agents.sorcar.persistence as th
import kiss.agents.vscode.vscode_config as vc
from kiss.agents.vscode.web_server import (
    RemoteAccessServer,
    _generate_self_signed_cert,
)


def _redirect_persistence(tmpdir: str) -> tuple[Path, object, Path]:
    """Point the sorcar persistence layer at a per-test directory."""
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_dir = Path(tmpdir) / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    return saved  # type: ignore[return-value]


def _restore_persistence(saved: tuple[Path, object, Path]) -> None:
    """Undo :func:`_redirect_persistence`."""
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved  # type: ignore[assignment]


def _find_free_port() -> int:
    """Return an available TCP port."""
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


def _build_merge_data(work_dir: Path) -> dict:
    """Create real files and a 3-hunk single-file merge data structure."""
    current = work_dir / "f_current.txt"
    base = work_dir / "f_base.txt"
    target = work_dir / "f.txt"
    lines = "".join(f"line{i}\n" for i in range(30))
    current.write_text(lines)
    base.write_text(lines)
    target.write_text(lines)

    def hunk(start: int) -> dict[str, int]:
        return {"bs": start, "bc": 1, "cs": start, "cc": 1}

    return {
        "work_dir": str(work_dir),
        "files": [
            {
                "name": "f.txt",
                "current": str(current),
                "base": str(base),
                "target": str(target),
                "hunks": [hunk(2), hunk(10), hunk(18)],
            }
        ],
    }


class TestMergeReplayOnReconnect(IsolatedAsyncioTestCase):
    """A reconnecting client must receive the in-flight merge review."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt4-mrgreplay-")
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)

        self.port = _find_free_port()
        self.url = f"wss://127.0.0.1:{self.port}/ws"
        self.ctx = _no_verify_ssl()
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=self.port,
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=Path(self.tmpdir) / "sorcar.sock",
        )
        await self.server.start_async()
        self._sockets: list[ClientConnection] = []

    async def asyncTearDown(self) -> None:
        for ws in self._sockets:
            try:
                await ws.close()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_ok(self) -> ClientConnection:
        """Open + successfully authenticate one WSS connection."""
        ws = await connect(self.url, ssl=self.ctx)
        self._sockets.append(ws)
        await ws.send(json.dumps({"type": "auth", "password": ""}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        self.assertEqual(resp["type"], "auth_ok")
        return ws

    async def _collect_events(
        self, ws: ClientConnection, wanted: set[str], timeout: float = 6.0,
    ) -> dict[str, dict]:
        """Collect the first event of each *wanted* type within *timeout*."""
        got: dict[str, dict] = {}
        deadline = time.monotonic() + timeout
        while wanted - set(got) and time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
                break
            ev = json.loads(raw)
            etype = ev.get("type", "")
            if etype in wanted and etype not in got:
                got[etype] = ev
        return got

    async def test_reconnect_replays_in_flight_merge_review(self) -> None:
        """``ready`` with a restored mid-review tab must replay the review."""
        tab_id = "tab-m"
        Path(self.tmpdir, "work").mkdir(exist_ok=True)
        merge_data = _build_merge_data(Path(self.tmpdir) / "work")

        # Open the review through the real broadcast path (this is what
        # ``_start_merge_session`` does); the printer callback registers
        # the server-side ``_WebMergeState``.
        self.server._printer.broadcast({
            "type": "merge_data",
            "tabId": tab_id,
            "data": merge_data,
            "hunk_count": 3,
        })
        with self.server._merge_states_lock:
            self.assertIn(tab_id, self.server._merge_states)

        # User resolves one hunk through the real action handler, then
        # the browser reloads mid-review.
        await self.server._handle_web_merge_action(
            {"type": "mergeAction", "action": "accept", "tabId": tab_id},
        )

        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "ready",
            "tabId": "tab-fresh",
            "restoredTabs": [{"tabId": tab_id, "chatId": ""}],
        }))

        got = await self._collect_events(
            ws, {"merge_data", "merge_started", "merge_nav"},
        )

        self.assertIn(
            "merge_data", got,
            "BUG: reconnecting client never received the in-flight merge "
            "review — the merge UI is lost forever after a page reload",
        )
        self.assertEqual(got["merge_data"].get("tabId"), tab_id)
        files = got["merge_data"].get("data", {}).get("files", [])
        self.assertEqual(len(files), 1)
        self.assertIn("current_text", files[0])
        self.assertEqual(len(files[0].get("hunks", [])), 3)
        self.assertIn("merge_started", got)
        self.assertEqual(got["merge_started"].get("tabId"), tab_id)
        self.assertIn("merge_nav", got)
        nav = got["merge_nav"]
        self.assertEqual(nav.get("tabId"), tab_id)
        self.assertEqual(nav.get("remaining"), 2)
        self.assertEqual(nav.get("total"), 3)
        self.assertEqual(
            nav.get("resolved"),
            [{"fi": 0, "hi": 0, "status": "accepted"}],
        )

    async def test_ready_without_merge_state_sends_no_merge_events(self) -> None:
        """A plain reconnect with no in-flight review must not emit merge events."""
        ws = await self._connect_ok()
        await ws.send(json.dumps({
            "type": "ready",
            "tabId": "tab-clean",
            "restoredTabs": [{"tabId": "tab-idle", "chatId": ""}],
        }))
        got = await self._collect_events(
            ws, {"merge_data", "merge_started", "merge_nav"}, timeout=2.0,
        )
        self.assertEqual(
            got, {},
            f"unexpected merge replay for tabs with no review: {got}",
        )


if __name__ == "__main__":
    unittest.main()
