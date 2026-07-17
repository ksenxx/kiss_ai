# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 5: client ``all-done`` leaves a zombie server merge state.

The VS Code extension's TypeScript ``MergeManager`` runs the per-hunk
review entirely in the extension host: individual accept/reject
actions never reach the Python backend — only a single
``mergeAction action=all-done`` is sent (over the UDS transport) when
the review finishes (``SorcarSidebarView.sendMergeAllDone``).

``RemoteAccessServer._dispatch_client_command`` forwards that
``all-done`` to the backend (``_cmd_merge_action`` → ``_finish_merge``)
but never pops the server-side ``_WebMergeState`` shadow that was
registered when the ``merge_data`` event was broadcast.  The stale,
fully-unresolved state then:

* makes a later webview reload (``ready`` with the same tab id) replay
  a ZOMBIE merge review (``merge_data`` + ``merge_started`` +
  ``merge_nav``) for a review that already finished, and
* leaks one ``_WebMergeState`` (with full file payloads) per VS Code
  merge review until the connection drops, where the deferred-close
  path then fires a spurious second ``all-done``.

This test drives a real :class:`RemoteAccessServer` over its real UDS
transport (exactly what the VS Code extension uses; no mocks).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import socket
import tempfile
import time
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server.web_server import (
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


def _build_merge_data(work_dir: Path) -> dict:
    """Create real files and a 2-hunk single-file merge data structure."""
    current = work_dir / "f.txt"
    base = work_dir / "f_base.txt"
    lines = "".join(f"line{i}\n" for i in range(20))
    current.write_text(lines.replace("line2\n", "edit2\n"))
    base.write_text(lines)
    return {
        "work_dir": str(work_dir),
        "files": [
            {
                "name": "f.txt",
                "current": str(current),
                "base": str(base),
                "target": str(current),
                "hunks": [
                    {"bs": 2, "bc": 1, "cs": 2, "cc": 1},
                    {"bs": 10, "bc": 1, "cs": 10, "cc": 1},
                ],
            }
        ],
    }


class TestStaleStateAfterClientAllDone(IsolatedAsyncioTestCase):
    """A client all-done must drop the server-side merge shadow state."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt5-alldone-")
        self.saved = _redirect_persistence(self.tmpdir)
        self._orig_cfg_dir = vc.CONFIG_DIR
        self._orig_cfg_path = vc.CONFIG_PATH
        vc.CONFIG_DIR = Path(self.tmpdir) / "config"
        vc.CONFIG_PATH = vc.CONFIG_DIR / "config.json"

        certfile = Path(self.tmpdir) / "cert.pem"
        keyfile = Path(self.tmpdir) / "key.pem"
        _generate_self_signed_cert(certfile, keyfile)

        self.uds_path = Path(self.tmpdir) / "sorcar.sock"
        self.server = RemoteAccessServer(
            host="127.0.0.1",
            port=_find_free_port(),
            certfile=str(certfile),
            keyfile=str(keyfile),
            url_file=Path(self.tmpdir) / "remote-url.json",
            uds_path=self.uds_path,
        )
        await self.server.start_async()
        self._conns: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []

    async def asyncTearDown(self) -> None:
        for _, writer in self._conns:
            try:
                writer.close()
            except Exception:
                pass
        await self.server.stop_async()
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        vc.CONFIG_DIR = self._orig_cfg_dir
        vc.CONFIG_PATH = self._orig_cfg_path
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def _connect_uds(
        self,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Open one UDS connection (the VS Code extension transport)."""
        reader, writer = await asyncio.open_unix_connection(
            str(self.uds_path),
            limit=16 * 1024 * 1024,
        )
        self._conns.append((reader, writer))
        return reader, writer

    async def _send(self, writer: asyncio.StreamWriter, cmd: dict) -> None:
        """Write one newline-delimited JSON command."""
        writer.write(json.dumps(cmd).encode("utf-8") + b"\n")
        await writer.drain()

    async def _collect_merge_events(
        self, reader: asyncio.StreamReader, timeout: float,
    ) -> list[str]:
        """Return the types of merge events received within *timeout*."""
        got: list[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(
                    reader.readline(), timeout=remaining,
                )
            except TimeoutError:
                break
            if not raw:
                break
            ev = json.loads(raw)
            if ev.get("type", "") in (
                "merge_data", "merge_started", "merge_nav",
            ):
                got.append(ev["type"])
        return got

    def _open_review(self, tab_id: str) -> None:
        """Register an in-flight review through the real broadcast path."""
        work = Path(self.tmpdir) / f"work-{tab_id}"
        work.mkdir(exist_ok=True)
        merge_data = _build_merge_data(work)
        self.server._printer.broadcast({
            "type": "merge_data",
            "tabId": tab_id,
            "data": merge_data,
            "hunk_count": 2,
        })
        with self.server._merge_states_lock:
            self.assertIn(tab_id, self.server._merge_states)

    async def test_all_done_drops_state_and_blocks_zombie_replay(self) -> None:
        """all-done from the extension must pop the server merge state."""
        tab_id = "tab-ts-review"
        self._open_review(tab_id)

        reader, writer = await self._connect_uds()
        # The TS MergeManager finished its editor-managed review: only
        # this single all-done ever reaches the backend.
        await self._send(writer, {
            "type": "mergeAction",
            "action": "all-done",
            "tabId": tab_id,
            "workDir": self.tmpdir,
        })

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with self.server._merge_states_lock:
                if tab_id not in self.server._merge_states:
                    break
            await asyncio.sleep(0.05)
        with self.server._merge_states_lock:
            self.assertNotIn(
                tab_id, self.server._merge_states,
                "BUG: server-side _WebMergeState survived the client's "
                "all-done — it leaks and later replays a zombie review",
            )

        # A webview reload after the finished review must NOT resurrect
        # the merge UI.
        await self._send(writer, {
            "type": "ready", "tabId": tab_id, "restoredTabs": [],
        })
        got = await self._collect_merge_events(reader, timeout=2.0)
        self.assertEqual(
            got, [],
            f"BUG: zombie merge review replayed after all-done: {got}",
        )

    async def test_web_driven_review_still_finishes_normally(self) -> None:
        """Control: the server-driven (web) review path is unaffected."""
        tab_id = "tab-web-review"
        self._open_review(tab_id)
        # Resolve both hunks through the real web action handler.
        for _ in range(2):
            await self.server._handle_web_merge_action(
                {"type": "mergeAction", "action": "accept", "tabId": tab_id},
            )
        with self.server._merge_states_lock:
            self.assertNotIn(tab_id, self.server._merge_states)


if __name__ == "__main__":
    unittest.main()
