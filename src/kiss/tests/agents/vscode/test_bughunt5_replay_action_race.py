# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 5: merge-review replay races an in-flight merge action.

Every merge action (``accept``/``reject``/``reject-all``/...) is
serialised per tab through ``RemoteAccessServer._merge_action_lock``:
the reject branches rewrite the reviewed files on disk (truncate +
write) and mutate the shared hunk ``cs`` offsets in the
:class:`_WebMergeState` while holding the lock.

``_replay_merge_review`` (the reconnect/reload path) reads those same
files (``_augment_merge_data``) and the same hunk dicts WITHOUT taking
the lock.  A browser that reloads while another client's ``reject`` /
``reject-all`` is mid-write therefore receives a torn replay: a
half-truncated ``current_text`` (``open(w)`` truncates before the new
content lands) and/or hunk offsets mid-mutation — the review UI
renders garbage that no later ``merge_nav`` broadcast can repair
(``merge_nav`` carries no file text).

The test pins the synchronisation contract deterministically: while
the per-tab action lock is held (exactly what an in-flight reject
does), a reconnecting client's ``ready`` must NOT receive the merge
replay; the replay must be delivered only after the action completes
(lock released).  A second test pins the post-wait re-check: if the
review finished while the replay waited, no merge events may be sent.
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


def _no_verify_ssl() -> ssl.SSLContext:
    """Return an SSL client context that skips certificate verification."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


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


class TestReplayMergeActionRace(IsolatedAsyncioTestCase):
    """Replay must serialise with in-flight merge actions on the tab."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bughunt5-replayrace-")
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

    async def _collect_merge_events(
        self, ws: ClientConnection, timeout: float,
    ) -> list[str]:
        """Return the types of merge events received within *timeout*."""
        got: list[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            except TimeoutError:
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

    async def test_replay_waits_for_in_flight_merge_action(self) -> None:
        """No merge replay may be emitted while a merge action is running."""
        tab_id = "tab-replay-race"
        self._open_review(tab_id)

        ws = await self._connect_ok()

        # Hold the tab's merge-action lock, exactly as an in-flight
        # reject (mid file-rewrite) does in _handle_web_merge_action.
        lock = self.server._merge_action_lock(tab_id)
        await lock.acquire()
        try:
            await ws.send(json.dumps({
                "type": "ready", "tabId": tab_id, "restoredTabs": [],
            }))
            during = await self._collect_merge_events(ws, timeout=1.5)
            self.assertEqual(
                during, [],
                "BUG: merge replay was sent while a merge action was "
                "still in flight on the tab — the reconnecting client "
                "can receive torn file contents / mid-mutation hunk "
                f"offsets (got {during})",
            )
        finally:
            lock.release()

        after = await self._collect_merge_events(ws, timeout=6.0)
        self.assertIn(
            "merge_data", after,
            "replay must be delivered once the in-flight action ends",
        )
        self.assertIn("merge_started", after)
        self.assertIn("merge_nav", after)

    async def test_replay_recheck_after_wait_skips_finished_review(self) -> None:
        """If the review ended while the replay waited, send nothing."""
        tab_id = "tab-replay-finished"
        self._open_review(tab_id)

        ws = await self._connect_ok()
        lock = self.server._merge_action_lock(tab_id)
        await lock.acquire()
        try:
            await ws.send(json.dumps({
                "type": "ready", "tabId": tab_id, "restoredTabs": [],
            }))
            # While the replay waits, the review finishes (this is what
            # the final accept/reject action does before releasing the
            # lock: the state is popped, then all-done is dispatched).
            await asyncio.sleep(0.3)
            with self.server._merge_states_lock:
                self.server._merge_states.pop(tab_id, None)
        finally:
            lock.release()

        got = await self._collect_merge_events(ws, timeout=2.0)
        self.assertEqual(
            got, [],
            f"no merge replay may be sent for a finished review: {got}",
        )


if __name__ == "__main__":
    unittest.main()
