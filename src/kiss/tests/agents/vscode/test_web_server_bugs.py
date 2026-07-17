# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for three web_server.py bugs (audit findings F1/F2/F5).

* D1 (F1): ``_handle_ready`` must skip non-dict ``restoredTabs``
  elements instead of raising ``AttributeError`` and tearing down the
  whole authenticated WebSocket connection.
* D2 (F2): ``reject-file`` / ``reject-all`` must surgically revert
  only the UNRESOLVED hunks of a file; hunks the user already ACCEPTED
  must keep their content on disk (the old whole-file ``shutil.copy2``
  silently wiped them while still reporting them "accepted").
* D3 (F5): ``_translate_webview_command`` no longer rewrites
  ``userActionDone`` (the branch was dead — no client ever sends it;
  ``media/main.js`` posts ``userAnswer`` directly), so the command
  passes through unchanged with all dispatch stamps intact.

No mocks/patches: a real :class:`RemoteAccessServer` (and its real
:class:`VSCodeServer`) is constructed; merge actions run through the
real ``_handle_web_merge_action`` lock + ``_apply_web_merge_action``
code path against real temp files on disk.
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
        # Record (instead of executing) backend commands and broadcasts
        # so the tests stay hermetic: ``getModels`` / ``resumeSession``
        # would otherwise hit the model registry / persistence layer.
        self.run_cmds: list[dict[str, Any]] = []

        async def record_cmd(cmd: dict[str, Any]) -> None:
            self.run_cmds.append(cmd)

        self.server._run_cmd = record_cmd  # type: ignore[assignment]
        self.broadcasts: list[dict[str, Any]] = []
        self.server._printer.broadcast = self.broadcasts.append  # type: ignore[method-assign, assignment]

    async def asyncTearDown(self) -> None:
        # Join the orphan-task-sweep daemon thread BEFORE closing the
        # per-thread sqlite connection, restoring persistence paths, and
        # deleting the temp dir that holds the DB.  Otherwise the sweep
        # thread can still be executing ``db.execute`` against a
        # connection whose backing file is being removed, which crashes
        # the interpreter with a segfault inside the sqlite3 C layer.
        sweep = self.server._vscode_server._orphan_sweep_thread
        if sweep is not None and sweep.is_alive():
            await asyncio.to_thread(sweep.join, 30)
        with self.server._merge_states_lock:
            self.server._merge_states.clear()
        with self.server._pending_tab_closes_lock:
            for h in list(self.server._pending_tab_closes.values()):
                h.cancel()
            self.server._pending_tab_closes.clear()
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
        await self.server._handle_ready(cmd, endpoint)  # must not raise
        # The connection still got its focusInput reply.
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


class TestRejectFilePreservesAcceptedHunks(_ServerTestBase):
    """D2/F2: reject-file / reject-all must not revert ACCEPTED hunks."""

    BASE = "line1\nold2\nline3\nold4\nline5\n"
    CURRENT = "line1\nNEW2\nline3\nNEW4\nline5\n"

    def _make_merge_tab(self, tab_id: str) -> Path:
        """Register a real merge state for a temp file with 2 hunks."""
        base = Path(self.tmpdir) / f"{tab_id}.base"
        current = Path(self.tmpdir) / f"{tab_id}.txt"
        base.write_text(self.BASE)
        current.write_text(self.CURRENT)
        merge_data = {
            "work_dir": self.tmpdir,
            "files": [{
                "name": current.name,
                "base": str(base),
                "current": str(current),
                "hunks": [
                    {"bs": 1, "bc": 1, "cs": 1, "cc": 1},
                    {"bs": 3, "bc": 1, "cs": 3, "cc": 1},
                ],
            }],
        }
        self.server._register_merge_state(tab_id, merge_data)
        return current

    async def _accept_current_hunk(self, tab_id: str) -> None:
        await self.server._handle_web_merge_action(
            {"type": "mergeAction", "action": "accept", "tabId": tab_id},
        )

    def _assert_accepted_kept_unresolved_reverted(self, current: Path) -> None:
        disk = current.read_text()
        self.assertIn(
            "NEW2", disk,
            "accepted hunk 0 was reverted on disk (lost update): " + disk,
        )
        self.assertNotIn("NEW4", disk, "unresolved hunk 1 was not reverted")
        self.assertIn("old4", disk, "hunk 1 base content missing after revert")
        self.assertEqual(disk, "line1\nNEW2\nline3\nold4\nline5\n")

    def _last_merge_nav_resolved(self) -> dict[tuple[int, int], str]:
        navs = [b for b in self.broadcasts if b.get("type") == "merge_nav"]
        self.assertTrue(navs, "no merge_nav broadcast")
        return {
            (r["fi"], r["hi"]): r["status"] for r in navs[-1]["resolved"]
        }

    async def test_reject_file_keeps_accepted_hunk_on_disk(self) -> None:
        """accept hunk 0, then reject-file: hunk 0's content must stay
        on disk (it is still reported "accepted" in resolutions) while
        hunk 1 is reverted to base."""
        tab_id = "merge-reject-file"
        current = self._make_merge_tab(tab_id)
        await self._accept_current_hunk(tab_id)
        await self.server._handle_web_merge_action(
            {"type": "mergeAction", "action": "reject-file", "tabId": tab_id},
        )
        resolved = self._last_merge_nav_resolved()
        self.assertEqual(resolved[(0, 0)], "accepted")
        self.assertEqual(resolved[(0, 1)], "rejected")
        self._assert_accepted_kept_unresolved_reverted(current)

    async def test_reject_all_keeps_accepted_hunk_on_disk(self) -> None:
        """Same lost-update through the ``reject-all`` call site."""
        tab_id = "merge-reject-all"
        current = self._make_merge_tab(tab_id)
        await self._accept_current_hunk(tab_id)
        await self.server._handle_web_merge_action(
            {"type": "mergeAction", "action": "reject-all", "tabId": tab_id},
        )
        resolved = self._last_merge_nav_resolved()
        self.assertEqual(resolved[(0, 0)], "accepted")
        self.assertEqual(resolved[(0, 1)], "rejected")
        self._assert_accepted_kept_unresolved_reverted(current)

    async def test_reject_file_all_unresolved_reverts_whole_file(self) -> None:
        """With NO accepted hunks, reject-file still fully reverts."""
        tab_id = "merge-reject-file-full"
        current = self._make_merge_tab(tab_id)
        await self.server._handle_web_merge_action(
            {"type": "mergeAction", "action": "reject-file", "tabId": tab_id},
        )
        self.assertEqual(current.read_text(), self.BASE)


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
