# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9 (batch A / A1): ``current()`` returning a resolved hunk.

After a PARTIAL ``reject-all`` failure — one file restored successfully
(hunks marked ``"rejected"``, ``advance()`` never called in that
branch), a sibling file's restore raising ``OSError`` (canonical
trigger: the agent replaced the deleted file with a directory) —
``_WebMergeState.current()`` still pointed ``_pos`` at the RESOLVED
hunk of the restored file:

* the ``merge_nav`` broadcast highlighted a resolved hunk as current,
* a follow-up ``accept`` click called ``mark_resolved(*cur, "accepted")``
  on it, silently flipping its recorded status from ``"rejected"`` to
  ``"accepted"`` while the on-disk content had already been reverted
  to base — recorded status and disk contents disagreed, and the
  actually-unresolved hunk stayed unaddressed.

``current()`` must skip to the next unresolved hunk (one is guaranteed
to exist whenever ``remaining > 0``).  Real server, real files, real
``IsADirectoryError`` — no mocks.
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import RemoteAccessServer


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


def _modified_file_entry(work: Path, name: str) -> dict[str, Any]:
    """Build a merge entry for a normally-modified text file."""
    current = work / name
    base = work / f"{name}.base"
    lines = "".join(f"line{i}\n" for i in range(8))
    current.write_text(lines.replace("line2\n", "edit2\n"))
    base.write_text(lines)
    return {
        "name": name,
        "current": str(current),
        "base": str(base),
        "target": str(current),
        "hunks": [{"bs": 2, "bc": 1, "cs": 2, "cc": 1}],
    }


def _deleted_file_entry(work: Path, name: str) -> dict[str, Any]:
    """Merge entry for a deleted tracked file (``.deleted`` placeholder)."""
    merge_tmp = work / "merge-temp"
    placeholder = merge_tmp / ".deleted" / name
    placeholder.parent.mkdir(parents=True, exist_ok=True)
    placeholder.write_text("")
    base = merge_tmp / name
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("alpha\nbeta\n")
    return {
        "name": name,
        "current": str(placeholder),
        "base": str(base),
        "target": str(work / name),
        "hunks": [{"bs": 0, "bc": 2, "cs": 0, "cc": 0}],
    }


class TestCurrentSkipsResolvedHunk(IsolatedAsyncioTestCase):
    """``current()`` must never point at an already-resolved hunk."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bh9-a1-")
        self.saved = _redirect_persistence(self.tmpdir)
        self.work = Path(self.tmpdir) / "work"
        self.work.mkdir()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=0, work_dir=str(self.work),
        )
        self.server._loop = asyncio.get_running_loop()

    async def asyncTearDown(self) -> None:
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _open_partial_failure_review(self, tab_id: str) -> Any:
        """Register a 2-file review whose second file cannot be restored."""
        good = _modified_file_entry(self.work, "good.txt")
        bad = _deleted_file_entry(self.work, "cfg")
        # The agent replaced the deleted file with a directory of the
        # same name, so restoring it raises IsADirectoryError.
        (self.work / "cfg").mkdir()
        self.server._register_merge_state(
            tab_id, {"work_dir": str(self.work), "files": [good, bad]},
        )
        with self.server._merge_states_lock:
            return self.server._merge_states[tab_id]

    async def test_current_is_unresolved_after_partial_reject_all(self) -> None:
        """After a partial reject-all failure, current() must be unresolved."""
        tab_id = "tab-a1-cur"
        state = self._open_partial_failure_review(tab_id)

        await self.server._handle_web_merge_action({
            "type": "mergeAction", "action": "reject-all", "tabId": tab_id,
        })

        # The good file was rejected; the bad file's hunk stayed open.
        self.assertTrue(state.is_resolved(0, 0))
        self.assertFalse(state.is_resolved(1, 0))
        self.assertEqual(state.remaining, 1)

        cur = state.current()
        self.assertIsNotNone(cur)
        assert cur is not None
        self.assertFalse(
            state.is_resolved(*cur),
            "BUG: current() points at an already-resolved hunk after a "
            f"partial reject-all failure (cur={cur})",
        )
        self.assertEqual(cur, (1, 0))

    async def test_followup_accept_cannot_flip_rejected_status(self) -> None:
        """An accept after the partial failure must act on the OPEN hunk."""
        tab_id = "tab-a1-accept"
        state = self._open_partial_failure_review(tab_id)

        await self.server._handle_web_merge_action({
            "type": "mergeAction", "action": "reject-all", "tabId": tab_id,
        })
        # The user accepts what the UI shows as the current hunk.
        await self.server._handle_web_merge_action({
            "type": "mergeAction", "action": "accept", "tabId": tab_id,
        })

        statuses = {
            (r["fi"], r["hi"]): r["status"] for r in state.resolutions()
        }
        self.assertEqual(
            statuses.get((0, 0)), "rejected",
            "BUG: the follow-up accept flipped a REJECTED hunk (whose "
            "content was already reverted on disk) to 'accepted'",
        )
        self.assertEqual(
            statuses.get((1, 0)), "accepted",
            "the accept must resolve the still-open hunk of the failed file",
        )
        # Review is complete now, so the state must have been popped.
        with self.server._merge_states_lock:
            self.assertNotIn(tab_id, self.server._merge_states)


if __name__ == "__main__":
    unittest.main()
