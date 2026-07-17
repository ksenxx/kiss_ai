# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt 9 (batch A / A2): non-idempotent reject retry corrupts files.

``_reject_all_hunks_in_file`` splices each hunk's ``bc`` base lines
over its ``cc`` current lines, then shifts the ``cs`` offsets of later
PENDING hunks — but never updated the spliced hunk's own ``cc`` to
``bc``.  When a LATER hunk's write fails mid-file (a real disk fault:
``ENOSPC``/``EFBIG``/``EIO``), the ``reject-all`` handler marks NONE
of the file's hunks resolved, so the user's natural retry (clicking
reject-all again after the error banner) re-applies the
already-spliced first hunk with its STALE ``cc`` against content whose
region now holds ``bc`` lines — duplicating ``bc - cc`` lines and
shifting every later hunk onto the wrong lines.

The fault here is genuine: the test lowers ``RLIMIT_FSIZE`` (with
``SIGXFSZ`` ignored) so the second hunk's restore write really fails
with ``EFBIG`` after the first hunk's write succeeded.  Real server,
real handler, real files — no mocks.
"""

from __future__ import annotations

import asyncio
import resource
import shutil
import signal
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import IsolatedAsyncioTestCase

import kiss.agents.sorcar.persistence as th
from kiss.server.web_server import RemoteAccessServer

# Soft file-size limit during the faulted first attempt.  Must be
# comfortably larger than every incidental file the process may touch
# in that window (fresh sqlite db, log lines) yet far smaller than the
# huge base line the second hunk restores.
_FSIZE_LIMIT = 512 * 1024
_BIG_LINE = "x" * (4 * 1024 * 1024) + "\n"


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


def _two_hunk_entry(work: Path) -> tuple[dict[str, Any], str]:
    """Build a 2-hunk file entry whose first hunk has ``bc != cc``.

    Hunk 0: the agent collapsed 5 base lines into 2 (``bc=5, cc=2``).
    Hunk 1: the agent replaced one HUGE base line (the last line) with
    a small one; restoring it pushes the file far past
    :data:`_FSIZE_LIMIT`, so its write fails while the limit is low.

    Returns the merge file entry and the exact expected base content.
    """
    base_lines = [f"b{i}\n" for i in range(5)]
    ctx = [f"ctx{i}\n" for i in range(3)]
    base_text = "".join(base_lines + ctx) + _BIG_LINE
    cur_text = "c0\nc1\n" + "".join(ctx) + "small\n"
    current = work / "f.txt"
    base = work / "f_base.txt"
    current.write_text(cur_text)
    base.write_text(base_text)
    entry = {
        "name": "f.txt",
        "current": str(current),
        "base": str(base),
        "target": str(current),
        "hunks": [
            {"bs": 0, "bc": 5, "cs": 0, "cc": 2},
            {"bs": 8, "bc": 1, "cs": 5, "cc": 1},
        ],
    }
    return entry, base_text


class TestRejectRetryIdempotent(IsolatedAsyncioTestCase):
    """Retrying reject-all after a mid-file fault must not corrupt."""

    async def asyncSetUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp(prefix="kiss-bh9-a2-")
        self.saved = _redirect_persistence(self.tmpdir)
        self.work = Path(self.tmpdir) / "work"
        self.work.mkdir()
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=0, work_dir=str(self.work),
        )
        self.server._loop = asyncio.get_running_loop()
        # Ignore SIGXFSZ so an over-limit write returns EFBIG (a plain
        # OSError) instead of killing the test process.
        self._old_sig = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        self._old_limit = resource.getrlimit(resource.RLIMIT_FSIZE)

    async def asyncTearDown(self) -> None:
        resource.setrlimit(resource.RLIMIT_FSIZE, self._old_limit)
        signal.signal(signal.SIGXFSZ, self._old_sig)
        if th._db_conn is not None:
            th._db_conn.close()
        _restore_persistence(self.saved)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    async def test_reject_all_retry_after_midfile_fault_restores_base(
        self,
    ) -> None:
        """First attempt faults on hunk 1; the retry must yield BASE bytes."""
        entry, base_text = _two_hunk_entry(self.work)
        tab_id = "tab-a2-retry"
        self.server._register_merge_state(
            tab_id, {"work_dir": str(self.work), "files": [entry]},
        )
        with self.server._merge_states_lock:
            state = self.server._merge_states[tab_id]

        # --- attempt 1: hunk 0 splices fine, hunk 1's write hits EFBIG.
        soft, hard = self._old_limit
        resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_LIMIT, hard))
        try:
            await self.server._handle_web_merge_action({
                "type": "mergeAction", "action": "reject-all",
                "tabId": tab_id,
            })
        finally:
            resource.setrlimit(resource.RLIMIT_FSIZE, self._old_limit)

        # The fault must have kept every hunk unresolved (per-file
        # marking happens only after the whole file restored cleanly).
        self.assertEqual(state.remaining, 2)
        self.assertEqual(state.resolutions(), [])

        # --- attempt 2: the user clicks reject-all again.
        await self.server._handle_web_merge_action({
            "type": "mergeAction", "action": "reject-all", "tabId": tab_id,
        })

        final_text = (self.work / "f.txt").read_text()
        self.assertTrue(
            final_text == base_text,
            "BUG: retrying reject-all after a mid-file write fault "
            "re-applied the already-spliced hunk 0 with its stale cc, "
            "duplicating lines instead of restoring the base content "
            f"(got {len(final_text)} bytes, want {len(base_text)}; "
            f"head={final_text[:80]!r})",
        )
        # Both hunks resolved; the completed review was popped.
        with self.server._merge_states_lock:
            self.assertNotIn(tab_id, self.server._merge_states)


if __name__ == "__main__":
    unittest.main()
